#!/usr/bin/env python
"""
Aggregate comprehensive benchmark results from the centralized validation registry.
This script processes individual benchmark outputs and creates a unified summary
for integration with the spider plot.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def aggregate_comprehensive_benchmarks():
    """
    Aggregate benchmark results from all comprehensive evaluations.
    """
    
    # Get parameters from snakemake
    dataset_metadata_pairs = snakemake.params.dataset_metadata_pairs
    comprehensive_benchmarks = snakemake.params.comprehensive_benchmarks
    benchmark_results = snakemake.input.benchmark_results
    output_path = snakemake.output.aggregated_comprehensive
    
    logger.info(f"Aggregating {len(benchmark_results)} comprehensive benchmark results")
    logger.info(f"Dataset-metadata pairs: {len(dataset_metadata_pairs)}")
    
    all_results = []
    
    # Process each benchmark result file
    for i, result_file in enumerate(benchmark_results):
        try:
            # Load the performance metrics
            metrics_df = pd.read_csv(result_file, index_col=0)
            
            # Get corresponding benchmark info
            if i < len(dataset_metadata_pairs):
                dataset, metadata_col = dataset_metadata_pairs[i]
            else:
                # Fallback: parse from file path
                result_path = Path(result_file)
                dataset = result_path.parent.parent.name
                metadata_col = result_path.parent.name
                logger.warning(f"Using fallback parsing for {result_file}: {dataset}/{metadata_col}")
            
            # Find the benchmark spec for additional info
            benchmark_spec = None
            for spec in comprehensive_benchmarks:
                if spec.dataset == dataset and spec.metadata_col == metadata_col:
                    benchmark_spec = spec
                    break
            
            if benchmark_spec is None:
                logger.warning(f"No benchmark spec found for {dataset}/{metadata_col}")
                benchmark_name = f"zshot_{dataset}_{metadata_col}"
                category = "unknown"
                description = f"{dataset} {metadata_col} prediction"
            else:
                benchmark_name = benchmark_spec.name
                category = benchmark_spec.category  
                description = benchmark_spec.description
            
            # Create a record for this benchmark
            result_record = {
                'benchmark_name': benchmark_name,
                'dataset': dataset,
                'metadata_col': metadata_col,
                'category': category,
                'description': description,
            }
            
            # Add all performance metrics
            for metric_name in metrics_df.index:
                metric_value = metrics_df.loc[metric_name, 'value'] if 'value' in metrics_df.columns else metrics_df.iloc[0, 0]
                result_record[metric_name] = metric_value
            
            all_results.append(result_record)
            
            logger.info(f"Processed {benchmark_name}: {description}")
            
        except Exception as e:
            logger.error(f"Error processing {result_file}: {e}")
            continue
    
    # Create aggregated DataFrame
    if all_results:
        aggregated_df = pd.DataFrame(all_results)
        
        # Set index to benchmark_name for easy lookup
        aggregated_df.set_index('benchmark_name', inplace=True)
        
        # Save aggregated results
        aggregated_df.to_csv(output_path)
        
        logger.info(f"Aggregated results saved to {output_path}")
        logger.info(f"Summary statistics:")
        logger.info(f"  - Total benchmarks: {len(aggregated_df)}")
        
        # Log summary by category
        if 'category' in aggregated_df.columns:
            category_counts = aggregated_df['category'].value_counts()
            for category, count in category_counts.items():
                logger.info(f"  - {category}: {count} benchmarks")
        
        # Log key metrics if available
        key_metrics = ['accuracy', 'f1', 'rocauc']
        for metric in key_metrics:
            if metric in aggregated_df.columns:
                mean_val = aggregated_df[metric].mean()
                std_val = aggregated_df[metric].std()
                logger.info(f"  - {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Log per-category averages for key metrics
        if 'category' in aggregated_df.columns:
            logger.info("\nPer-category averages:")
            for category in aggregated_df['category'].unique():
                category_data = aggregated_df[aggregated_df['category'] == category]
                logger.info(f"  {category} ({len(category_data)} benchmarks):")
                for metric in key_metrics:
                    if metric in category_data.columns:
                        mean_val = category_data[metric].mean()
                        logger.info(f"    - {metric}: {mean_val:.4f}")
                        
    else:
        logger.error("No benchmark results could be processed!")
        # Create empty DataFrame to avoid downstream errors
        aggregated_df = pd.DataFrame()
        aggregated_df.to_csv(output_path)
    
    logger.info("Comprehensive benchmark aggregation complete")


if __name__ == "__main__":
    aggregate_comprehensive_benchmarks()