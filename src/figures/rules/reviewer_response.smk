include: "zero_shot_llm.smk"


rule reviewer_response_all:
    input:
        # Benchmark against text-only LLMs
        expand(rules.aggregate_zero_shot_llm_property_predictions.output.aggregated_predictions,
               metadata_col=["celltype"],  # NOTE for time reasons, we only run for `celltype`. Alternative: list(set([v for l in config["metadata_cols_per_zero_shot_validation_dataset"].values() for v in l])),
               # metadata_col=["Tissue", "celltype", "organ_tissue", "Disease_subtype"],  # same same TODO delete
               grouping=["by_cell"], # , "by_class"]
               ),

        # Perplexity-based reviewer responses
        expand(
            rules.llava_comparative_perplexity_plots.output.individual_performances,
            plot_metric=["log2_correct_ppl"],
            plot_type=["llava_cw_vs_geneformer", "text_only_vs_cw", "cw_preprompt_useless"],
            dataset=["main", "tabula_sapiens_100_cells_per_type"],  # only 'normal' datasets
        ),

        # Top 50 genes prediction (also perplexity-based)
        expand(PROJECT_DIR / config["paths"]["llava"]["evaluation_results"] / "{plot_name}.svg",
               plot_name=["detailed", "perplexity_quantile"],
               dataset=["main_top50genes", "tabula_sapiens_100_cells_per_type_top50genes"],
               base_model=[LLAVA_BASE_MODEL],
               model=[config["model_name_path_map"]["cellwhisperer_geneformer"], "geneformer"],  # config["model_name_path_map"]["cellwhisperer_uce"], "uce"],
               prompt_variation=["without50topgenes", "without50topgenesresponsepermuted"],
               llava_dataset=["_default"],
               ),
        # expand(
        #     rules.llava_comparative_perplexity_plots.output.individual_performances,
        #     plot_metric=["log2_correct_ppl"],
        #     plot_type=["llava_cw_vs_geneformer", "gene_predictability"],
        #     dataset=["main_top50genes", "tabula_sapiens_100_cells_per_type_top50genes"],  # only top50 datasets
        # ),
