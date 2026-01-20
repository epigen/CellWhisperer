"""
Centralized registry for all validation benchmarks.
Single source of truth for both training validation and Snakemake evaluation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from cellwhisperer.config import config


@dataclass
class ValidationBenchmarkSpec:
    """Specification for a validation benchmark task."""

    name: str
    dataset: str
    metadata_col: str
    dataset_kwargs: Dict[str, Any]
    processor_kwargs: Dict[str, Any]
    description: str
    category: str  # e.g., "celltype", "disease", "tissue"


class ValidationRegistry:
    """Registry of all validation benchmarks used across the project."""

    @staticmethod
    def get_cellwhisperer_benchmarks() -> List[ValidationBenchmarkSpec]:
        """
        Returns the comprehensive benchmark suite from fig2_embedding_validations.
        This is the single source of truth for all zero-shot evaluations.
        """
        return [
            # Tabula Sapiens - Celltype
            ValidationBenchmarkSpec(
                name="zshot_TabSap_celltype",
                dataset="tabula_sapiens",
                metadata_col="celltype",
                dataset_kwargs={
                    "cell_number_threshold_per_celltype": 100,
                    "celltype_obs_colname": "cell_ontology_class",
                },
                processor_kwargs={},
                description="Tabula Sapiens celltype prediction",
                category="celltype",
            ),
            # Tabula Sapiens - Well Studied Celltypes
            ValidationBenchmarkSpec(
                name="zshot_TabSapWellStudied_celltype",
                dataset="tabula_sapiens_well_studied_celltypes",
                metadata_col="celltype",
                dataset_kwargs={
                    "celltypes": config.get("top20_lung_liver_blood_celltypes", []),
                    "celltype_obs_colname": "cell_ontology_class",
                },
                processor_kwargs={},
                description="Tabula Sapiens well-studied celltypes",
                category="celltype",
            ),
            # Tabula Sapiens - Organ/Tissue
            ValidationBenchmarkSpec(
                name="zshot_TabSap_organ_tissue",
                dataset="tabula_sapiens",
                metadata_col="organ_tissue",
                dataset_kwargs={
                    "cell_number_threshold_per_celltype": 100,
                    "celltype_obs_colname": "organ_tissue",
                },
                processor_kwargs={},
                description="Tabula Sapiens organ/tissue prediction",
                category="tissue",
            ),
            # Human Disease - Disease Subtype
            ValidationBenchmarkSpec(
                name="zshot_HumanDisease_disease",
                dataset="human_disease",
                metadata_col="Disease",
                dataset_kwargs={"celltype_obs_colname": "Disease"},
                processor_kwargs={},
                description="Human disease prediction",
                category="disease",
            ),
            # Human Disease - Tissue
            ValidationBenchmarkSpec(
                name="zshot_HumanDisease_tissue",
                dataset="human_disease",
                metadata_col="Tissue",
                dataset_kwargs={"celltype_obs_colname": "Tissue"},
                processor_kwargs={},
                description="Human disease tissue prediction",
                category="tissue",
            ),
            # Pancreas
            ValidationBenchmarkSpec(
                name="zshot_Pancreas_celltype",
                dataset="pancreas",
                metadata_col="celltype",
                dataset_kwargs={"celltype_obs_colname": "celltype"},
                processor_kwargs={},
                description="Pancreas celltype prediction",
                category="celltype",
            ),
            # ImmGen
            ValidationBenchmarkSpec(
                name="zshot_ImmGen_celltype",
                dataset="immgen",
                metadata_col="celltype",
                dataset_kwargs={"celltype_obs_colname": "celltype"},
                processor_kwargs={},
                description="ImmGen celltype prediction",
                category="celltype",
            ),
            # AIDA
            ValidationBenchmarkSpec(
                name="zshot_AIDA_celltype",
                dataset="aida",
                metadata_col="celltype",
                dataset_kwargs={"celltype_obs_colname": "celltype"},
                processor_kwargs={},
                description="AIDA celltype prediction",
                category="celltype",
            ),
        ]

    @staticmethod
    def get_benchmark_by_name(name: str) -> Optional[ValidationBenchmarkSpec]:
        """Get a specific benchmark by name."""
        benchmarks = ValidationRegistry.get_cellwhisperer_benchmarks()
        return next((b for b in benchmarks if b.name == name), None)

    @staticmethod
    def get_benchmarks_by_category(category: str) -> List[ValidationBenchmarkSpec]:
        """Get benchmarks filtered by category."""
        benchmarks = ValidationRegistry.get_cellwhisperer_benchmarks()
        return [b for b in benchmarks if b.category == category]

    @staticmethod
    def get_benchmark_names() -> List[str]:
        """Get all benchmark names for use in Snakemake wildcard constraints."""
        return [b.name for b in ValidationRegistry.get_cellwhisperer_benchmarks()]

    @staticmethod
    def get_dataset_metadata_pairs() -> List[tuple]:
        """Get all (dataset, metadata_col) pairs for Snakemake expansion."""
        benchmarks = ValidationRegistry.get_cellwhisperer_benchmarks()
        return [(b.dataset, b.metadata_col) for b in benchmarks]
