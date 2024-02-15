import subprocess
from pathlib import Path
import logging
import yaml
import json
from typing import Union, List
import gseapy as gp
import anndata

# Load the config
try:
    pwd = Path(__file__).parent
except NameError:
    pwd = "/home/moritz/Projects/cellwhisperer/src"
PROJECT_DIR = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=pwd)
    .decode("utf-8")
    .strip()
)
with open(PROJECT_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)


def load_enrichr_terms(
    selected_libraries: List[str] = [
        "Achilles_fitness_decrease",
        "Achilles_fitness_increase",
        "Azimuth_2023",
        "Disease_Perturbations_from_GEO_down",
        "Disease_Perturbations_from_GEO_up",
        "GO_Biological_Process_2023",
        "GO_Cellular_Component_2023",
        "GO_Molecular_Function_2023",
        "Gene_Perturbations_from_GEO_down",
        "Gene_Perturbations_from_GEO_up",
        "MSigDB_Hallmark_2020",
        "MSigDB_Oncogenic_Signatures",
        "OMIM_Disease",
        "OMIM_Expanded",
        "PanglaoDB_Augmented_2021",
        "Tabula_Sapiens",
    ],
    organism: str = "Human",
) -> dict:
    """
    Write the EnrichR terms to a json file. Keys are the selected libraries, values are the terms in the library.
    Example: library: "GO_Biological_Process_2021", terms: ["GO:0000001", "GO:0000002", ...]. \
        Stored as {"GO_Biological_Process_2021": ["GO_Biological_Process_2021: GO:0000001", "GO_Biological_Process_2021: GO:0000002", ...]}
    :param terms_json_path: Path to the json file to write the terms to.
    :param selected_libraries: List of EnrichR libraries to include.
    :param organism: Organism to use. Must be one of  ['Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm']
    :return: None
    """
    terms = {}
    for lib in selected_libraries:
        terms[lib] = list(gp.get_library(lib, organism=organism).keys())
    return terms


def load_tabsap_terms(
    tabsap_cols=[
        "organ_tissue",
        "anatomical_information",
        "gender",
        "cell_ontology_class",
        "free_annotation",
        "compartment",
    ]
):
    tabsap = anndata.read_h5ad(
        PROJECT_DIR / "results/tabula_sapiens/full_data.h5ad", backed="r"
    )
    terms = {}
    for col in tabsap_cols:
        terms[col] = list(tabsap.obs[col].unique())
    return terms


# Prepare the enrichr terms:
# TODO get team feedback on which libraries to include by default - list of all available libraries with 'gp.get_library_name()'
logging.info("Preparing EnrichR terms...")
enrichr_terms = load_enrichr_terms()

# load terms from tabsap as well

tabsap_terms = load_tabsap_terms()

terms_json_path = PROJECT_DIR / config["paths"]["enrichr_terms_json"]
Path(terms_json_path).parent.mkdir(parents=True, exist_ok=True)
with open(terms_json_path, "w") as f:
    json.dump({**tabsap_terms, **enrichr_terms}, f)

logging.info(f"Terms written to {terms_json_path}")


# Alternative example: Write only Tabula Sapiens data, with alternative prefix:
if False:
    terms_json_path = (
        PROJECT_DIR
        / "resources/enrichr_terms/terms.tabula_sapiens_human_adapted_prefix.json"
    )
    terms_dict = load_enrichr_terms(
        terms_json_path="/dev/null",
        selected_libraries=["Tabula_Sapiens"],
        organism="Human",
    )
    terms_dict = {
        key: [
            entry.replace("Tabula_Sapiens: ", "Transcriptome of a ") for entry in value
        ]
        for key, value in terms_dict.items()
    }
    with open(terms_json_path, "w") as f:
        json.dump(terms_dict, f)
