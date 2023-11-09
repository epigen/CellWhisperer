import subprocess
from pathlib import Path
import logging
import yaml
import json
from typing import Union, List
import gseapy as gp


def write_enrichr_terms_to_json(
    terms_json_path: Union[str, Path],
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
        "PanglaoDB_Augmented_2021",
        "Tabula_Sapiens",
    ],
    organism: str = "Human",
) -> Union[None, dict]:
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
        terms[lib] = [
            f"{lib}: {term}"
            for term in list(gp.get_library(lib, organism=organism).keys())
        ]

    Path(terms_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(terms_json_path, "w") as f:
        json.dump(terms, f)

    return terms


# Load the config
pwd = Path(__file__).parent
PROJECT_DIR = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=pwd)
    .decode("utf-8")
    .strip()
)
with open(PROJECT_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

# Prepare the enrichr terms:
# TODO get team feedback on which libraries to include by default - list of all available libraries with 'gp.get_library_name()'
logging.info("Preparing EnrichR terms...")
terms_json_path = PROJECT_DIR / config["paths"]["enrichr_terms_json"]
write_enrichr_terms_to_json(terms_json_path=terms_json_path)
logging.info(f"EnrichR terms written to {terms_json_path}")

# Alternative example: Write only Tabula Sapiens data, with alternative prefix:
if False:
    terms_json_path = (
        PROJECT_DIR
        / "resources/enrichr_terms/terms.tabula_sapiens_human_adapted_prefix.json"
    )
    terms_dict = write_enrichr_terms_to_json(
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
