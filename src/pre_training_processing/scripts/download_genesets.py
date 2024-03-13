from typing import List
import gseapy as gp
import pandas as pd
from pathlib import Path
import cytopus as cp
import numpy as np


def load_enrichr_terms(
    selected_library: str,
    organism: str = "Human",
) -> dict:
    """

    :param selected_library: EnrichR library.
    :param organism: Organism to use. Must be one of  ['Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm']
    :return: terms dict
    """
    terms = {
        f"{term}": "\t".join(genes)
        for term, genes in gp.get_library(selected_library, organism=organism).items()
    }

    return terms


def load_cytopus(library):
    G = cp.KnowledgeBase()

    terms = {}

    if library == "Cytopus_processes":
        for gs in G.processes.keys():
            gene_list = G.processes[gs]
            line = "\t".join(gene_list)
            terms[gs] = line

    elif library == "Cytopus_immune_identities":
        for gs in G.identities.keys():
            gene_list = G.identities[gs]
            line = "\t".join(gene_list)
            terms[gs] = line
    else:
        raise ValueError(f"Unknown library {library}")
    return terms


if snakemake.wildcards.library.startswith("Cytopus"):
    terms = load_cytopus(snakemake.wildcards.library)
else:
    terms = load_enrichr_terms(snakemake.wildcards.library)


# Save as GMT file
with open(snakemake.output.geneset_gmt, "w") as f:
    for term, genes in terms.items():
        f.write(f"{term}\t\t{genes}\n")
