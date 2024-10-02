import gseapy as gp


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


terms = load_enrichr_terms(snakemake.wildcards.library)  # type: ignore [reportUndefinedVariable]

# Save as GMT file
with open(snakemake.output.geneset_gmt, "w") as f:  # type: ignore [reportUndefinedVariable]
    for term, genes in terms.items():
        f.write(f"{term}\t\t{genes}\n")
