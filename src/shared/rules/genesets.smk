rule download_genesets:
    """
    cytopus from https://www.nature.com/articles/s41587-023-01940-3
    """
    output:
        geneset_gmt=PROJECT_DIR / config["paths"]["geneset_gmt"],
    conda:
        "../envs/download_genesets.yaml"
    script:
        "../scripts/download_genesets.py"

rule geneset_terms:
    """
    Integrate all terms into a common file
    """
    input:
        expand(rules.download_genesets.output.geneset_gmt, library=config["genesets"])
    output:
        PROJECT_DIR / config["paths"]["enrichr_terms_json"]
    run:
        import pdb; pdb.set_trace()

# TODO
# logging.info("Preparing EnrichR terms...")
# enrichr_terms = {lib: list(sets.keys()) for lib, sets in load_enrichr_terms().items()}

# # load terms from tabsap as well

# tabsap_terms = load_tabsap_terms()

# terms_json_path = PROJECT_DIR / config["paths"]["enrichr_terms_json"]
# Path(terms_json_path).parent.mkdir(parents=True, exist_ok=True)
# with open(terms_json_path, "w") as f:
#     json.dump({**tabsap_terms, **enrichr_terms}, f)
