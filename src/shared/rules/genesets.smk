rule download_genesets:
    """
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
        from pathlib import Path
        import json
        enrichr_terms = {}

        for fn in input:
            enrichr_terms[Path(fn).stem] = [l.strip().split("\t")[0] for l in open(fn)]

        with open(output[0], "w") as f:
            json.dump(enrichr_terms, f)

