import os
import random
import json

import numpy as np
import pandas as pd

import scanpy as sc
import anndata as ad

import mygene
import pickle

def gpt_sample_guess_from_ranked_genes(sample1, sample2, 
                                       anndata_generanks, 
                                       system_prompt, 
                                       gpt_model = "gpt-3.5-turbo", N = 100):
    sample1_genelist = get_sample_ranked_genes(sample1, anndata_generanks, N)
    sample2_genelist = get_sample_ranked_genes(sample2, anndata_generanks, N)

    gene_list_1 = ", ".join(sample1_genelist["symbol"].tolist())
    gene_list_2 = ", ".join(sample2_genelist["symbol"].tolist())

    choice = random.choice([1, 2])

    if choice == 1:
        user_content = f"""The sample annotation is:
     {annots[sample1]}
     
     gene list 1:
     {gene_list_1}

     gene list 2:
     {gene_list_2}
     """

    else:
        user_content = f"""The sample annotation is:
     {annots[sample_2]}
     
     gene list 1:
     {gene_list_1}

     gene list 2:
     {gene_list_2}
     """

    gpt_response = client.chat.completions.create(
      model=gpt_model,
      messages=[
          {"role": "system", "content": f"{system_prompt}"},
          {"role": "user", "content": f"{user_content}"}
           ] )

    gpt_out = gpt_response.choices[0].message.content.split("\n")

    try:
        output = {
            "sample1" : sample1,
            "sample2" : sample2,
            "choice" : choice,
            "match" : choice == int(gpt_out[1]),
            "reason" : gpt_out[0]
        }
    except (IndexError, ValueError): ## This happens when the second line is not provided and sample is not provided.
        output = {
            "sample1" : sample1,
            "sample2" : sample2,
            "choice" : choice,
            "match" : None,
            "reason" : gpt_out[0]
        }
    
    return({"gpt_response" : gpt_response, "output" : output})


## add an argument to switch between structured and processed annotations
## This can be achieved by providing processed annotations instead of annotations

def get_user_input(sid, anns, grs, keywords, N = 100):
    
    annot_input = anns [sid]

    ont_terms = grs.obs.mapped_ontology_terms[sid]

    smp_keywords = keywords[sid]

    top_genes = get_sample_ranked_genes(sid, grs, N)
    top_genes_str = ", ".join(top_genes["symbol"].tolist())
    
    out = f"""Sample annotation:
{annot_input}
     
Ontology terms:
{ont_terms}
    
The similarity scores of the transcriptome embedding of the sample to Enrichr terms:   
{smp_keywords}
    
Upregulated genes:
{top_genes_str}
    """

    return(out)


def get_sample_ranked_genes(sid, anndata_generanks, N = 200):
    sample_gene_ranks = anndata_generanks[sid].X.toarray()
    sorted_indices = np.argsort(sample_gene_ranks).flatten()
    # print(anndata_generanks[sid].layers["gene_ranks"][0, sorted_indices[:N]])
    out_genes = anndata_generanks.var.iloc[sorted_indices[:N]]
    return(out_genes)


