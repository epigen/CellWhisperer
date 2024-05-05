## Downloading was done with

`GODEBUG=netdns=cgo rclone copy --drive-shared-with-me gdrive:forMoritz/moritz/joined_adata_30042024/harm_s1tos3_scvi5000genes_moritz.adata .`



## Current description by Salvo

The information that you should be interested in is:

.X = scvi-normalized values 

layer[‘raw’] = raw values of the corresponding number of genes (2500, 5000, 10000)

.obs[‘anno_og’] = original annotation

.obs[‘cell_cycle_phase’] = estimation of the cell cycle
 
.obs[‘umap_ddhodge_potential’] = potential calculated in the vector field of the umap space

.obs[‘anno_og_time’] = original annotation condensed by time

.obs[‘anno_new’] = our annotation after Leiden/potential clustering

.obs['dpt_pseudotime’] = it is the pseudotime calculated in respect to day1


Our clustering potentially needs some refinement and it might change. The pseudotime calculated in such a broad time window doesn’t make so much sense. I think the most interesting things for you are the: anno_og_time and the umap_ddhodge_potential.



## Previous description by Salvo


I try to explain a bit the name of the files:
s1, s2, s3 refers to the periods of embryo development:

S1: until blastocyst implantation
S2: until gastrulation
S3: from gastrulation

The files with scvi2500genes have the normalised	values with scvi for the top 2500 highly variable genes as adata.X


The files with raw_2500genes have the not-normalised (not batch corrected) values for the top 2500 highly variable genes as adata.X (those are integers).

The files with raw_totgenes have the not-normalised (not batch corrected) values for all genes as adata.X (those are integers).

Personally, I trust more the normalised values, but you can experiment a bit with them.

The annotations that we have generated have the original clusters each cell belongs to as anno_og, the cluster that we have identified as anno_new, and the time as anno_og_time.

In our annotations (anno_new), we have used up to the top 3 most likely cell types separated by “_”. For example, if we are sure that that cluster is only “erythroid”, then it will be “erythroid”, but if it is in between “erythroid” and “endothelium”, we will have “erythroid_endothelium”. They are ordered based on the frequency of cells for cluster (the first cell type is the most frequent and so on).


We can provide more information if you are interested in them…Some that I think might be potentially useful are: 
- estimated cell cycle phase for each cell
- potential in our manifold (the height of the landscape in our transcriptional manifold) for each cell
- marker genes (~500 genes) for each cluster
