I have copied the h5ads to /dfs/user/moritzs/cellwhisperer_scoring_results. They contain a new column `.obs["celltype_cellwhisperer"]`.  Does that work for you? I paste an overview of the detected cell types below and attach the script used to compute the cell types.

Overall, it took me ~4 hours (about half of it was due to setting up a GPU-capable environment using uv on the SNAP cluster). I used AI for in-editor code completion (GH Copilot), and for selecting 10 relevant cell types per tissue.

Thank you for mentioning it in the acknowledgements section.
Indeed, I will be happy to contribute further, if there are tasks that fit my expertise. I don't have a good overview of the project, so it's hard to make "creative" contributions from my end currently. 

I hope all is going well for you and let me know if you have any questions! 
Moritz

Cell types overview:

=== BRAIN.H5AD - CellWhisperer Annotations ===   
Cell type distribution (CellWhisperer):          
  oligodendrocytes: 14,624 cells (62.3%)         
  ependymal cells: 5,427 cells (23.1%)           
  microglia: 2,225 cells (9.5%)
  astrocytes: 685 cells (2.9%)                   
  endothelial cells: 226 cells (1.0%)            
  pericytes: 199 cells (0.8%)                    
  glutamatergic neurons: 52 cells (0.2%)
  gabaergic neurons: 16 cells (0.1%)
  opcs: 2 cells (0.0%)

=== BREAST.H5AD - CellWhisperer Annotations ===
Cell type distribution (CellWhisperer):
  adipocytes: 4,386 cells (32.3%)
  basal cells: 3,583 cells (26.4%)
  endothelial cells: 1,731 cells (12.7%)
  luminal epithelial cells: 1,274 cells (9.4%)
  myoepithelial cells: 1,081 cells (8.0%)
  fibroblasts: 637 cells (4.7%)
  macrophages: 467 cells (3.4%)
  nk cells: 255 cells (1.9%)
  t cells: 91 cells (0.7%)
  b cells: 74 cells (0.5%)

=== GUT.H5AD - CellWhisperer Annotations ===
Cell type distribution (CellWhisperer):
  fibroblasts: 20,502 cells (32.6%)
  stem cells: 16,024 cells (25.5%)
  enterocytes: 10,300 cells (16.4%)
  tuft cells: 7,739 cells (12.3%)
  enteroendocrine cells: 2,795 cells (4.4%)
  transit amplifying cells: 2,629 cells (4.2%)
  m cells: 1,423 cells (2.3%)
  goblet cells: 925 cells (1.5%)
  macrophages: 431 cells (0.7%)
  paneth cells: 81 cells (0.1%)

=== HEART.H5AD - CellWhisperer Annotations ===
Cell type distribution (CellWhisperer):
  epicardial cells: 17,328 cells (28.6%)
  cardiomyocytes: 16,855 cells (27.8%)             
  endothelial cells: 9,209 cells (15.2%)           
  adipocytes: 5,318 cells (8.8%)                   
  macrophages: 3,712 cells (6.1%)
  fibroblasts: 2,831 cells (4.7%)                  
  schwann cells: 2,074 cells (3.4%)                
  t cells: 1,499 cells (2.5%)                      
  pericytes: 1,092 cells (1.8%)
  smooth muscle cells: 750 cells (1.2%)

=== LUNG.H5AD - CellWhisperer Annotations ===
Cell type distribution (CellWhisperer):
  type ii pneumocytes: 13,881 cells (56.2%)
  alveolar macrophages: 3,651 cells (14.8%)
  basal cells: 2,903 cells (11.8%)
  ciliated cells: 2,695 cells (10.9%)
  endothelial cells: 652 cells (2.6%)
  goblet cells: 378 cells (1.5%)
  type i pneumocytes: 215 cells (0.9%)
  t cells: 168 cells (0.7%)
  club cells: 117 cells (0.5%)
  fibroblasts: 20 cells (0.1%)

=== SKIN.H5AD - CellWhisperer Annotations ===
Cell type distribution (CellWhisperer):
  sebocytes: 6,425 cells (41.6%)
  dermal dendritic cells: 2,969 cells (19.2%)
  keratinocytes: 2,765 cells (17.9%)
  endothelial cells: 1,509 cells (9.8%)
  fibroblasts: 1,139 cells (7.4%)
  melanocytes: 237 cells (1.5%)
  t cells: 214 cells (1.4%)
  merkel cells: 87 cells (0.6%)
  langerhans cells: 69 cells (0.4%)
  macrophages: 43 cells (0.3%)
