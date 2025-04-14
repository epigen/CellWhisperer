This is the full notebook used for generating various versions of the GEO cluster exploratory analysis (Word count analysis based on metadata of cells in a particular cluster).

The current version (analysing the words that appear in the metadata of the most cells of a cluster) is in the beginning, older versions (including different metics) are further down (some of those are quite quick-and-dirty). 

Currently uses some files from hardcoded paths, so it's not fully reproducible.

Required additional modifications to include in paper pipeline:
- use relative paths
- specify clusters of interest via config
- remove old analyses
- Automate extraction of words with biomedical relevance (currently done via Gemini web app; results are hardcoded in the notebook)


(score_fit.py: Various tries to automate scoring how well cluster labels represent cell metadata. Doesn't work properly and not used at the moment.)
