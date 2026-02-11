
TMA2: val
TMA3: test

# architecture
- CNN
- MPNN
- UUL (make sure the CNN is trained either way)
# data
- cosmx6k
- cellxgene_census-only
# hyperparameters (take the previous insights (biopod))
- ramp-up CNN - add context model - filter good quality cores - filter good quality TMAs

# For later (ignore for now)
- Data:
  - filter cellxgene census for lymphoma
  - take and prepare the lymphoma atlas from before (it's one dataset on cellxgene)
  - optional: archs4 filtered for lymphoma sub-species?
- consider metric: retrieval auroc *within single cores*
- follow-up: need better metric based on the lymphoma IHC datasets
## questions
- does performance across the two hold-outs correlate?
- does exclusion of the two "bad" TMAs improve performance?
