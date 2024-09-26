# Instructions

As described in the CellWhisperer project README, do the following

- Place your dataset in `cellwhisperer/resources/resources/<dataset_name>/read_count_table.h5ad`
- Edit the `docker-compose.yaml` within this folder to reflect the correct `<dataset_name>`
- Run `docker-compose up`

Note: you'll need to switch off the cellwhisperer llava hosting in the meanwhile, because the GPU RAM is not large enough for 2x LLaVA
