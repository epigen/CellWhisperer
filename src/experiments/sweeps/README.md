Semi-structured folder for sweep-based experiments (primarily config files).

For a new experiment, create a new folder with an appropriate name. Within it create 

- README.md (document your experiment)
- base_config.yaml  # generate with a run to `single_cellm fit --print_config`
- sweep_config.yaml  # your sweep config


checkout `broad_initial` for an example
