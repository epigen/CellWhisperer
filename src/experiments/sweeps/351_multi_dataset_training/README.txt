Multiple dataset training

https://github.com/epigen/cellwhisperer/issues/351



- First, we test here a single batch_size (to accommodate memory for all model types)
- In a next sweep, we can test whether increased batch size (and/or gradient accumulation) is helpful/relevant


~/cellwhisperer/slurm/run  cellwhisperer_sweeping --sweep_config ~/cellwhisperer/src/experiments/sweeps/351_multi_dataset_training/ --sweep_id  12spml6j
