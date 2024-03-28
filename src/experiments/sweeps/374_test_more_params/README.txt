Test more params

Associated issue: https://github.com/epigen/cellwhisperer/issues/374
- In this sweep, we fixed UL and enforce LL on the first epoch.
- We also test for increased batch size (and/or gradient accumulation) is helpful/relevant
- We also go for 5 epochs only

./slurm/run  cellwhisperer_sweeping --sweep_config ~/cellwhisperer/src/experiments/sweeps/374_test_more_params/ --sweep_id  t2m2vcn2
