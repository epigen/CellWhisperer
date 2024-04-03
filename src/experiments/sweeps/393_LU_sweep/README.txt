Test more params

Associated issue: https://github.com/epigen/cellwhisperer/issues/393
- In this sweep, we only use LU (as it showed to be the best
- Thus, we can also use BS: 512
- increase LR accordingly: 1e-5 to 1e3 is the learning from the previous sweep. Now we can go 1e-2 to 1e-5 (we train more epochs, so the 10x works nicely)
- 16 epochs

TODO:
- make sure that BS512 improves upon previous sweep (compare to https://github.com/epigen/cellwhisperer/issues/374)
- make sure that the learning does not deteriorate with the increased number of epochs


./slurm/run  cellwhisperer_sweeping --sweep_config ~/cellwhisperer/src/experiments/sweeps/393_LU_sweep/ --sweep_id  cjeqxzch
