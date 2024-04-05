Associated issue: https://github.com/epigen/cellwhisperer/issues/333

Logic:
- base_args defines everything
- the individual configs define delta parameters

We run every configuration 3 times (with 3 seeds and 3 LRs)

previous training concluded: LR 1e-4 to 1e-3



Note
Symlinking config.yaml here works, because it resolves to the real one and then also PROJECT_ROOT will be set correctly
