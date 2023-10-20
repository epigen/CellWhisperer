# single-cellm
single ceLLM project

For more information on project management, follow
https://github.com/epigen/single-cellm/wiki

## Installation instructions

1. Install Git LFS. This can usually be done with the command `git lfs install`. They only need to do this once per machine.
2. Clone/Pull this repository
3. Develop :)

## Folder structure

- data: Computationally non-reproducible, expensive, or painful to reproduce
- metadata: Computationally non-reproducible, e.g., sample annotation sheets, clinical annotation
- results: Can be reproduced with your scripts and pipelines
- resources: External, references, datasets and tools that are project inherent and can be reproduced or downloaded with your scripts and pipelines
- src (and all other directories needed to run the source code)
