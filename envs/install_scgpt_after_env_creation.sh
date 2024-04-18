# Changes specifc for peter's server (different cuda version)
# mamba install -c pytorch -c conda-forge -c nvidia -c defaults "pytorch-cuda ==11.8"
# mamba uninstall torchaudio torchtext
# mamba install -c conda-forge -c pytorch torchaudio torchtext

pip install "flash-attn<1.0.5" --no-build-isolation

# v0.1.9, but with fix suggested in here: https://github.com/bowang-lab/scGPT/issues/69
pip install --no-deps scgpt==0.1.9


# potentially useful: https://github.com/bowang-lab/scGPT/issues/15#issuecomment-1791120487
# or https://github.com/bowang-lab/scGPT/issues/69#issuecomment-1737520314


# TODO expand this to also install cellxgene
