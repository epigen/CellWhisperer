import pandas as pd

df = pd.read_csv(snakemake.input.metadata)
df = df[
    df["not_histology"] == 0
]  # Filter out non-histology images (it's only 5000 or so, but they are all really bad at least)
df = df[
    df["magnification"] == 2
]  # NOTE: this will imply only QUILT data (openpath etc. don't have magnification estimation)
# df = df[df['single_wsi'] == 0]  # I checked both options, but neither seems to be better than the other
df.to_csv(snakemake.output.metadata_filtered, index=False)
