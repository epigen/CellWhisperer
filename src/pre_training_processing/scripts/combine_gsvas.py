import pandas as pd
from pathlib import Path

dfs = []
for f in snakemake.input:
    library = Path(f).stem.split("_", maxsplit=1)[1]
    df = pd.read_csv(f)
    df["library"] = library
    dfs.append(df)
pd.concat(dfs).to_parquet(snakemake.output[0], index=False)
