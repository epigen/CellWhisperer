import os
import glob
import pandas as pd

# Define the pattern to search for CSV files
os.chdir("/home/peter/peter_on_isilon/cellwhisperer/single-cellm")
pattern = 'results/MS_zero_shot_scfoundation_replication_v2_pancreas_renamed/cellwhisperer_clip_v1_*/*/performance_metrics_cellwhisperer.celltype_as_label.macrovag.csv'
pattern2 = 'results/MS_zero_shot_scfoundation_replication_v2_pancreas_renamed/f6fjywkb.ckpt/*/performance_metrics_cellwhisperer.celltype_as_label.macrovag.csv'
pattern3 = 'results/MS_zero_shot_scfoundation_replication_v2/f6fjywkb.ckpt/*/performance_metrics_cellwhisperer.celltype_as_label.macrovag.csv'

# Use glob to get a list of file paths matching the pattern
file_paths = glob.glob(pattern)+glob.glob(pattern2)+glob.glob(pattern3)

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each file and read it into a DataFrame
for file_path in file_paths:
    # Use try-except to handle cases where reading the file fails
    try:
        df = pd.read_csv(file_path)
        # Add a new column 'dirname' with the directory name
        df['dataset'] = os.path.basename(os.path.dirname(file_path))
        df["run"] = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        df["0"]=df["0"].str.replace("tensor(","").str.replace(")","").astype(float)
        dfs.append(df)
    except pd.errors.EmptyDataError:
        print(f"Empty file: {file_path}")

# Concatenate all DataFrames in the list horizontally (sideways)
combined_df = pd.concat(dfs, axis=0)

combined_df.columns=["metric","value","dataset","run"]

# drop duplicates with same run, dataset, metric combination
combined_df=combined_df.drop_duplicates(subset=["run","dataset","metric"])

# pivot by metric
combined_df=combined_df.pivot(index=["run","dataset"],columns="metric",values="value")
# sort by dataset
combined_df=combined_df.sort_values(by="dataset")


tabsap_wellstudied=combined_df[combined_df.index.get_level_values("dataset")=="tabula_sapiens_100_cells_per_type_well_studied_celltypes"]
tabsap_wellstudied=tabsap_wellstudied.reset_index(level="dataset",drop=True)
tabsap_wellstudied

pancreas=combined_df[combined_df.index.get_level_values("dataset")=="pancreas"]
pancreas=pancreas.reset_index(level="dataset",drop=True)
pancreas

tabsap_min100=combined_df[combined_df.index.get_level_values("dataset")=="tabula_sapiens_100_cells_per_type_min_100"]
tabsap_min100=tabsap_min100.reset_index(level="dataset",drop=True)
tabsap_min100