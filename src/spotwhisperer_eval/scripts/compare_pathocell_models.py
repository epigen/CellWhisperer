#!/usr/bin/env python3
"""
Statistical comparison between two PathoCell models based on aggregated *_values metrics
and per-class metrics across datasets/seeds.

- Performs t-tests (paired if lengths match, else independent) for all keys ending with `_values`
  in the model summary JSONs.
- Computes Cohen's d (independent) or standardized mean difference (paired) as effect size.
- Identifies the metric with the most pronounced difference.
- If per-class CSVs are provided, computes per-class t-tests and finds the class with strongest difference.

Assumes Snakemake provides inputs/outputs/params.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

# Inputs via snakemake
model_a_summary_path = Path(snakemake.input.model_a_summary)
model_b_summary_path = Path(snakemake.input.model_b_summary)
per_class_a_files = [Path(p) for p in getattr(snakemake.input, "per_class_a", [])]
per_class_b_files = [Path(p) for p in getattr(snakemake.input, "per_class_b", [])]

out_metrics_csv = Path(snakemake.output.metrics_comparison_csv)
out_classes_csv = Path(snakemake.output.per_class_comparison_csv)
out_summary_json = Path(snakemake.output.summary_json)

model_a_name = snakemake.params.model_a
model_b_name = snakemake.params.model_b
prediction_level = snakemake.params.prediction_level

# Load summaries
with open(model_a_summary_path) as f:
    a_summary = json.load(f)
with open(model_b_summary_path) as f:
    b_summary = json.load(f)

# Compare *_values metrics
metrics_rows = []
best_metric = None
best_effect = -np.inf

for key in sorted(a_summary.keys()):
    if not key.endswith("_values"):
        continue
    if key not in b_summary:
        continue
    a_vals = np.array(a_summary[key], dtype=float)
    b_vals = np.array(b_summary[key], dtype=float)

    paired = len(a_vals) == len(b_vals)
    if paired:
        t_stat, p_val = ttest_rel(a_vals, b_vals)
        diffs = a_vals - b_vals
        effect = float(np.mean(diffs) / (np.std(diffs, ddof=1) if np.std(diffs, ddof=1) != 0 else 1.0))
        test_type = "paired"
    else:
        t_stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
        # Cohen's d (Hedges' g not necessary for large N)
        s1 = np.var(a_vals, ddof=1)
        s2 = np.var(b_vals, ddof=1)
        n1 = len(a_vals)
        n2 = len(b_vals)
        sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
        effect = float((np.mean(a_vals) - np.mean(b_vals)) / (sp if sp != 0 else 1.0))
        test_type = "independent"

    mean_a = float(np.mean(a_vals))
    mean_b = float(np.mean(b_vals))
    delta = float(mean_a - mean_b)

    metrics_rows.append({
        "metric": key,
        "model_a": model_a_name,
        "model_b": model_b_name,
        "prediction_level": prediction_level,
        "n_a": len(a_vals),
        "n_b": len(b_vals),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta_mean": delta,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "effect_size": effect,
        "test_type": test_type,
        "paired": paired,
    })

    if abs(effect) > best_effect:
        best_effect = abs(effect)
        best_metric = {
            "metric": key,
            "delta_mean": delta,
            "effect_size": effect,
            "p_value": float(p_val),
            "test_type": test_type,
            "paired": paired,
        }

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.sort_values(["effect_size"], key=lambda s: s.abs(), ascending=False, inplace=True)
out_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
metrics_df.to_csv(out_metrics_csv, index=False)

# Per-class comparison if files provided
classes_rows = []
best_class = None
best_class_effect = -np.inf

if per_class_a_files and per_class_b_files:
    # Load and stack per-class metrics
    def load_per_class(files):
        dfs = []
        for fp in files:
            df = pd.read_csv(fp)
            # Expect a column for class label; try common names
            label_col = None
            for cand in ["class_label", "label", "cell_type", "class", "target"]:
                if cand in df.columns:
                    label_col = cand
                    break
            # Minimal normalization: keep numeric metric columns
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            keep_cols = [label_col] + num_cols if label_col else num_cols
            df = df[keep_cols].copy()
            if label_col:
                df.rename(columns={label_col: "class_label"}, inplace=True)
            else:
                df["class_label"] = "unknown"
            df["source_file"] = fp.name
            dfs.append(df)
        return pd.concat(dfs, axis=0, ignore_index=True)

    a_classes = load_per_class(per_class_a_files)
    b_classes = load_per_class(per_class_b_files)

    # Determine shared metric columns
    metric_cols = [c for c in a_classes.columns if c not in ["class_label", "source_file"]]
    metric_cols = [c for c in metric_cols if c in b_classes.columns]

    # For each class and metric, test differences
    for cls in sorted(set(a_classes.class_label) & set(b_classes.class_label)):
        a_sub = a_classes[a_classes.class_label == cls]
        b_sub = b_classes[b_classes.class_label == cls]
        for m in metric_cols:
            a_vals = a_sub[m].to_numpy(dtype=float)
            b_vals = b_sub[m].to_numpy(dtype=float)
            paired = len(a_vals) == len(b_vals)
            if paired:
                t_stat, p_val = ttest_rel(a_vals, b_vals)
                diffs = a_vals - b_vals
                effect = float(np.mean(diffs) / (np.std(diffs, ddof=1) if np.std(diffs, ddof=1) != 0 else 1.0))
                test_type = "paired"
            else:
                t_stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
                s1 = np.var(a_vals, ddof=1)
                s2 = np.var(b_vals, ddof=1)
                n1 = len(a_vals)
                n2 = len(b_vals)
                sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
                effect = float((np.mean(a_vals) - np.mean(b_vals)) / (sp if sp != 0 else 1.0))
                test_type = "independent"
            mean_a = float(np.mean(a_vals))
            mean_b = float(np.mean(b_vals))
            delta = float(mean_a - mean_b)
            row = {
                "class_label": cls,
                "metric": m,
                "model_a": model_a_name,
                "model_b": model_b_name,
                "prediction_level": prediction_level,
                "n_a": len(a_vals),
                "n_b": len(b_vals),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "delta_mean": delta,
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "effect_size": effect,
                "test_type": test_type,
                "paired": paired,
            }
            classes_rows.append(row)
            if abs(effect) > best_class_effect:
                best_class_effect = abs(effect)
                best_class = {
                    "class_label": cls,
                    "metric": m,
                    "delta_mean": delta,
                    "effect_size": effect,
                    "p_value": float(p_val),
                    "test_type": test_type,
                    "paired": paired,
                }

if classes_rows:
    classes_df = pd.DataFrame(classes_rows)
    classes_df.sort_values(["effect_size"], key=lambda s: s.abs(), ascending=False, inplace=True)
    out_classes_csv.parent.mkdir(parents=True, exist_ok=True)
    classes_df.to_csv(out_classes_csv, index=False)

# Summary JSON
summary = {
    "model_a": model_a_name,
    "model_b": model_b_name,
    "prediction_level": prediction_level,
    "top_metric": best_metric,
    "top_class_difference": best_class,
    "n_metrics_compared": int(len(metrics_rows)),
    "n_classes_compared": int(len(classes_rows)),
}

out_summary_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_summary_json, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
