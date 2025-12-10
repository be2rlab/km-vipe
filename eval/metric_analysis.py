import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
import colorsys
import math

# Load all metrics.csv paths
metric_files = glob("results/osma-bench/*/*/*/metrics.csv", recursive=True)

# Output directory
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Collect data
records = []
for filepath in metric_files:
    parts = filepath.split("/")
    approach = parts[-4]
    dataset = parts[-3]
    label = parts[-2]

    df = pd.read_csv(filepath)
    for _, row in df.iterrows():
        records.append({
            "approach": approach,
            "dataset": dataset,
            "label": label,
            "scene": row["scene"],
            "miou": row["miou"],
            "fmiou": row["fmiou"],
            "macc": row["macc"]
        })

df = pd.DataFrame(records)

# Metrics to plot
metrics = ["miou", "fmiou", "macc"]

# Assign color per label
labels = sorted(df["label"].unique())
label_colors = {}
for i, label in enumerate(labels):
    hue = i / len(labels)
    lightness = 0.5
    saturation = 0.6
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    label_colors[label] = rgb

# Plot per dataset and metric
for dataset, dataset_df in df.groupby("dataset"):
    for metric in metrics:
        approaches = sorted(dataset_df["approach"].unique())
        n = len(approaches)

        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 4 * n), sharex=True)

        if n == 1:
            axes = [axes]

        for i, approach in enumerate(approaches):
            ax = axes[i]
            approach_df = dataset_df[dataset_df["approach"] == approach]

            for label in sorted(approach_df["label"].unique()):
                sub_df = approach_df[approach_df["label"] == label].sort_values(by="scene")
                ax.plot(
                    sub_df["scene"],
                    sub_df[metric],
                    marker='o',
                    label=label,
                    color=label_colors[label]
                )

            ax.set_title(f"{approach}")
            ax.set_ylabel(metric)
            ax.grid(True)
            ax.legend(title="Label", fontsize="small", title_fontsize="medium")
            ax.tick_params(axis='x', rotation=90)

        axes[-1].set_xlabel("Scene")
        fig.suptitle(f"{metric.upper()} - {dataset}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        filename = f"{dataset}_{metric}_by_approach.png"
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)

print("Saved vertically stacked subplot plots to 'plots/' folder.")
