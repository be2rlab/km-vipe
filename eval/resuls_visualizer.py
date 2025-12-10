import os
import pandas as pd

root_dir = "results/osma-bench"

# Table rows will be stored here
rows = []

for approach in os.listdir(root_dir):
    approach_path = os.path.join(root_dir, approach)
    if not os.path.isdir(approach_path):
        continue
    for dataset in os.listdir(approach_path):
        dataset_path = os.path.join(approach_path, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for label in os.listdir(dataset_path):
            csv_path = os.path.join(dataset_path, label, "metrics.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)

            # Determine correct summary row based on dataset
            if dataset == "hm3d":
                summary_row = df[df["scene"] == "overall_mean"]
            elif dataset == "replica_cad":
                summary_row = df[df["scene"] == "overall"]
            else:
                continue

            # Count actual scenes (excluding summary rows)
            scene_count = df[~df["scene"].isin(["overall_mean", "overall"])].shape[0]

            if not summary_row.empty:
                miou = summary_row["miou"].values[0]
                fmiou = summary_row["fmiou"].values[0]
                macc = summary_row["macc"].values[0]

                rows.append({
                    "Approach": approach,
                    "Dataset": dataset,
                    "Label": label,
                    "Scene Count": scene_count,
                    "macc": (macc * 100).round(2),
                    "fmiou": (fmiou * 100).round(2),
                    # "miou": miou * 100,
                })

# Create DataFrame and display
summary_df = pd.DataFrame(rows)
summary_df = summary_df.sort_values(by=["Dataset", "Approach", "Label"])
print(summary_df.to_markdown(index=False))

print(summary_df)

#######################################3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set professional styling
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 14, 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})

# Start from your summary_df
df = summary_df.copy()

# Normalize approach and label for display
df['Approach'] = df['Approach'].replace({'conceptgraphs': 'ConceptGraphs', 'openscene': 'OpenScene', 'bbq': 'BBQ'})
df['Condition'] = df['Label'].replace({
    'baseline': 'Baseline',
    'camera_lights': 'Camera Light',
    'dynamic_lights': 'Dynamic Lights',
    'no_lights': 'Nominal Lights',
    'velocity': 'Velocity'
})

# Filter only replica_cad (or use hm3d instead)
replica_df = df[df['Dataset'] == 'replica_cad']

# Get baseline metrics for each approach
baselines = replica_df[replica_df['Label'] == 'baseline'].set_index('Approach')[['macc', 'fmiou']]

# Function to compute change
def compute_change(row):
    base = baselines.loc[row['Approach']]
    row['mAcc Change (%)'] = 100 * (row['macc'] - base['macc']) / base['macc']
    row['f-mIoU Change (%)'] = 100 * (row['fmiou'] - base['fmiou']) / base['fmiou']
    return row

# Apply change computation
change_df = replica_df[replica_df['Label'] != 'baseline'].apply(compute_change, axis=1)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
approach_order = ['ConceptGraphs', 'BBQ', 'OpenScene']

sns.barplot(data=change_df, x='Approach', y='mAcc Change (%)', hue='Condition', palette='Set2', ax=ax[0], order=approach_order)
# ax[0].set_title('mAcc Change (%)')
ax[0].set_xlabel('')
ax[0].set_ylabel('mAcc Change (%)', fontsize=20, fontweight='bold')
ax[0].axhline(0, color='black', linewidth=1)
ax[0].legend(prop={'weight': 'bold', 'size': 16}) 
ax[0].set_xticklabels(ax[0].get_xticklabels(), fontweight='bold', fontsize=18)
ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=16)

sns.barplot(data=change_df, x='Approach', y='f-mIoU Change (%)', hue='Condition', palette='Set2', ax=ax[1], order=approach_order)
# ax[1].set_title('f-mIoU Change (%)')
ax[1].set_xlabel('')
ax[1].set_ylabel('f-mIoU Change (%)', fontsize=20, fontweight='bold')
ax[1].axhline(0, color='black', linewidth=1)
ax[1].legend(prop={'weight': 'bold', 'size': 16})
ax[1].set_xticklabels(ax[1].get_xticklabels(), fontweight='bold', fontsize=18)
ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize=16)

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)

# Save plots
os.makedirs("plots", exist_ok=True)
fig.savefig("plots/change_barplot.png", dpi=300)
fig.savefig("plots/change_barplot.pdf")

plt.show()
