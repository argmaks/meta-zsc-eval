# %%
import pickle
import numpy as np
from pathlib import Path

# Load the saved xxp data
load_path = Path(__file__).parent / "results" / "xxp" / "xxp_results-rnn-details.pkl"

with open(load_path, 'rb') as f:
    data = pickle.load(f)

# Cast xxp to numpy array
xxp = np.array(data['xxp']).mean(-1)

# Print some basic info
print(f"Loaded data with {len(data['sample_names'])} samples")
print(f"XXP matrix shape: {xxp.shape}")
print(f"XXP matrix dtype: {xxp.dtype}")

# %%
# xp results

xxp_means = np.zeros((xxp.shape[0], xxp.shape[1]))
xxp_stds = np.zeros((xxp.shape[0], xxp.shape[1]))
xxp_tols = np.zeros((xxp.shape[0], xxp.shape[1]))



# %%
sxp_mask = np.eye(xxp.shape[0], dtype=bool)
xp_mask = np.eye(xxp.shape[2], dtype=bool)

xxp_means[sxp_mask] = xxp[sxp_mask][:, ~xp_mask].mean(axis=(1))
xxp_stds[sxp_mask] = xxp[sxp_mask][:, ~xp_mask].std(axis=(1))
xxp_tols[sxp_mask] = 1.96 * xxp_stds[sxp_mask] / np.sqrt(xxp[sxp_mask][:, ~xp_mask].reshape(xxp[sxp_mask].shape[0], -1).shape[1])

# %%
xxp_means[~sxp_mask] = xxp[~sxp_mask].mean(axis=(1,2))
xxp_stds[~sxp_mask] = xxp[~sxp_mask].std(axis=(1,2))
xxp_tols[~sxp_mask] = 1.96 * xxp_stds[~sxp_mask] / np.sqrt(xxp[~sxp_mask].reshape(xxp[~sxp_mask].shape[0], -1).shape[1])

# %%
xxp_means
# %%
xxp_tols
# %%
xxp_stds
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set seaborn theme
sns.set_theme()

# Flag to control whether to sort samples by XXP scores
SORT_BY_XXP = True

# Extract sample names
sample_names = data['sample_names']
n_samples = len(sample_names)

# Optionally sort by XXP scores (diagonal values)
if SORT_BY_XXP:
    # Get diagonal values (XXP scores) and sort indices in descending order
    xxp_scores = np.diag(xxp_means)
    sorted_indices = np.argsort(xxp_scores)[::-1]  # Sort descending
    
    # Reorder the matrices and sample names according to sorted XXP scores
    xxp_means_sorted = xxp_means[sorted_indices, :][:, sorted_indices]
    xxp_tols_sorted = xxp_tols[sorted_indices, :][:, sorted_indices]
    sample_names_sorted = [sample_names[i] for i in sorted_indices]
else:
    # Use original order
    xxp_means_sorted = xxp_means
    xxp_tols_sorted = xxp_tols
    sample_names_sorted = sample_names

# Create a pandas DataFrame for the crossplay matrix
crossplay_df = pd.DataFrame(xxp_means_sorted, index=sample_names_sorted, columns=sample_names_sorted)
tol_df = pd.DataFrame(xxp_tols_sorted, index=sample_names_sorted, columns=sample_names_sorted)

# Create annotations with mean ± tolerance
annot_array = np.empty(crossplay_df.shape, dtype=object)
for row_idx in range(crossplay_df.shape[0]):
    for col_idx in range(crossplay_df.shape[1]):
        mean_val = crossplay_df.iloc[row_idx, col_idx]
        tol_val = tol_df.iloc[row_idx, col_idx]
        annot_array[row_idx, col_idx] = f'{mean_val:.2f}±{tol_val:.2f}'

# Set fixed color scale range
vmin, vmax = xxp_means.min(), xxp_means.max()
print(f"Value range: {vmin:.3f} to {vmax:.3f}")

# Create the heatmap
fig, ax = plt.subplots(figsize=(16, 14))

sns.heatmap(
    crossplay_df, 
    ax=ax,
    annot=annot_array,
    fmt='',
    cbar=True,
    vmin=vmin,
    vmax=vmax,
    cmap='RdYlGn',
    linewidths=0.5,
    square=True,
    cbar_kws={'label': 'Average Reward'}
)

# Set labels
ax.set_xlabel('Sample Names', fontsize=12)
ax.set_ylabel('Sample Names', fontsize=12)
ax.set_title('Cross-Play Performance Matrix\n(Mean ± 95% CI)', fontsize=16, fontweight='bold', pad=20)

# Rotate labels for better readability
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)

# Fix label alignment
for label in ax.get_xticklabels():
    label.set_horizontalalignment('right')

plt.tight_layout()
plt.show()

# %%
