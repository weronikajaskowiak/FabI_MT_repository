import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

np.random.seed(2023)

BASESET = "ZINC12"   # PCA reference space

base_dir = "/scratch/project_2005393/weronika/ML_models/PCA_analysis"
out_dir  = "/scratch/project_2005393/weronika/ML_models/PCA_analysis/plots_2"
os.makedirs(out_dir, exist_ok=True)

train_path = f"{base_dir}/training_fingerprints_{BASESET}.csv"
test_path  = f"{base_dir}/test_fingerprints_{BASESET}.csv"

OTHERSET = "Enamine" if BASESET == "ZINC12" else "ZINC12"

other_train_path = f"{base_dir}/training_fingerprints_{OTHERSET}.csv"
other_test_path  = f"{base_dir}/test_fingerprints_{OTHERSET}.csv"


X_train = pd.read_csv(train_path).astype(np.uint8)
X_test  = pd.read_csv(test_path).astype(np.uint8)

X_other_train = pd.read_csv(other_train_path).astype(np.uint8)
X_other_test  = pd.read_csv(other_test_path).astype(np.uint8)


X_test        = X_test.reindex(columns=X_train.columns)
X_other_train = X_other_train.reindex(columns=X_train.columns)
X_other_test  = X_other_test.reindex(columns=X_train.columns)


pca = PCA(n_components=3, svd_solver="full")

train_pca = pca.fit_transform(X_train)
test_pca  = pca.transform(X_test)

other_train_pca = pca.transform(X_other_train)
other_test_pca  = pca.transform(X_other_test)

variance = pca.explained_variance_ratio_ * 100


train_df = pd.DataFrame(train_pca, columns=["PC1", "PC2", "PC3"])
test_df  = pd.DataFrame(test_pca, columns=["PC1", "PC2", "PC3"])

other_train_df = pd.DataFrame(other_train_pca, columns=["PC1", "PC2", "PC3"])
other_test_df  = pd.DataFrame(other_test_pca, columns=["PC1", "PC2", "PC3"])


all_pc1 = np.concatenate([
    train_df.PC1, test_df.PC1,
    other_train_df.PC1, other_test_df.PC1
])

all_pc2 = np.concatenate([
    train_df.PC2, test_df.PC2,
    other_train_df.PC2, other_test_df.PC2
])

xlim = (all_pc1.min(), all_pc1.max())
ylim = (all_pc2.min(), all_pc2.max())



sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams.update({"font.size":16})


def plot_comparison(reference_df, compare_df,
                    label_reference, label_compare,
                    filename, title):

    fig, axes = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)

    # Left dataset
    sns.kdeplot(
        data=reference_df,
        x="PC1",
        y="PC2",
        fill=True,
        cmap="Blues",
        alpha=0.7,
        ax=axes[0]
    )

    axes[0].set_title(label_reference)

    # Right dataset
    sns.kdeplot(
        data=compare_df,
        x="PC1",
        y="PC2",
        fill=True,
        cmap="Reds",
        alpha=0.7,
        ax=axes[1]
    )

    axes[1].set_title(label_compare)

    for ax in axes:
        ax.set_xlabel(f'PC1 ({variance[0]:.2f}%)')
        ax.set_ylabel(f'PC2 ({variance[1]:.2f}%)')
        ax.set_xlim(xlim[0]-0.5, xlim[1]+0.5)
        ax.set_ylim(ylim[0]-0.5, ylim[1]+0.5)

    fig.suptitle(title)

    plt.tight_layout()

    plt.savefig(
        f"{out_dir}/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


plot_comparison(
    train_df,
    test_df,
    f"Train {BASESET}",
    f"Test {BASESET}",
    f"PCA_{BASESET}_train_vs_test.svg",
    f"PCA Chemical Space: Train vs Test {BASESET}"
)

# Train ZINC12 vs Train Enamine
plot_comparison(
    train_df,
    other_train_df,
    f"Train {BASESET}",
    f"Train {OTHERSET}_balanced",
    f"PCA_{BASESET}_train_vs_{OTHERSET}_balanced_train.svg",
    f"PCA Chemical Space: Train {BASESET} vs Train {OTHERSET}_balanced"
)

# Train ZINC12 vs Test Enamine
plot_comparison(
    train_df,
    other_test_df,
    f"Train {BASESET}",
    f"Test {OTHERSET}_balanced",
    f"PCA_{BASESET}_train_vs_{OTHERSET}_balanced_test.svg",
    f"PCA Chemical Space: Train {BASESET} vs Test {OTHERSET}_balanced"
)

# Enamine internal comparison
plot_comparison(
    other_train_df,
    other_test_df,
    f"Train {OTHERSET}_balanced",
    f"Test {OTHERSET}_balanced",
    f"PCA_{OTHERSET}_balanced_train_vs_test.svg",
    f"PCA Chemical Space: Train vs Test {OTHERSET}_balanced"
)