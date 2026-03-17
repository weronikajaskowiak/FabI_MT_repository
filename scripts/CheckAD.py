import os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from scipy.spatial.distance import cityblock, cosine, braycurtis
from scipy.stats import wasserstein_distance as wasd

np.random.seed(2023)

BASESET = "ZINC12"
COMPARE = "other_test"

base_dir = "/scratch/project_2005393/weronika/ML_models/PCA_analysis"
out_dir  = "/scratch/project_2005393/weronika/ML_models/PCA_analysis/AD_tables"
os.makedirs(out_dir, exist_ok=True)

train_path = f"{base_dir}/training_fingerprints_{BASESET}.csv"
test_path  = f"{base_dir}/test_fingerprints_{BASESET}.csv"

OTHERSET = "Enamine" if BASESET == "ZINC12" else "ZINC12"
other_train_path = f"{base_dir}/training_fingerprints_{OTHERSET}.csv"
other_test_path  = f"{base_dir}/test_fingerprints_{OTHERSET}.csv"

def make_consensus(row): 
     return row.sum() >= 3 


def distances_to_center(X_pca: np.ndarray, center: np.ndarray) -> pd.DataFrame:
    euc = np.linalg.norm(X_pca - center, axis=1)
    man = np.sum(np.abs(X_pca - center), axis=1)
    cos = np.array([cosine(center, x) for x in X_pca])
    bray = np.array([braycurtis(center, x) for x in X_pca])
    wass = np.array([wasd(center, x) for x in X_pca])

    return pd.DataFrame({
        "euclidean": euc,
        "cityblock": man,
        "cosine": cos,
        "wasserstein": wass,
        "braycurtis": bray
    })


def flag_outside_by_counting(test_vals: np.ndarray, train_vals: np.ndarray, q: float = 0.95) -> np.ndarray:
    train_sorted = np.sort(train_vals)
    n = train_sorted.size
    counts = np.searchsorted(train_sorted, test_vals, side="left")  # strict ">"
    return counts > int(n * q)

X_train = pd.read_csv(train_path).astype(np.uint8)
X_test  = pd.read_csv(test_path).astype(np.uint8)
X_other_train = pd.read_csv(other_train_path).astype(np.uint8)
X_other_test  = pd.read_csv(other_test_path).astype(np.uint8)

X_test        = X_test.reindex(columns=X_train.columns)
X_other_train = X_other_train.reindex(columns=X_train.columns)
X_other_test  = X_other_test.reindex(columns=X_train.columns)

if COMPARE == "base_test":
    X_eval = X_test
    eval_name = f"{BASESET}_test"
elif COMPARE == "other_test":
    X_eval = X_other_test
    eval_name = f"{OTHERSET}_test_in_{BASESET}_space"
else:
    raise ValueError("COMPARE must be 'base_test' or 'other_test'")

pca = PCA(n_components=0.85, svd_solver="full")
X_train_pca = pca.fit_transform(X_train)
center = X_train_pca.mean(axis=0)
X_eval_pca = pca.transform(X_eval)

train_dist_df = distances_to_center(X_train_pca, center)
eval_dist_df  = distances_to_center(X_eval_pca, center)

methods = ["euclidean", "cityblock", "cosine", "wasserstein", "braycurtis"]

flags = {}
for m in methods:
    flags[m] = flag_outside_by_counting(
        eval_dist_df[m].to_numpy(),
        train_dist_df[m].to_numpy(),
        q=0.95
    )

AD_df = pd.DataFrame(flags)
AD_df["consensus"] = AD_df[methods].apply(make_consensus, axis=1)
AD_df["row_index"] = np.arange(len(AD_df))

out_path = f"{out_dir}/AD_{eval_name}_PCA85.xlsx"
AD_df.to_excel(out_path, index=False)
print("Saved:", out_path)