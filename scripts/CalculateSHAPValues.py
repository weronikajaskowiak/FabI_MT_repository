import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import descriptors as desc
from MASSA_Algorithm import MASSAopen_files
from sklearn.pipeline import Pipeline
from rdkit.Chem import MolToSmiles

# -----------------------------
# Files paths
# -----------------------------

descriptors_dir = "/scratch/project_2005393/weronika/ML_models/SHAP_values_for_found_model/SHAP_ZINC12/descriptors_ZINC12"
model_path = "/scratch/project_2005393/weronika/ML_models/SHAP_values_for_found_model/SHAP_ZINC12/model_SVM/model_trial_1000.pkl"
out_dir = "/scratch/project_2005393/weronika/ML_models/SHAP_values_for_found_model/SHAP_ZINC12"
os.makedirs(out_dir, exist_ok=True)

job_name = "4FS3_zinc12dlc90_no_duplicates_after_MASSA_pubchem"

# How many samples to use (tune for runtime)
N_BG = 50          # background samples from train
N_EXPLAIN = 50      # samples to explain from test
MAX_DISPLAY = 10    # top features to display in plots / prints

# -----------------------------
# Load descriptors 
# -----------------------------

descriptor_path = os.path.join(
    descriptors_dir,
    f"{job_name}_fp-rpca.pkl"
)

print("Loading descriptors from:")
print(descriptor_path)
descriptor = desc.Descriptors.load_fp_robustpca(descriptor_path)
#print("First element type:", type(descriptor.file[0])) #reassurance that full MASSA molecule list is in descriptors file 

 # -----------------------------
 # Load trained model
 # -----------------------------

model = joblib.load(model_path)
print("Loaded model:", type(model))

 # -----------------------------
 # Build a pipeline 
 # f(raw bits)=model(PCA(scaler(raw bits)))
 # -----------------------------
if not hasattr(descriptor, "pipe"):
    raise RuntimeError("descriptor.pipe not found; cannot integrate preprocessing with model.")

full_model = Pipeline([
    ("preprocess", descriptor.pipe),  #scaler + PCA
    ("model", model),
])

print("\nFull integrated pipeline:\n", full_model)

# -----------------------------
# Extract raw bits 
# -----------------------------
X_train_raw = descriptor.training_bits
X_test_raw = descriptor.test_bits

print("\nRAW shapes:",
      "X_train_raw =", X_train_raw.shape,
      "X_test_raw  =", X_test_raw.shape)

# -----------------------------
# Features names as bits for plots
# -----------------------------

n_bits = X_train_raw.shape[1]
bit_names = [f"bit_{i}" for i in range(n_bits)]

X_train_df = pd.DataFrame(X_train_raw, columns=bit_names)
X_test_df  = pd.DataFrame(X_test_raw, columns=bit_names)

# -----------------------------
# SHAP on integrated pipeline 
# -----------------------------

bg = X_train_df.iloc[:N_BG]
X_explain = X_test_df.iloc[:N_EXPLAIN]

explainer = shap.Explainer(
    full_model.predict,     
    bg,                     
    algorithm="permutation", #for MLP/SVM --> non linear models 
    max_evals=2 * X_train_df.shape[1] + 1 #1763
    #for this warning in MLP 
    #ValueError: max_evals=500 is too low for the Permutation explainer, it must be at least 2 * num_features + 1 = 823!
    #Raw X has 881 features. 
    #Permutation SHAP needs to do enough model evaluations per sample to estimate contributions of each feature.
)

sv = explainer(X_explain)

print("\nSHAP done.")
print("sv.values shape:", sv.values.shape)  # (N_EXPLAIN, n_raw_features)


# -----------------------------
# Importance table
# -----------------------------

shap_vals = sv.values  # (n_samples, n_bits)

mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
mean_shap = np.mean(shap_vals, axis=0)  

# Feature labels returned by your fingerprint object
feat_names = descriptor.fp.get_feature_names_out()

df = pd.DataFrame({
    "bit_index": np.arange(shap_vals.shape[1], dtype=int),
    "feature_name": feat_names,         # substructure
    "mean_abs_shap": mean_abs_shap,     # importance
    "shap value": mean_shap              # direction
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

out_csv = os.path.join(out_dir, "shap_bit_importance.csv")
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)
print(df.head(20))

# -----------------------------
# Waterfall for selected molecules (custom titles)
# -----------------------------
selected_ks = [12904,12854, 5795]

titles = {
    12904: "ZINC00398859",
    12854: "ZINC84482347",
    5795:  "ZINC03844427",
}


X_sel = X_test_df.iloc[selected_ks]

sv_sel = explainer(X_sel)

print("\nSHAP done for selected molecules.")
print("sv_sel.values shape:", sv_sel.values.shape)


for i, k in enumerate(selected_ks):
    out_png = os.path.join(out_dir, f"shap_waterfall_{titles[k]}_bits.png")

    plt.figure()
    shap.plots.waterfall(sv_sel[i], show=False, max_display=MAX_DISPLAY)
    plt.title(titles[k], fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print("Saved:", out_png)


# -----------------------------
# Save plots
# -----------------------------

beeswarm_path = os.path.join(out_dir, "shap_beeswarm_pubchem_bits_10.png")
bar_path = os.path.join(out_dir, "shap_bar_pubchem_bits.png")

plt.figure()
shap.plots.beeswarm(sv, show=False, max_display=MAX_DISPLAY)
plt.tight_layout()
plt.savefig(beeswarm_path, dpi=300)
plt.close()
print("Saved:", beeswarm_path)

plt.figure()
shap.plots.bar(sv, show=False, max_display=MAX_DISPLAY)
plt.tight_layout()
plt.savefig(bar_path, dpi=300)
plt.close()
print("Saved:", bar_path)

