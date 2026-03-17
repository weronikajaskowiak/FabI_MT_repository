from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- INPUT ----
before_docking = "/Users/weronikajaskowiak/Desktop/HighlightFingerprints/massa_file_analysis/zinc12dlc90_ligprep-massa.sdf"
after_docking = "/Users/weronikajaskowiak/Desktop/Master_thesis/MASSA_output_ZINC/7UMW_zinc12_MASSA_output/massa_output_7UMW_zinc12.sdf"
out_prefix = "7UMW_comparison_mw_before_after_docking"

def read_exact_mw(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path)
    mw = []
    for mol in suppl:
        if mol is None:
            continue
        if mol.HasProp("ExactMolWt"):
            mw.append(float(mol.GetProp("ExactMolWt")))
    return mw

mw_before = read_exact_mw(before_docking)
mw_after = read_exact_mw(after_docking)

print(f"Before docking: {len(mw_before)} molecules")
print(f"After docking: {len(mw_after)} molecules")

bins = np.linspace(
    min(mw_before + mw_after),
    max(mw_before + mw_after),
    30
)

plt.figure(figsize=(8, 6))

plt.hist(
    mw_before,
    bins=bins,
    color="blue",
    alpha=0.35,
    label="Before docking"
)

plt.hist(
    mw_after,
    bins=bins,
    color="orange",
    alpha=0.75,
    label="After docking"
)

plt.xlabel("Exact molecular weight")
plt.ylabel("Count")
plt.title("ExactMolWt distribution (before vs after docking)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_prefix}_overlay_hist.png", dpi=200)
plt.close()
