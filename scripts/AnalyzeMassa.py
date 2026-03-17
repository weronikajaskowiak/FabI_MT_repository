from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt

# ---- INPUT ----
sdf_file = "/Users/weronikajaskowiak/Desktop/HighlightFingerprints/massa_file_analysis/4FS3_roc_ligprep_massa_output.sdf"
stats_file = "basic_stats.csv"
out_prefix = "ROC_analysis"

selected_ids = {
    "ZINC00398859",
    "ZINC03844427",
    "ZINC84482347"
}

# ---- READ SDF ----
suppl = Chem.SDMolSupplier(sdf_file)

rows = []
selected_rows = []

for mol in suppl:
    if mol is None:
        continue
    if not mol.HasProp("r_i_docking_score"):
        continue
    if not mol.HasProp("ExactMolWt"):
        continue
    if not mol.HasProp("HBA count"):
        continue
    if not mol.HasProp("HBD count"):
        continue

    name = mol.GetProp("_Name") if mol.HasProp("_Name") else None

    rows.append({
        "DockingScore": float(mol.GetProp("r_i_docking_score")),
        "ExactMolWt": float(mol.GetProp("ExactMolWt")),
        "HBA": int(mol.GetProp("HBA count")),  # safe if stored as "1" or "1.0"
        "HBD": int(mol.GetProp("HBD count"))
    })

    if name in selected_ids:
        selected_rows.append({
            "ZINC_ID": name,
            "ExactMolWt": float(mol.GetProp("ExactMolWt")),
            "HBA": int(mol.GetProp("HBA count")),  # safe if stored as "1" or "1.0"
            "HBD": int(mol.GetProp("HBD count"))
        })

df = pd.DataFrame(rows)

print(f"File read successfully. Loaded {len(df)} molecules")

# ---- BASIC STATISTICS ----
stats = df.describe().loc[["count", "mean", "std", "min", "50%", "max"]]
stats.rename(index={"50%": "median"}, inplace=True)

print("\nBasic statistics:")
print(stats)

stats_out = f"{out_prefix}_{stats_file}_2"
stats.to_csv(stats_out)
print(f"\nBasic statistics saved to: {stats_out}")

selected_out = f"{out_prefix}_selected_molecules_2.csv"
pd.DataFrame(selected_rows).to_csv(selected_out, index=False)
print(f"Selected molecules saved to: {selected_out}")


#---- HISTOGRAM: DOCKING SCORE ----
plt.figure()
plt.hist(df["DockingScore"], bins=30, color="skyblue")
plt.xlabel("Docking score")
plt.ylabel("Count")
plt.title("Docking score distribution")
plt.tight_layout()
plt.savefig(f"{out_prefix}_docking_hist_30_bins.png", dpi=200)
plt.close()

#---- HISTOGRAM: MOLECULAR WEIGHT ----
plt.figure()
plt.hist(df["ExactMolWt"], bins=30, color="skyblue")
plt.xlabel("Exact molecular weight")
plt.ylabel("Count")
plt.title("ExactMolWt distribution")
plt.tight_layout()
plt.savefig(f"{out_prefix}_mw_hist_30_bins.png", dpi=200)
plt.close()

# ---- HISTOGRAM: HBA ----
plt.figure()
plt.hist(df["HBA"], bins="fd", color="skyblue")
plt.xlabel("HBA")
plt.ylabel("Count")
plt.title("HBA distribution")
plt.tight_layout()
plt.savefig(f"{out_prefix}_hba_hist_FD_method.png", dpi=200)
plt.close()

# ---- HISTOGRAM: HBD ----
plt.figure()
plt.hist(df["HBD"], bins="fd", color="skyblue")
plt.xlabel("HBD")
plt.ylabel("Count")
plt.title("HBD distribution")
plt.tight_layout()
plt.savefig(f"{out_prefix}_hbd_hist_FD_method.png", dpi=200)
plt.close()

print("\nPlots saved:")
print(f"  {out_prefix}_docking_hist.png")
print(f"  {out_prefix}_mw_hist.png")
print(f"  {out_prefix}_hba_hist.png")
print(f"  {out_prefix}_hbd_hist.png")
