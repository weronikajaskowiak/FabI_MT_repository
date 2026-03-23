import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

TARGET_COL = "r_i_docking_score"
BIN_WIDTH = 0.5
TARGET_TOTAL = 150000
RANDOM_STATE = 2025

input_file = "/home/agkronenberger/Documents/Gabriel/lumi/new_2026/0_datasets/1_Glide/1_EnamineROC/7UMW_EnamineROC.csv"
output_file = input_file.replace(".csv", "_balanced.csv")
info_file = output_file.replace(".csv", "_info.csv")
bin_info_file = output_file.replace(".csv", "_bin_stats.csv")

df = pd.read_csv(input_file)

min_val, max_val = df[TARGET_COL].min(), df[TARGET_COL].max()
bins = np.arange(min_val, max_val + BIN_WIDTH, BIN_WIDTH)
df["_bin"] = pd.cut(df[TARGET_COL], bins=bins, include_lowest=True)


bin_counts = df["_bin"].value_counts(sort=False)
n_bins = len(bin_counts)
target_per_bin = TARGET_TOTAL // n_bins  

balanced_parts = []
for i, (bin_interval, count) in enumerate(bin_counts.items()):
    group = df[df["_bin"] == bin_interval]
    
    if len(group) < target_per_bin:
        balanced_parts.append(group)
    else:
        balanced_parts.append(group.sample(n=target_per_bin, random_state=RANDOM_STATE))

df_balanced = pd.concat(balanced_parts, ignore_index=True)

df_balanced["_bin"] = pd.cut(df_balanced[TARGET_COL], bins=bins, include_lowest=True)

df_balanced.drop(columns="_bin").to_csv(output_file, index=False)

ks_stat, p_value = ks_2samp(df[TARGET_COL].to_numpy(), df_balanced[TARGET_COL].to_numpy())

summary = pd.DataFrame({
    "full": df[TARGET_COL].describe(),
    "balanced": df_balanced[TARGET_COL].describe(),
    "KS statistic": {"count": ks_stat},
    "KS p-value": {"count": p_value}
})

summary.to_csv(info_file)

bin_stats = (
    df.groupby("_bin", observed=True)
      .agg(
          bin_min=(TARGET_COL, "min"),
          bin_max=(TARGET_COL, "max"),
          n_full=(TARGET_COL, "size"),
      )
      .join(
          df_balanced.groupby("_bin", observed=True)
                     .agg(n_balanced=(TARGET_COL, "size")),
          how="left"
      )
      .fillna(0)
      .reset_index()
)

bin_stats["bin_id"] = np.arange(len(bin_stats))
bin_stats = bin_stats[["bin_id", "_bin", "bin_min", "bin_max", "n_full", "n_balanced"]]
bin_stats.to_csv(bin_info_file, index=False)

print(f"Original size: {len(df)}")
print(f"Final balanced size: {len(df_balanced)}")
print(f"KS statistic: {ks_stat:.6f}")
print(f"KS p-value:   {p_value:.3e}")
print("\nGlobal summary:")
print(summary)
print("\nBin audit (first 5 bins):")
print(bin_stats.head())
