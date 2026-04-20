# %%
# avoids the need for users to install TD2C as a package
import sys

sys.path.append("../../")

# %%
from src.td2c.dags.utils import run_long_range_analysis

# %%
# Scenario 1: Simple Case (Z_i -> Z_j)
rules_simple = [
    ("Z_i", "Z_i", 1),
    ("Z_j", "Z_j", 1),
    ("Z_i", "Z_j", 1),
]

# Scenario 2: Complex Case (Z_k -> Z_i -> Z_j, plus longer lags)
rules_complex = rules_simple + [
    ("Z_i", "Z_i", 2),
    ("Z_j", "Z_j", 2),
]

# Scenario 3: Latent Confounder Case (L -> Z_i, L -> Z_j)
rules_latent = rules_complex + [
    ("L", "L", 1),
    ("L", "Z_i", 1),
    ("L", "Z_j", 1),
]

# --- Generate and Print Results ---
length = 10

def print_scenario(label, df):
    """Print scenario table with Forward, Backward, and Difference (Delta) rows."""
    fw_col = [c for c in df.columns if c.startswith("FW")][0]
    bw_col = [c for c in df.columns if c.startswith("BW")][0]
    print(f"{'Lag (k)':<10}", end="")
    for lag in df["Lag (k)"]:
        print(f"{lag:>8}", end="")
    print()
    print(f"{'Forward':.<10}", end="")
    for val in df[fw_col]:
        print(f"{val:>8}", end="")
    print()
    print(f"{'Backward':.<10}", end="")
    for val in df[bw_col]:
        print(f"{val:>8}", end="")
    print()
    diff = df[fw_col].values - df[bw_col].values
    print(f"{'Delta':.<10}", end="")
    for val in diff:
        print(f"{val:>8}", end="")
    print("\n")

print("=" * 35 + " SCENARIO 1: Simple Case " + "=" * 35)
df_simple = run_long_range_analysis(rules_simple, length)
print_scenario("Simple", df_simple)

print("=" * 35 + " SCENARIO 2: Complex Case " + "=" * 35)
df_complex = run_long_range_analysis(rules_complex, length)
print_scenario("Complex", df_complex)

print("=" * 35 + " SCENARIO 3: Latent Case " + "=" * 35)
df_latent = run_long_range_analysis(rules_latent, length)
print_scenario("Latent", df_latent)
