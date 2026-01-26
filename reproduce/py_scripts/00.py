# %%
# avoids the need for users to install TD2C as a package
import sys

sys.path.append("../../")

# %%
from src.dags.utils import run_long_range_analysis

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

print("=" * 35 + " SCENARIO 1: Simple Case " + "=" * 35)
df_simple = run_long_range_analysis(rules_simple, length)
print(df_simple.to_string(index=False))
print("\n")

print("=" * 35 + " SCENARIO 2: Complex Case " + "=" * 35)
df_complex = run_long_range_analysis(rules_complex, length)
print(df_complex.to_string(index=False))
print("\n")


print("=" * 35 + " SCENARIO 2: Latent Case " + "=" * 35)
df_latent = run_long_range_analysis(rules_latent, length)
print(df_latent.to_string(index=False))
print("\n")
