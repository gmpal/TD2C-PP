# %%
# avoids the need for users to install TD2C as a package
import sys

sys.path.append("../..")

# %%
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier

# %%
# Dictionary mapping dataset names to their file paths
all_datasets = {
    "NETSIM_5": "data/descriptors/descriptors_netsim_5.pkl",
    "NETSIM_10": "data/descriptors/descriptors_netsim_10.pkl",
    "DREAM3_10": "data/descriptors/descriptors_dream3_10.pkl",
    "DREAM3_50": "data/descriptors/descriptors_dream3_50.pkl",
    "SYNTHETIC": "data/descriptors/descriptors_df_test.pkl",
}

# Dictionary to store the ranked feature DataFrames for each dataset
ranked_features_per_dataset = {}

# --- Train a separate model for each dataset and rank its features ---
for name, path in all_datasets.items():
    print(f"--- Processing: {name} ---")

    # Load the dataset
    df = pd.read_pickle(path)

    # Fill NaNs with the mean of that specific dataset
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Prepare features and targets
    feature_cols = [
        col
        for col in df.columns
        if col not in ["graph_id", "edge_source", "edge_dest", "is_causal"]
    ]

    X = df[feature_cols]
    y = df["is_causal"]

    # Train a Balanced Random Forest model
    clf = BalancedRandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=50,
        sampling_strategy="auto",  # Automatically balance the classes
        replacement=True,  # Allow replacement to ensure balanced sampling
        bootstrap=True,  # Use bootstrap sampling
    )
    clf.fit(X, y)

    # Create a DataFrame for this dataset's feature importances
    feature_importances = pd.DataFrame(
        {"Feature": feature_cols, "Importance": clf.feature_importances_}
    )

    # Sort the DataFrame by importance in descending order
    feature_importances.sort_values(by="Importance", ascending=False, inplace=True)

    # Reset the index to reflect the rank
    feature_importances.reset_index(drop=True, inplace=True)

    # Store the fully ranked DataFrame
    ranked_features_per_dataset[name] = feature_importances

    # Print the top 15 features for immediate review
    print(f"Top 15 Features for {name}:")
    print(feature_importances.head(15))
    print("-" * 50 + "\n")


# %%
# --- Create a single table showing the ranked features for all datasets side-by-side ---

# Use the list of feature names from each sorted DataFrame
final_table_data = {
    name: df["Feature"].tolist() for name, df in ranked_features_per_dataset.items()
}

# Create the final DataFrame
final_ranked_table_df = pd.DataFrame(final_table_data)

# Set the index to be the rank (starting from 1)
final_ranked_table_df.index = np.arange(1, len(final_ranked_table_df) + 1)
final_ranked_table_df.index.name = "Rank"

print("\n\n--- Consolidated Table of Ranked Features per Dataset ---")
# Display the top 20 ranked features across all datasets
print(final_ranked_table_df.head(20))

# Save the complete ranked table to a CSV file
output_filename = "feature_rankings_per_dataset.csv"
final_ranked_table_df.to_csv(output_filename)

print(f"\nComplete feature rankings saved to '{output_filename}'")

# %%
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.ensemble import BalancedRandomForestClassifier

# %%
# Dictionary mapping test set names to their file paths
all_tests_set = {
    "NETSIM_5": "data/descriptors/descriptors_netsim_5.pkl",
    "NETSIM_10": "data/descriptors/descriptors_netsim_10.pkl",
    "DREAM3_10": "data/descriptors/descriptors_dream3_10.pkl",
    "DREAM3_50": "data/descriptors/descriptors_dream3_50.pkl",
    "SYNTHETIC": "data/descriptors/descriptors_df_test.pkl",
}

# Dictionary to store feature importances for each dataset
all_importances = {}

# --- Train a separate model for each dataset to get specific feature importances ---
for name, path in all_tests_set.items():
    print(f"Analyzing feature importances for: {name}...")

    # Load the dataset
    df = pd.read_pickle(path)

    # Fill NaNs with the mean of that specific dataset
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Prepare features and targets
    feature_cols = [
        col
        for col in df.columns
        if col not in ["graph_id", "edge_source", "edge_dest", "is_causal"]
    ]

    X = df[feature_cols]
    y = df["is_causal"]

    # Train a Balanced Random Forest model
    clf = BalancedRandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=50,
        sampling_strategy="auto",  # Automatically balance the classes
        replacement=True,  # Allow replacement to ensure balanced sampling
        bootstrap=True,  # Use bootstrap sampling
    )
    clf.fit(X, y)

    # Store the feature importances in a pandas Series with feature names as the index
    importances_series = pd.Series(clf.feature_importances_, index=feature_cols)
    all_importances[name] = importances_series

# Combine all importance Series into a single DataFrame
importance_df = pd.DataFrame(all_importances)

# %%
# --- Select the top features for visualization ---

# Find the union of the top 10 features from each dataset
top_features = set()
for dataset_name in importance_df.columns:
    top_10 = importance_df[dataset_name].nlargest(10).index
    top_features.update(top_10)

# Filter the DataFrame to only include the overall top features
top_features_df = importance_df.loc[list(top_features)]

# Sort the features by their mean importance across all datasets for better visualization
top_features_df = top_features_df.loc[
    top_features_df.mean(axis=1).sort_values(ascending=False).index
]


print("\n--- Top Feature Importances Across Datasets ---")
print(top_features_df)

# Save the results to a CSV file
top_features_df.to_csv("top_feature_importances_per_dataset.csv")
print(
    "\nTop feature importance data saved to 'top_feature_importances_per_dataset.csv'"
)

# %%
# --- Visualize the results in a heatmap ---

plt.style.use("default")
plt.figure(figsize=(12, 14))  # Adjust size as needed based on the number of features

sns.heatmap(
    top_features_df,
    annot=True,  # Show the importance scores on the heatmap
    fmt=".3f",  # Format numbers to 3 decimal places
    cmap="viridis",  # Color scheme
    linewidths=0.5,
)

plt.title("Top 10 Feature Importances per Dataset", fontsize=16, pad=20)
plt.xlabel("Dataset", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# %%


# %%


# %%


# %%


# %%
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
import pandas as pd

# %%
all_tests_set = {
    "NETSIM_5": "data/descriptors/descriptors_netsim_5.pkl",
    "NETSIM_10": "data/descriptors/descriptors_netsim_10.pkl",
    "DREAM3_10": "data/descriptors/descriptors_dream3_10.pkl",
    "DREAM3_50": "data/descriptors/descriptors_dream3_50.pkl",
    "SYNTHETIC": "data/descriptors/descriptors_df_test.pkl",
}

multiple_dfs = []
for name, path in all_tests_set.items():
    single_df = pd.read_pickle(path)
    multiple_dfs.append(single_df)

train_descriptors = pd.concat(multiple_dfs, ignore_index=True)
# Fill NaNs
train_descriptors.fillna(0, inplace=True)

# Prepare features and targets
feature_cols = [
    col
    for col in train_descriptors.columns
    if col not in ["graph_id", "edge_source", "edge_dest", "is_causal"]
]

X_train = train_descriptors[feature_cols]
y_train = train_descriptors["is_causal"]

clf = BalancedRandomForestClassifier(n_estimators=500, random_state=42, n_jobs=50)
clf.fit(X_train, y_train)

# %%
import matplotlib.pyplot as plt

# Get feature importances
importances = clf.feature_importances_

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({"Feature": feature_cols, "Importance": importances})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Plotting
plt.figure(figsize=(10, 30))
plt.barh(
    feature_importances["Feature"], feature_importances["Importance"], color="skyblue"
)
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.show()


# %%
feature_importances.to_csv("feature_importances.csv", index=False)
