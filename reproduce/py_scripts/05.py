# %%
# avoids the need for users to install TD2C as a package
import sys

sys.path.append("../../")

# %%
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)


def analyze_test_dataset(
    test_pickle_path="data/causal_dfs/causal_dfs_TEST.pkl", output_dir="TEST_analysis"
):
    """
    Analyzes the TEST dataset by computing metrics for each error type and process.

    Args:
        test_pickle_path (str): Path to the TEST pickle file
        output_dir (str): Directory to save results

    Returns:
        dict: Dictionary containing all computed metrics and dataframes
    """
    THRESHOLD = 0.309  # Threshold for binary classification
    # Configuration
    TUPLE_INDEX_TO_METHOD = [
        "VAR",
        "VARLiNGAM",
        "PCMCI",
        "MVGC",
        "PCMCI-GPDC",
        "Granger",
        "DYNOTEARS",
        "D2C",
    ]
    # Error type and process mapping (based on your generation code)
    error_process_map = {
        "gaussian": [2, 4, 6, 8, 10, 12, 14, 16, 18],
        "uniform": [2, 4, 6, 8, 10, 12, 14, 16, 18],
        "laplace": [2, 4, 6, 8, 10, 12, 14, 16, 18],
    }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/figures/error_type", exist_ok=True)
    os.makedirs(f"{output_dir}/figures/process", exist_ok=True)

    # Load the TEST data
    print(f"Loading TEST dataset from {test_pickle_path}")
    with open(test_pickle_path, "rb") as f:
        loaded_data = pickle.load(f)

    method_results_tuple = loaded_data[:-1]
    true_causal_dfs_dict = loaded_data[-1]

    # Get all run IDs
    all_dicts = [d for d in method_results_tuple if d is not None] + [
        true_causal_dfs_dict
    ]
    common_keys = set(all_dicts[0].keys())
    for d in all_dicts[1:]:
        common_keys.intersection_update(d.keys())
    run_ids = sorted(list(common_keys))

    print(f"Found {len(run_ids)} runs in the dataset")

    # Map run IDs to error types and processes
    def map_run_to_error_and_process(run_id):
        """Map run ID to error type and process number based on generation order."""
        # Assuming runs are generated in order: gaussian processes, then uniform, then laplace
        # Each error type has 40 time series per process, and 9 processes
        total_per_error = 40 * 9  # 360 runs per error type

        if run_id < total_per_error:
            error_type = "gaussian"
            process_idx = run_id // 40
            process_num = error_process_map["gaussian"][process_idx]
        elif run_id < 2 * total_per_error:
            error_type = "uniform"
            process_idx = (run_id - total_per_error) // 40
            process_num = error_process_map["uniform"][process_idx]
        else:
            error_type = "laplace"
            process_idx = (run_id - 2 * total_per_error) // 40
            process_num = error_process_map["laplace"][process_idx]

        return error_type, process_num

    # Create mapping dictionaries
    run_to_error = {}
    run_to_process = {}
    run_to_error_process = {}

    for run_id in run_ids:
        error_type, process_num = map_run_to_error_and_process(run_id)
        run_to_error[run_id] = error_type
        run_to_process[run_id] = process_num
        run_to_error_process[run_id] = f"{error_type}_process_{process_num}"

    # Compute metrics function
    def compute_metrics(y_true, y_pred):
        """Compute all metrics for given true and predicted values."""
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        }

        return metrics

    # Initialize results storage
    results = {
        "error_type_results": {},
        "process_results": {},
        "error_process_results": {},
        "detailed_results": [],
    }

    # =============================================================================
    # 1. ANALYSIS BY ERROR TYPE
    # =============================================================================
    print("\n=== ANALYZING BY ERROR TYPE ===")

    for error_type in ["gaussian", "uniform", "laplace"]:
        print(f"\nProcessing error type: {error_type}")

        # Get runs for this error type
        error_runs = [
            run_id for run_id in run_ids if run_to_error[run_id] == error_type
        ]

        # Micro-averaging (pooled)
        pooled_method_dfs = {}
        for i, method_dfs_dict in enumerate(method_results_tuple):
            method_name = TUPLE_INDEX_TO_METHOD[i]
            if method_dfs_dict is None:
                continue

            error_dfs = [
                method_dfs_dict[run_id]
                for run_id in error_runs
                if run_id in method_dfs_dict
            ]
            if error_dfs:
                pooled_method_dfs[method_name] = pd.concat(error_dfs).reset_index(
                    drop=True
                )

        # True labels for this error type
        true_dfs = [
            true_causal_dfs_dict[run_id]
            for run_id in error_runs
            if run_id in true_causal_dfs_dict
        ]
        y_true_pooled = (
            pd.concat(true_dfs)["is_causal"].astype(int).reset_index(drop=True)
        )

        # Compute micro-averaged metrics
        micro_scores = []
        for method_name, pred_df in pooled_method_dfs.items():
            if method_name == "D2C":
                y_proba = pred_df["probability"].astype(float)
                y_pred = (y_proba > THRESHOLD).astype(int)
            else:
                y_pred = pred_df["is_causal"].astype(int)

            metrics = compute_metrics(y_true_pooled, y_pred)
            metrics["Method"] = method_name
            micro_scores.append(metrics)

        micro_df = pd.DataFrame(micro_scores).set_index("Method")

        # Macro-averaging (per-run)
        macro_results = []
        for run_id in error_runs:
            y_true_run = true_causal_dfs_dict[run_id]["is_causal"].astype(int)

            for i, method_dfs_dict in enumerate(method_results_tuple):
                method_name = TUPLE_INDEX_TO_METHOD[i]
                if method_dfs_dict is None or run_id not in method_dfs_dict:
                    continue

                if method_name == "D2C":
                    y_proba_run = method_dfs_dict[run_id]["probability"].astype(float)
                    y_pred_run = (y_proba_run > THRESHOLD).astype(int)
                else:
                    y_pred_run = method_dfs_dict[run_id]["is_causal"].astype(int)

                metrics = compute_metrics(y_true_run, y_pred_run)

                for metric_name, score in metrics.items():
                    macro_results.append(
                        {
                            "Method": method_name,
                            "Metric": metric_name,
                            "Score": score,
                            "Run_ID": run_id,
                        }
                    )

        macro_df = pd.DataFrame(macro_results).dropna(subset=["Score"])
        macro_summary = (
            macro_df.groupby(["Method", "Metric"])["Score"]
            .agg(["mean", "std"])
            .fillna(0)
        )
        macro_summary["Mean ± Std"] = macro_summary.apply(
            lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
        )
        macro_pivoted = macro_summary["Mean ± Std"].unstack(level="Metric")

        # Store results
        results["error_type_results"][error_type] = {
            "micro_scores": micro_df,
            "macro_summary": macro_pivoted,
            "macro_detailed": macro_df,
        }

        # Save tables
        micro_df.to_csv(
            f"{output_dir}/tables/micro_{error_type}.csv", float_format="%.4f"
        )
        macro_pivoted.to_csv(f"{output_dir}/tables/macro_{error_type}.csv")

        print(f"Micro-averaged results for {error_type}:")
        print(micro_df.to_string(float_format="%.4f"))
        print(f"\nMacro-averaged results for {error_type}:")
        print(macro_pivoted.to_string())

    # =============================================================================
    # 2. ANALYSIS BY PROCESS
    # =============================================================================
    print("\n\n=== ANALYZING BY PROCESS ===")

    all_processes = sorted(set(run_to_process.values()))

    for process_num in all_processes:
        print(f"\nProcessing process: {process_num}")

        # Get runs for this process (across all error types)
        process_runs = [
            run_id for run_id in run_ids if run_to_process[run_id] == process_num
        ]

        # Similar analysis as for error types
        pooled_method_dfs = {}
        for i, method_dfs_dict in enumerate(method_results_tuple):
            method_name = TUPLE_INDEX_TO_METHOD[i]
            if method_dfs_dict is None:
                continue

            process_dfs = [
                method_dfs_dict[run_id]
                for run_id in process_runs
                if run_id in method_dfs_dict
            ]
            if process_dfs:
                pooled_method_dfs[method_name] = pd.concat(process_dfs).reset_index(
                    drop=True
                )

        # True labels for this process
        true_dfs = [
            true_causal_dfs_dict[run_id]
            for run_id in process_runs
            if run_id in true_causal_dfs_dict
        ]
        y_true_pooled = (
            pd.concat(true_dfs)["is_causal"].astype(int).reset_index(drop=True)
        )
        # Compute micro-averaged metrics
        micro_scores = []
        for method_name, pred_df in pooled_method_dfs.items():
            if method_name == "D2C":
                y_proba = pred_df["probability"].astype(float)
                y_pred = (y_proba > THRESHOLD).astype(int)
            else:
                y_pred = pred_df["is_causal"].astype(int)

            metrics = compute_metrics(y_true_pooled, y_pred)
            metrics["Method"] = method_name
            micro_scores.append(metrics)

        micro_df = pd.DataFrame(micro_scores).set_index("Method")

        # Macro-averaging
        macro_results = []
        for run_id in process_runs:
            y_true_run = true_causal_dfs_dict[run_id]["is_causal"].astype(int)

            for i, method_dfs_dict in enumerate(method_results_tuple):
                method_name = TUPLE_INDEX_TO_METHOD[i]
                if method_dfs_dict is None or run_id not in method_dfs_dict:
                    continue

                if method_name == "D2C":
                    y_proba_run = method_dfs_dict[run_id]["probability"].astype(float)
                    y_pred_run = (y_proba_run > THRESHOLD).astype(int)
                else:
                    y_pred_run = method_dfs_dict[run_id]["is_causal"].astype(int)

                metrics = compute_metrics(y_true_run, y_pred_run)

                for metric_name, score in metrics.items():
                    macro_results.append(
                        {
                            "Method": method_name,
                            "Metric": metric_name,
                            "Score": score,
                            "Run_ID": run_id,
                            "Error_Type": run_to_error[run_id],
                        }
                    )

        macro_df = pd.DataFrame(macro_results).dropna(subset=["Score"])
        macro_summary = (
            macro_df.groupby(["Method", "Metric"])["Score"]
            .agg(["mean", "std"])
            .fillna(0)
        )
        macro_summary["Mean ± Std"] = macro_summary.apply(
            lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
        )
        macro_pivoted = macro_summary["Mean ± Std"].unstack(level="Metric")

        # Store results
        results["process_results"][process_num] = {
            "micro_scores": micro_df,
            "macro_summary": macro_pivoted,
            "macro_detailed": macro_df,
        }

        # Save tables
        micro_df.to_csv(
            f"{output_dir}/tables/micro_process_{process_num}.csv", float_format="%.4f"
        )
        macro_pivoted.to_csv(f"{output_dir}/tables/macro_process_{process_num}.csv")

    # =============================================================================
    # 3. CREATE SUMMARY VISUALIZATIONS
    # =============================================================================
    print("\n\n=== CREATING VISUALIZATIONS ===")

    # Set plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # =============================================================================
    # 3.1 ERROR TYPE VISUALIZATIONS
    # =============================================================================
    print("Creating error type visualizations...")

    # Collect all macro data for error type comparison
    all_error_data = []
    for error_type, data in results["error_type_results"].items():
        df = data["macro_detailed"].copy()
        df["Error_Type"] = error_type
        all_error_data.append(df)

    if all_error_data:
        combined_error_df = pd.concat(all_error_data)

        # Create boxplot for each metric
        metrics = combined_error_df["Metric"].unique()
        for metric in metrics:
            metric_data = combined_error_df[combined_error_df["Metric"] == metric]

            fig, ax = plt.subplots(figsize=(14, 8))
            sns.boxplot(
                data=metric_data,
                x="Error_Type",
                y="Score",
                hue="Method",
                ax=ax,
                palette="Set3",
            )
            ax.set_title(f"{metric} by Error Type", fontsize=16, pad=20)
            ax.set_xlabel("Error Type", fontsize=14)
            ax.set_ylabel(f"{metric} Score", fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            fig.tight_layout()
            plt.savefig(
                f"{output_dir}/figures/error_type/boxplot_{metric}_by_error_type.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # =============================================================================
    # 3.2 PROCESS VISUALIZATIONS
    # =============================================================================
    print("Creating process visualizations...")

    # Collect all macro data for process comparison
    all_process_data = []
    for process_num, data in results["process_results"].items():
        df = data["macro_detailed"].copy()
        df["Process"] = process_num
        all_process_data.append(df)

    if all_process_data:
        combined_process_df = pd.concat(all_process_data)

        # --- Create Overall Summary Table (Macro-Averaged) ---
        print("\n--- Creating Overall Summary Table (Macro-Averaged) ---")

        # Use the combined_process_df which contains all per-run scores
        overall_summary = (
            combined_process_df.groupby(["Method", "Metric"])["Score"]
            .agg(["mean", "std"])
            .fillna(0)
        )
        overall_summary["Mean ± Std"] = overall_summary.apply(
            lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
        )
        overall_pivoted_summary = overall_summary["Mean ± Std"].unstack(level="Metric")

        # Store results for later access
        results["overall_summary"] = overall_pivoted_summary

        # Save the summary table to a CSV file
        overall_pivoted_summary.to_csv(f"{output_dir}/tables/overall_macro_summary.csv")

        print("Overall Macro-Averaged Results (Mean ± Std across all runs):")
        print(overall_pivoted_summary.to_string())
        print("----------------------------------------------------------\n")

        # Create boxplots for each metric by process
        metrics = combined_process_df["Metric"].unique()
        for metric in metrics:
            metric_data = combined_process_df[combined_process_df["Metric"] == metric]

            # Boxplot by process
            fig, ax = plt.subplots(figsize=(16, 8))
            sns.boxplot(
                data=metric_data,
                x="Process",
                y="Score",
                hue="Method",
                ax=ax,
                palette="Set3",
            )
            ax.set_title(f"{metric} by Process Number", fontsize=16, pad=20)
            ax.set_xlabel("Process Number", fontsize=14)
            ax.set_ylabel(f"{metric} Score", fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            fig.tight_layout()
            plt.savefig(
                f"{output_dir}/figures/process/boxplot_{metric}_by_process.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Heatmap showing method performance across processes
            pivot_data = (
                metric_data.groupby(["Process", "Method"])["Score"].mean().unstack()
            )
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                pivot_data.T,
                annot=True,
                fmt=".3f",
                cmap="RdYlBu_r",
                ax=ax,
                cbar_kws={"label": f"{metric} Score"},
            )
            ax.set_title(f"{metric} Heatmap: Methods vs Processes", fontsize=16, pad=20)
            ax.set_xlabel("Process Number", fontsize=14)
            ax.set_ylabel("Method", fontsize=14)
            fig.tight_layout()
            plt.savefig(
                f"{output_dir}/figures/process/heatmap_{metric}_by_process.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # Create violin plots showing distribution differences across processes
        for metric in metrics:
            metric_data = combined_process_df[combined_process_df["Metric"] == metric]

            fig, ax = plt.subplots(figsize=(16, 8))
            sns.violinplot(
                data=metric_data,
                x="Process",
                y="Score",
                hue="Method",
                ax=ax,
                palette="Set3",
            )
            ax.set_title(
                f"{metric} Distribution by Process Number", fontsize=16, pad=20
            )
            ax.set_xlabel("Process Number", fontsize=14)
            ax.set_ylabel(f"{metric} Score", fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.xticks(rotation=45)
            fig.tight_layout()
            plt.savefig(
                f"{output_dir}/figures/process/violin_{metric}_by_process.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # Process-wise method ranking visualization
        print("Creating process-wise method ranking visualization...")

        # Calculate average F1-score for each method across all processes
        f1_data = combined_process_df[combined_process_df["Metric"] == "F1-Score"]
        process_method_f1 = (
            f1_data.groupby(["Process", "Method"])["Score"].mean().unstack()
        )

        # Rank methods for each process (1 = best)
        process_rankings = process_method_f1.rank(axis=1, ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            process_rankings.T,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn_r",
            ax=ax,
            cbar_kws={"label": "Rank (1=Best)"},
        )
        ax.set_title(
            "Method Rankings by Process (Based on F1-Score)", fontsize=16, pad=20
        )
        ax.set_xlabel("Process Number", fontsize=14)
        ax.set_ylabel("Method", fontsize=14)
        fig.tight_layout()
        plt.savefig(
            f"{output_dir}/figures/process/method_rankings_by_process.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Process difficulty analysis (based on overall performance)
        print("Creating process difficulty analysis...")

        process_difficulty = f1_data.groupby("Process")["Score"].agg(["mean", "std"])
        process_difficulty["difficulty"] = (
            1 - process_difficulty["mean"]
        )  # Higher difficulty = lower average F1
        process_difficulty = process_difficulty.sort_values(
            "difficulty", ascending=False
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Bar plot of average F1-score by process
        ax1.bar(
            range(len(process_difficulty)),
            process_difficulty["mean"],
            yerr=process_difficulty["std"],
            capsize=5,
            alpha=0.7,
            color="skyblue",
        )
        ax1.set_xlabel("Process Number", fontsize=12)
        ax1.set_ylabel("Average F1-Score", fontsize=12)
        ax1.set_title("Average Method Performance by Process", fontsize=14)
        ax1.set_xticks(range(len(process_difficulty)))
        ax1.set_xticklabels(process_difficulty.index, rotation=45)

        # Process difficulty ranking
        ax2.barh(
            range(len(process_difficulty)),
            process_difficulty["difficulty"],
            alpha=0.7,
            color="coral",
        )
        ax2.set_ylabel("Process Number", fontsize=12)
        ax2.set_xlabel("Difficulty Score (1 - Avg F1)", fontsize=12)
        ax2.set_title("Process Difficulty Ranking", fontsize=14)
        ax2.set_yticks(range(len(process_difficulty)))
        ax2.set_yticklabels(process_difficulty.index)
        ax2.invert_yaxis()

        fig.tight_layout()
        plt.savefig(
            f"{output_dir}/figures/process/process_difficulty_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Error type performance within each process
        print("Creating error type performance within processes...")

        for metric in ["F1-Score", "Accuracy"]:
            metric_data = combined_process_df[combined_process_df["Metric"] == metric]

            # Create a summary plot showing error type performance within each process
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            axes = axes.flatten()

            for i, process_num in enumerate(sorted(all_processes)):
                if i < len(axes):
                    process_data = metric_data[metric_data["Process"] == process_num]

                    if not process_data.empty:
                        sns.boxplot(
                            data=process_data,
                            x="Error_Type",
                            y="Score",
                            hue="Method",
                            ax=axes[i],
                            palette="Set3",
                        )
                        axes[i].set_title(f"Process {process_num}", fontsize=12)
                        axes[i].set_xlabel("Error Type", fontsize=10)
                        axes[i].set_ylabel(f"{metric}", fontsize=10)
                        axes[i].legend().set_visible(False)  # Hide individual legends

            # Remove empty subplots
            for j in range(len(all_processes), len(axes)):
                fig.delaxes(axes[j])

            # Add a single legend for all subplots
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.98, 0.5))

            fig.suptitle(
                f"{metric} by Error Type within Each Process", fontsize=16, y=0.98
            )
            fig.tight_layout()
            plt.savefig(
                f"{output_dir}/figures/process/error_type_within_process_{metric.lower().replace('-', '_')}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # =============================================================================
    # 3.3 SUMMARY STATISTICS AND COMBINED VISUALIZATIONS
    # =============================================================================
    print("Creating summary statistics...")

    # Summary statistics across error types and processes
    summary_stats = {"error_type_summary": {}, "process_summary": {}}

    # Error type summary
    for error_type, data in results["error_type_results"].items():
        summary_stats["error_type_summary"][error_type] = {
            "n_runs": len(data["macro_detailed"]["Run_ID"].unique()),
            "best_method_f1": data["micro_scores"]["F1-Score"].idxmax(),
            "best_f1_score": data["micro_scores"]["F1-Score"].max(),
        }

    # Process summary
    for process_num, data in results["process_results"].items():
        summary_stats["process_summary"][process_num] = {
            "n_runs": len(data["macro_detailed"]["Run_ID"].unique()),
            "best_method_f1": data["micro_scores"]["F1-Score"].idxmax(),
            "best_f1_score": data["micro_scores"]["F1-Score"].max(),
        }

    results["summary_stats"] = summary_stats

    print("Creating F1-score summary table...")

    if all_process_data:
        f1_process_data = combined_process_df[
            combined_process_df["Metric"] == "F1-Score"
        ]
        f1_summary_table = (
            f1_process_data.groupby(["Method", "Process"])["Score"].mean().unstack()
        )

        # Save the table
        f1_summary_table.to_csv(
            f"{output_dir}/tables/f1_summary_methods_vs_processes.csv",
            float_format="%.4f",
        )

        print("\nF1-Score Summary Table (Methods vs Processes):")
        print(f1_summary_table.to_string(float_format="%.4f"))

        # Add to results for later access
        results["f1_summary_table"] = f1_summary_table

    # Create a comprehensive summary visualization
    print("Creating comprehensive summary visualization...")

    if all_error_data and all_process_data:
        # Combined overview plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 1. Error type performance overview
        f1_error_data = combined_error_df[combined_error_df["Metric"] == "F1-Score"]
        error_summary = (
            f1_error_data.groupby(["Error_Type", "Method"])["Score"].mean().unstack()
        )
        sns.heatmap(error_summary.T, annot=True, fmt=".3f", cmap="RdYlBu_r", ax=ax1)
        ax1.set_title("F1-Score by Error Type", fontsize=14)
        ax1.set_xlabel("Error Type", fontsize=12)
        ax1.set_ylabel("Method", fontsize=12)

        # 2. Process performance overview (top methods only for readability)
        f1_process_data = combined_process_df[
            combined_process_df["Metric"] == "F1-Score"
        ]
        process_summary = (
            f1_process_data.groupby(["Process", "Method"])["Score"].mean().unstack()
        )
        # Select top 4 methods based on overall performance
        method_means = process_summary.mean(axis=0).sort_values(ascending=False)
        top_methods = method_means.head(4).index
        sns.heatmap(
            process_summary[top_methods].T,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            ax=ax2,
        )
        ax2.set_title("F1-Score by Process (Top 4 Methods)", fontsize=14)
        ax2.set_xlabel("Process Number", fontsize=12)
        ax2.set_ylabel("Method", fontsize=12)

        # 3. Method consistency across error types
        error_std = (
            f1_error_data.groupby(["Method", "Error_Type"])["Score"].std().unstack()
        )
        sns.heatmap(error_std, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax3)
        ax3.set_title("F1-Score Standard Deviation by Error Type", fontsize=14)
        ax3.set_xlabel("Error Type", fontsize=12)
        ax3.set_ylabel("Method", fontsize=12)

        # 4. Method consistency across processes
        process_std = f1_process_data.groupby("Method")["Score"].std().sort_values()
        ax4.barh(range(len(process_std)), process_std.values, alpha=0.7)
        ax4.set_yticks(range(len(process_std)))
        ax4.set_yticklabels(process_std.index)
        ax4.set_xlabel("F1-Score Standard Deviation", fontsize=12)
        ax4.set_title("Method Consistency Across All Processes", fontsize=14)

        fig.suptitle("Comprehensive Performance Analysis Summary", fontsize=18, y=0.98)
        fig.tight_layout()
        plt.savefig(
            f"{output_dir}/figures/comprehensive_summary.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    print(f"- Tables saved to {output_dir}/tables/")
    print(f"- Error type figures saved to {output_dir}/figures/error_type/")
    print(f"- Process figures saved to {output_dir}/figures/process/")
    print(f"- Summary figure saved to {output_dir}/figures/")

    return results


# Example usage:
if __name__ == "__main__":
    results = analyze_test_dataset()

    # Print some summary information
    print("\n=== SUMMARY ===")
    print("\nError Type Summary:")
    for error_type, stats in results["summary_stats"]["error_type_summary"].items():
        print(
            f"{error_type}: {stats['n_runs']} runs, best method: {stats['best_method_f1']} (F1: {stats['best_f1_score']:.4f})"
        )

    print("\nProcess Summary:")
    for process_num, stats in results["summary_stats"]["process_summary"].items():
        print(
            f"Process {process_num}: {stats['n_runs']} runs, best method: {stats['best_method_f1']} (F1: {stats['best_f1_score']:.4f})"
        )

    # Create F1-score summary table (Methods vs Processes)


# %%
