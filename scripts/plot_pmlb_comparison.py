import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_radar_performance(csv_files, output_path):
    if not csv_files:
        print("Error: No CSV files provided.")
        return

    dataframes = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)[["dataset", "mean_test_score"]]
            print(len(df), csv_file)
            # Extract classifier name from filename (e.g. pmlb_dt.csv -> dt)
            base_name = os.path.basename(csv_file)
            name_without_ext = os.path.splitext(base_name)[0]
            # Try to remove 'pmlb_' prefix if it exists
            if name_without_ext.startswith("pmlb_"):
                clf_name = name_without_ext[5:]
            else:
                clf_name = name_without_ext

            dataframes[clf_name.upper()] = df
        except FileNotFoundError:
            print(f"Error: {csv_file} not found.")
            return

    # Merge all datasets on 'dataset' column
    clf_names = list(dataframes.keys())
    merged_df = dataframes[clf_names[0]].copy()
    merged_df = merged_df.rename(columns={"mean_test_score": f"score_{clf_names[0]}"})

    for clf_name in clf_names[1:]:
        df = dataframes[clf_name].copy()
        if tuple(sorted(df["dataset"].values.tolist())) != tuple(sorted(merged_df["dataset"].values.tolist())):
            print(set(df["dataset"].values) - set(merged_df["dataset"].values))
            print(set(merged_df["dataset"].values) - set(df["dataset"].values))
        df = df.rename(columns={"mean_test_score": f"score_{clf_name}"})
        merged_df = pd.merge(merged_df, df, on="dataset")

    # Sort by the first classifier's score for a smoother curve
    merged_df = merged_df.sort_values(f"score_{clf_names[0]}").reset_index(drop=True)

    # Calculate and print the number of times each algorithm got the best (or tied for best) score
    score_cols = [f"score_{clf_name}" for clf_name in clf_names]
    max_scores = merged_df[score_cols].max(axis=1)

    print("\n--- Performance Summary ---")
    for clf_name, score_col in zip(clf_names, score_cols):
        # Use np.isclose to handle floating point inaccuracies when checking for equality
        is_best = np.isclose(merged_df[score_col], max_scores, rtol=1e-5, atol=1e-8)
        best_count = is_best.sum()
        print(
            f"{clf_name} achieved best (or tied for best) on {best_count} out of {len(merged_df)} datasets."
        )
    print("---------------------------\n")

    datasets = merged_df["dataset"].values
    N = len(datasets)

    # Compute angles for each dataset
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={"projection": "polar"})
    ax.set_ylim(0, 1)

    colors = [
        "royalblue",
        "crimson",
        "darkorange",
        "forestgreen",
        "purple",
        "teal",
        "gold",
        "deeppink",
        "navy",
        "limegreen",
    ]

    for idx, clf_name in enumerate(clf_names):
        scores = merged_df[f"score_{clf_name}"].values
        scores_closed = np.append(scores, scores[0])

        color = colors[idx]

        # Draw the continuous lines
        ax.plot(
            angles_closed,
            scores_closed,
            color=color,
            linewidth=2,
            label=clf_name,
        )

        # Fill the area under the lines for better visibility
        ax.fill(angles_closed, scores_closed, color=color, alpha=0.1)

    # Emphasize the outer circle of radius 1
    circle_angles = np.linspace(0, 2 * np.pi, 200)
    ax.plot(circle_angles, np.ones_like(circle_angles), color="black", linewidth=1.5)

    # Set custom labels around the perimeter
    ax.set_xticks(angles)
    ax.set_xticklabels(datasets, fontsize=6)

    # Rotate labels to be readable and point outwards
    for label, angle in zip(ax.get_xticklabels(), angles):
        angle_deg = np.degrees(angle)
        if angle_deg < 90 or angle_deg > 270:
            label.set_ha("left")
            label.set_va("center")
            label.set_rotation(angle_deg)
        else:
            label.set_ha("right")
            label.set_va("center")
            label.set_rotation(angle_deg - 180)

    # Add some padding for the labels so they don't overlap with the circle
    ax.tick_params(axis="x", pad=15)

    # Configure radial ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
    ax.set_rlabel_position(0)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize=12)

    # Add a title
    plt.title(
        "Performance Comparison",
        y=1.08,
        fontsize=18,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot radar chart comparing performance from multiple CSV files."
    )
    parser.add_argument(
        "--csv-files",
        nargs="+",
        required=True,
        help="List of CSV files to compare (e.g., pmlb_dt.csv pmlb_gp_tuned.csv)",
    )
    parser.add_argument(
        "--out", type=str, default="pmlb_comparison.png", help="Output image path"
    )
    args = parser.parse_args()

    plot_radar_performance(args.csv_files, args.out)
