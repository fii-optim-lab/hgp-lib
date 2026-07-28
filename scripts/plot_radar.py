import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_radar_performance(series, output_path):
    if not series:
        print("Error: No CSV files provided.")
        return

    dataframes = {}
    colors = {}
    legend_names = {}

    for csv_file, color, legend_name in series:
        try:
            df = pd.read_csv(csv_file)[["dataset", "mean_test_score"]]
            print(len(df), csv_file)

            # Extract classifier name from filename
            base_name = os.path.basename(csv_file)
            name_without_ext = os.path.splitext(base_name)[0]

            # Remove 'pmlb_' prefix if it exists
            if name_without_ext.startswith("pmlb_"):
                clf_name = name_without_ext[5:]
            else:
                clf_name = name_without_ext

            clf_key = clf_name.upper()

            dataframes[clf_key] = df
            colors[clf_key] = color
            legend_names[clf_key] = legend_name

        except FileNotFoundError:
            print(f"Error: {csv_file} not found.")
            return

    # Merge all datasets on 'dataset' column
    clf_names = list(dataframes.keys())

    merged_df = dataframes[clf_names[0]].copy()
    merged_df = merged_df.rename(columns={"mean_test_score": f"score_{clf_names[0]}"})

    for clf_name in clf_names[1:]:
        df = dataframes[clf_name].copy()

        if tuple(sorted(df["dataset"].values.tolist())) != tuple(
            sorted(merged_df["dataset"].values.tolist())
        ):
            print(f"Dataset mismatch for {clf_name}")
            print("Only in current CSV:")
            print(set(df["dataset"].values) - set(merged_df["dataset"].values))
            print("Only in previous CSVs:")
            print(set(merged_df["dataset"].values) - set(df["dataset"].values))

        df = df.rename(columns={"mean_test_score": f"score_{clf_name}"})
        merged_df = pd.merge(merged_df, df, on="dataset")

    # Sort by the first classifier's score for a smoother curve
    merged_df = merged_df.sort_values(f"score_{clf_names[0]}").reset_index(drop=True)

    # Calculate and print the number of times each algorithm got the best score
    score_cols = [f"score_{clf_name}" for clf_name in clf_names]
    max_scores = merged_df[score_cols].max(axis=1)

    print("\n--- Performance Summary ---")
    for clf_name, score_col in zip(clf_names, score_cols):
        is_best = np.isclose(
            merged_df[score_col],
            max_scores,
            rtol=1e-5,
            atol=1e-8,
        )
        best_count = is_best.sum()

        print(
            f"{legend_names[clf_name]} achieved best "
            f"(or tied for best) on {best_count} out of {len(merged_df)} datasets."
        )
    print("---------------------------\n")

    datasets = merged_df["dataset"].values
    N = len(datasets)

    # Compute angles for each dataset
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    _fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={"projection": "polar"})
    ax.set_ylim(0, 1)

    for clf_name in clf_names:
        scores = merged_df[f"score_{clf_name}"].values
        scores_closed = np.append(scores, scores[0])

        color = colors[clf_name]
        legend_name = legend_names[clf_name]

        ax.plot(
            angles_closed,
            scores_closed,
            color=color,
            linewidth=2,
            label=legend_name,
        )

        ax.fill(
            angles_closed,
            scores_closed,
            color=color,
            alpha=0.1,
        )

    # Emphasize the outer circle of radius 1
    circle_angles = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        circle_angles,
        np.ones_like(circle_angles),
        color="black",
        linewidth=1.5,
    )

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

    ax.tick_params(axis="x", pad=15)

    # Configure radial ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"],
        color="grey",
        size=9,
    )
    ax.set_rlabel_position(0)

    # Add legend
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.1, 1.1),
        fontsize=12,
    )

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
        "--series",
        nargs=3,
        action="append",
        metavar=("CSV_FILE", "COLOR", "LEGEND"),
        required=True,
        help=(
            "CSV file, color, and legend name. Example: --series pmlb_dt.csv royalblue 'Decision Tree'. "
            "Can be used multiple times."
        ),
    )

    parser.add_argument(
        "--out",
        type=str,
        default="pmlb_comparison.png",
        help="Output image path",
    )

    args = parser.parse_args()

    plot_radar_performance(args.series, args.out)

# Possible colors:
# royalblue, crimson, darkorange, forestgreen, purple,
# teal, gold, deeppink, navy, limegreen, black, gray,
# brown, olive, cyan, magenta, tomato, dodgerblue

# One!
# python scripts/plot_radar.py --series pmlb_gp.csv royalblue "HGP (tuned)" --out images/hgp_tuned.png
# python scripts/plot_radar.py --series pmlb_gp_default.csv dodgerblue "HGP (default)" --out images/hgp_default.png

# tuned vs default
# python scripts/plot_radar.py --series pmlb_gp.csv royalblue "HGP (tuned)" --series pmlb_gp_default.csv dodgerblue "HGP (default)" --out images/hgp_tuned_vs_default.png

# tuned vs DT
# python scripts/plot_radar.py --series pmlb_gp.csv royalblue "HGP (tuned)" --series pmlb_dt.csv crimson "DT" --out images/hgp_tuned_vs_dt.png

# tuned vs boolxai
# python scripts/plot_radar.py --series pmlb_gp.csv royalblue "HGP (tuned)" --series pmlb_boolxai.csv darkorange "boolxai" --out images/hgp_tuned_vs_boolxai.png

# default vs boolxai
# python scripts/plot_radar.py --series pmlb_gp_default.csv dodgerblue "HGP (default)" --series pmlb_boolxai.csv darkorange "boolxai" --out images/hgp_default_vs_boolxai.png

# default vs DT
# python scripts/plot_radar.py --series pmlb_gp_default.csv dodgerblue "HGP (default)" --series pmlb_dt.csv crimson "DT" --out images/hgp_default_vs_dt.png

# tuned vs DT vs boolxai
# python scripts/plot_radar.py --series pmlb_gp.csv royalblue "HGP (tuned)" --series pmlb_dt.csv crimson "DT" --series pmlb_boolxai.csv darkorange "boolxai" --out images/hgp_tuned_vs_dt_vs_boolxai.png

# tuned vs DT limited
# python scripts/plot_radar.py --series pmlb_gp.csv royalblue "HGP (tuned)" --series pmlb_dt_100.csv tomato "DT pruned (100)" --out images/hgp_tuned_vs_dt_100.png
# python scripts/plot_radar.py --series pmlb_gp.csv royalblue "HGP (tuned)" --series pmlb_dt_50.csv magenta "DT pruned (50)" --out images/hgp_tuned_vs_dt_50.png
