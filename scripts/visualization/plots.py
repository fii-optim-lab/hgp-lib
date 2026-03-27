import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from hgp_lib.metrics.results import ExperimentResult

matplotlib.use("Agg")



# Color palette
COLORS = [
    "#2196F3",
    "#FF9800",
    "#4CAF50",
    "#E91E63",
    "#9C27B0",
    "#00BCD4",
    "#FF5722",
    "#795548",
    "#607D8B",
    "#CDDC39",
]


def plot_experiment_boxplots(
    experiment: ExperimentResult,
    title: str = "Experiment Score Distribution",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot enhanced boxplots with violin distribution, jittered points,
    annotated Q1/Q3/median/mean, for train, validation, and test scores.

    For each run, train = mean of best train score per fold,
    val = mean of best val score per fold, test = test score.

    Args:
        experiment: ExperimentResult containing multiple runs.
        title: Plot title.
        save_path: Optional path to save the plot.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    train_scores: list[float] = []
    val_scores: list[float] = []
    test_scores = experiment.test_scores

    for run in experiment.runs:
        fold_trains = []
        for fold in run.folds:
            if fold.generations:
                best_train = max(gen.best_train_score for gen in fold.generations)
                fold_trains.append(best_train)
        if fold_trains:
            train_scores.append(float(np.mean(fold_trains)))

        val_scores.append(run.mean_val_score)

    data = [train_scores, val_scores, test_scores]
    labels = ["Train", "Validation", "Test"]
    box_colors = [COLORS[0], COLORS[1], COLORS[2]]
    positions = [1, 2, 3]

    # --- Violin plot for distribution shape (transparent) ---
    for pos, d, color in zip(positions, data, box_colors):
        if len(d) >= 2:
            parts = ax.violinplot(
                d,
                positions=[pos],
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.7,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.18)
                pc.set_edgecolor("none")

    # --- Box plot (narrower, on top of violin) ---
    bp = ax.boxplot(
        data,
        positions=positions,
        tick_labels=labels,
        patch_artist=True,
        widths=0.3,
        showfliers=False,
        zorder=3,
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)

    for element in ["whiskers", "caps"]:
        for line, color in zip(bp[element], [c for c in box_colors for _ in range(2)]):
            line.set_color(color)
            line.set_linewidth(1.2)

    # Hide default median lines (we draw mean lines instead)
    for median_line in bp["medians"]:
        median_line.set_visible(False)

    # --- Jittered individual points ---
    rng = np.random.default_rng(42)
    for pos, d, color in zip(positions, data, box_colors):
        jitter = rng.uniform(-0.08, 0.08, size=len(d))
        ax.scatter(
            np.full(len(d), pos) + jitter,
            d,
            color=color,
            alpha=0.6,
            s=28,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
            marker="o",
        )

    # --- Mean line (solid white, like the old median style) ---
    half_w = 0.15
    for pos, d, _color in zip(positions, data, box_colors):
        mean_val = float(np.mean(d))
        ax.hlines(
            mean_val,
            pos - half_w,
            pos + half_w,
            colors="white",
            linewidths=2,
            linestyles="-",
            zorder=6,
        )

    # --- Annotate all stats on the left side (with anti-overlap) ---
    for pos, d, color in zip(positions, data, box_colors):
        q1 = float(np.percentile(d, 25))
        q3 = float(np.percentile(d, 75))
        median = float(np.median(d))
        mean_val = float(np.mean(d))
        min_val = float(np.min(d))
        max_val = float(np.max(d))
        std_val = float(np.std(d))

        offset = 0.22
        fontsize = 7.5

        # Compute y-range for minimum spacing
        y_range = max_val - min_val if max_val != min_val else 1.0
        min_gap = y_range * 0.06  # minimum vertical gap between labels

        # Left-side labels: spread them so they don't overlap
        raw_left = [
            ("Min", min_val),
            ("Q1", q1),
            ("Md", median),
            ("Q3", q3),
            ("Max", max_val),
        ]
        # Sort by actual value
        raw_left.sort(key=lambda x: x[1])
        # Push labels apart if too close
        placed_y = []
        for i, (_lbl, val) in enumerate(raw_left):
            y = val
            if placed_y and y - placed_y[-1] < min_gap:
                y = placed_y[-1] + min_gap
            placed_y.append(y)

        for (label, _val), y in zip(raw_left, placed_y):
            ax.annotate(
                f"{label}={_val:.3f}",
                xy=(pos - offset, y),
                fontsize=fontsize,
                color=color,
                ha="right",
                va="center",
                fontweight="bold",
            )

        # Right-side: μ and σ stacked using text coordinates
        ax.annotate(
            f"μ={mean_val:.3f}",
            xy=(pos + offset, mean_val),
            fontsize=fontsize,
            color=color,
            ha="left",
            va="center",
            fontweight="bold",
            fontstyle="italic",
        )
        ax.annotate(
            f"σ={std_val:.3f}",
            xy=(pos + offset, mean_val),
            fontsize=fontsize,
            color=color,
            ha="left",
            va="top",
            fontstyle="italic",
            xytext=(0, -10),
            textcoords="offset points",
        )

    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")

    # Legend: category patches only (no mean/std in legend)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=c, alpha=0.45, edgecolor=c, label=lbl)
        for c, lbl in zip(box_colors, labels)
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_best_fold_generations(
    experiment: ExperimentResult,
    title: str = "Best Fold: Score & Complexity over Generations",
    save_path: str | None = None,
    regeneration_patience: int | None = None,
) -> plt.Figure:
    """
    For the best fold in the best run, plot train score, val score,
    and complexity over generations on dual y-axes.

    Optionally marks regeneration points based on regeneration_patience:
    a vertical line is drawn wherever the best train score has not improved
    for ``regeneration_patience`` consecutive generations.

    Args:
        experiment: ExperimentResult containing multiple runs.
        title: Plot title.
        save_path: Optional path to save the plot.
        regeneration_patience: If set, marks generations where regeneration
            would have triggered (no improvement for this many epochs).

    Returns:
        matplotlib Figure object.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    best_run = experiment.best_run
    best_fold = best_run.folds[best_run.best_fold_idx]

    generations = tuple(range(1, len(best_fold.generations) + 1))
    train_scores = [g.best_train_score for g in best_fold.generations]
    complexities = [g.best_rule_complexity for g in best_fold.generations]

    val_gens = []
    val_scores = []
    for i, g in enumerate(best_fold.generations):
        if g.val_score is not None:
            val_gens.append(i + 1)
            val_scores.append(g.val_score)

    # Left axis: scores
    ax1.plot(
        generations, train_scores, color=COLORS[0], linewidth=2, label="Train Score"
    )
    if val_scores:
        ax1.plot(
            val_gens,
            val_scores,
            color=COLORS[1],
            linewidth=2,
            marker="o",
            markersize=3,
            label="Valid Score",
        )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Right axis: complexity
    ax2 = ax1.twinx()
    ax2.plot(
        generations,
        complexities,
        color=COLORS[2],
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label="Complexity",
    )
    ax2.set_ylabel("Complexity (nodes)", color=COLORS[2])
    ax2.tick_params(axis="y", labelcolor=COLORS[2])

    # Mark regeneration points
    if regeneration_patience is not None and regeneration_patience > 0:
        best_so_far = -float("inf")
        epochs_without_improvement = 0
        regen_gens = []
        for gen, score in zip(generations, train_scores):
            if score > best_so_far:
                best_so_far = score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement == regeneration_patience:
                regen_gens.append(gen)
                epochs_without_improvement = 0
                best_so_far = -float("inf")

        for i, rg in enumerate(regen_gens):
            ax1.axvline(
                x=rg,
                color="#9E9E9E",
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
                label="Regeneration" if i == 0 else None,
            )

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_all_folds_val_scores(
    experiment: ExperimentResult,
    title: str = "Validation Scores: Best Folds across Runs",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot the validation score over generations for the best fold of each run.

    Five key runs are highlighted (best, worst, median, Q1, Q3 by mean
    validation score) with full styling, complexity on the right axis,
    peak markers, and test score annotations. All other runs are drawn
    as transparent background lines without complexity.

    Args:
        experiment: ExperimentResult containing multiple runs.
        title: Plot title.
        save_path: Optional path to save the plot.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    # --- Determine the 5 highlighted runs by mean_val_score ---
    sorted_runs = sorted(experiment.runs, key=lambda r: r.mean_val_score)
    n = len(sorted_runs)

    def _pick(idx: int):
        return sorted_runs[min(idx, n - 1)]

    highlight_map: dict[int, str] = {}
    if n >= 1:
        highlight_map[_pick(n - 1).run_id] = "Best"  # max
        highlight_map[_pick(0).run_id] = "Worst"  # min
    if n >= 3:
        highlight_map[_pick(n // 2).run_id] = "Median"  # median
    if n >= 5:
        highlight_map[_pick(n // 4).run_id] = "Q1"  # Q1
        highlight_map[_pick(3 * n // 4).run_id] = "Q3"  # Q3

    # Draw order: Best first (on top in legend)
    highlight_order = ["Best", "Q3", "Median", "Q1", "Worst"]
    highlight_colors = {
        "Best": COLORS[0],
        "Worst": COLORS[3],
        "Median": COLORS[1],
        "Q1": COLORS[4],
        "Q3": COLORS[2],
    }

    # --- First pass: draw background (non-highlighted) runs ---
    for run in experiment.runs:
        if run.run_id in highlight_map:
            continue
        fold = run.folds[run.best_fold_idx]
        val_gens = []
        val_scores = []
        for i, g in enumerate(fold.generations):
            if g.val_score is not None:
                val_gens.append(i + 1)
                val_scores.append(g.val_score)
        if len(val_scores) != 0:
            ax.plot(
                val_gens,
                val_scores,
                color="#BDBDBD",
                linewidth=0.8,
                alpha=0.35,
                linestyle="-",
                zorder=1,
            )

    # --- Second pass: draw highlighted runs in order ---
    tag_to_run = {
        tag: run for run in experiment.runs if (tag := highlight_map.get(run.run_id))
    }
    for tag in highlight_order:
        if tag not in tag_to_run:
            continue
        run = tag_to_run[tag]
        color = highlight_colors[tag]
        fold = run.folds[run.best_fold_idx]

        val_gens = []
        val_scores = []
        val_complexities = []
        for i, g in enumerate(fold.generations):
            if g.val_score is not None:
                val_gens.append(i + 1)
                val_scores.append(g.val_score)
                val_complexities.append(g.best_rule_complexity)

        if not val_scores:
            continue

        # Complexity on right axis
        ax2.plot(
            val_gens,
            val_complexities,
            color=color,
            linewidth=1,
            alpha=0.3,
            linestyle="--",
            zorder=2,
        )

        # Validation score
        ax.plot(
            val_gens,
            val_scores,
            color=color,
            linewidth=1.5,
            marker=".",
            markersize=4,
            alpha=0.9,
            zorder=3,
            label=f"{tag} (Run {run.run_id}, μ_val={run.mean_val_score:.3f})",
        )

        # Peak marker + test annotation
        peak_idx = int(np.argmax(val_scores))
        peak_gen = val_gens[peak_idx]
        peak_val = val_scores[peak_idx]

        marker = "*" if tag == "Best" else "o"
        size = 120 if tag == "Best" else 50

        ax.scatter(
            peak_gen,
            peak_val,
            color=color,
            marker=marker,
            s=size,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        ax.annotate(
            f"test={run.test_score:.3f}",
            xy=(peak_gen, peak_val),
            fontsize=7,
            color=color,
            fontweight="bold",
            xytext=(5, 8),
            textcoords="offset points",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation Score")
    ax2.set_ylabel("Complexity (nodes)", color="#999999")
    ax2.tick_params(axis="y", labelcolor="#999999")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_population_bands(
    experiment: ExperimentResult,
    top_k: int,
    title: str = "Population Bands",
    save_path: str | None = None,
    top_n_children: int = 3,
) -> plt.Figure:
    """
    Plot mean ± 1 std bands for train scores (top subplot) and complexities
    (bottom subplot) of the root population and its top children.

    Root population is drawn prominently. For each depth level, only the
    top ``top_n_children`` child populations are shown (ranked by their
    peak best_train_score). Deeper children are more transparent.
    Both subplots share the same color per population.

    Args:
        experiment: ExperimentResult containing multiple runs.
        top_k: Number of top chromosomes per child (use config.gp_config.top_k_transfer).
        title: Plot suptitle.
        save_path: Optional path to save the plot.
        top_n_children: Number of best children to show per depth level.

    Returns:
        matplotlib Figure object.
    """
    import seaborn as sns

    sns.set_theme(style="whitegrid", rc={"grid.alpha": 0.3})
    fig, (ax_fit, ax_comp) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    best_run = experiment.best_run
    best_fold = best_run.folds[best_run.best_fold_idx]

    root_color = COLORS[0]

    def _collect_band_data(gen_metrics_list, is_root):
        """Collect per-generation means/stds for a population."""
        gens, all_scores, all_comps = [], [], []

        for i, gm in enumerate(gen_metrics_list):
            scores = sorted(gm.train_scores, reverse=True)
            comps = [
                c
                for _, c in sorted(zip(gm.train_scores, gm.complexities), reverse=True)
            ]
            if not is_root:
                scores = scores[:top_k]
                comps = comps[:top_k]
            if not scores:
                continue

            gens.append(i + 1)
            all_scores.append(scores)
            all_comps.append(comps)

        if not gens:
            return None

        gens = np.array(gens)
        means = np.array([np.mean(s) for s in all_scores])
        stds = np.array([np.std(s) for s in all_scores])
        c_means = np.array([np.mean(c) for c in all_comps])
        c_stds = np.array([np.std(c) for c in all_comps])
        return gens, means, stds, c_means, c_stds

    def _smooth(arr, window=15):
        """Exponential moving average for smooth trend lines."""
        alpha = 2 / (window + 1)
        out = np.empty_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    def _draw_band(gens, means, stds, c_means, c_stds, color, label, depth):
        """Draw smoothed fitness band on ax_fit, complexity band on ax_comp."""
        if depth == 0:
            alpha_band, alpha_line, lw = 0.25, 1.0, 2.2
        else:
            alpha_band = max(0.06, 0.15 - (depth - 1) * 0.04)
            alpha_line = max(0.25, 0.6 - (depth - 1) * 0.12)
            lw = 1.3

        s_means = _smooth(means)
        s_stds = _smooth(stds)
        s_c_means = _smooth(c_means)
        s_c_stds = _smooth(c_stds)

        # Fitness
        ax_fit.fill_between(
            gens,
            s_means - s_stds,
            s_means + s_stds,
            color=color,
            alpha=alpha_band,
            zorder=3 - depth,
        )
        ax_fit.plot(
            gens,
            s_means,
            color=color,
            linewidth=lw,
            alpha=alpha_line,
            label=f"{label}",
            zorder=4 - depth,
        )

        # Complexity (same color, same style)
        ax_comp.fill_between(
            gens,
            s_c_means - s_c_stds,
            s_c_means + s_c_stds,
            color=color,
            alpha=alpha_band,
            zorder=3 - depth,
        )
        ax_comp.plot(
            gens,
            s_c_means,
            color=color,
            linewidth=lw,
            alpha=alpha_line,
            linestyle="--",
            label=f"{label}",
            zorder=4 - depth,
        )

    # --- Root population ---
    root_data = _collect_band_data(best_fold.generations, is_root=True)
    if root_data:
        _draw_band(*root_data, root_color, "Root", depth=0)

    # --- Collect children per depth, pick top N ---
    def _gather_children(gen_metrics_list, depth):
        if not gen_metrics_list:
            return
        n_children = len(gen_metrics_list[0].child_population_generation_metrics)
        candidates = []
        for child_idx in range(n_children):
            child_gens = []
            for gm in gen_metrics_list:
                if child_idx < len(gm.child_population_generation_metrics):
                    child_gens.append(gm.child_population_generation_metrics[child_idx])
            data = _collect_band_data(child_gens, is_root=False)
            if data:
                candidates.append((child_idx, child_gens, data))

        candidates.sort(
            key=lambda x: max(gm.best_train_score for gm in x[1]),
            reverse=True,
        )
        for rank, (child_idx, child_gens, data) in enumerate(
            candidates[:top_n_children]
        ):
            color = COLORS[(child_idx + 1 + (depth - 1) * 3) % len(COLORS)]
            label = f"Child {child_idx} (d={depth})"
            _draw_band(*data, color, label, depth=depth)

    _gather_children(best_fold.generations, depth=1)

    # Fitness subplot
    ax_fit.set_ylabel("Train Score")
    ax_fit.set_title(title, fontsize=13, fontweight="bold")
    ax_fit.legend(loc="lower right", fontsize=7, ncol=2)
    ax_fit.grid(True, alpha=0.3)

    # Complexity subplot
    ax_comp.set_xlabel("Generation")
    ax_comp.set_ylabel("Complexity (nodes)")
    ax_comp.grid(True, alpha=0.3)

    plt.tight_layout()
    sns.reset_defaults()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
