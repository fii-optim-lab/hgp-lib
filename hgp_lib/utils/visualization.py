"""
Visualization utilities for Optuna trial details.

This module provides functions to generate matplotlib plots for:
- Epoch progression (train/validation curves)
- All runs overlay with best run highlighted
- Hierarchical GP parent vs children progression
"""

from typing import List

import matplotlib.pyplot as plt

from hgp_lib.metrics import RunMetrics, TrainingHistory


def plot_epoch_progression(
    train_history: TrainingHistory,
    val_history: TrainingHistory | None = None,
    title: str = "F1 Score Progression",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Generate epoch progression plot for train and validation scores.

    Creates a line plot showing F1 score progression across epochs.
    Train scores are shown in blue, validation scores in orange.

    Args:
        train_history: Training history with per-epoch metrics.
        val_history: Optional validation history. If None, only train
            curve is plotted.
        title: Plot title. Default: "F1 Score Progression".
        save_path: Optional path to save the plot as PNG. If None,
            plot is not saved to disk.

    Returns:
        matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract train scores and epochs
    train_epochs = [e.epoch for e in train_history.epochs]
    train_scores = train_history.best_scores()

    # Plot train curve (blue)
    ax.plot(
        train_epochs,
        train_scores,
        color="blue",
        label="Train",
        linewidth=2,
    )

    # Plot validation curve (orange) if available
    if val_history is not None and val_history.epochs:
        val_epochs = [e.epoch for e in val_history.epochs]
        val_scores = val_history.best_scores()
        ax.plot(
            val_epochs,
            val_scores,
            color="orange",
            label="Validation",
            linewidth=2,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_all_runs_progression(
    run_metrics: List[RunMetrics],
    best_run_idx: int,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Generate overlay plot of all runs with best run highlighted.

    Creates a plot showing validation score progression for all runs
    with transparency. The best run is highlighted with a thicker line.

    Args:
        run_metrics: List of RunMetrics from all benchmark runs.
        best_run_idx: Index of the best run to highlight.
        save_path: Optional path to save the plot as PNG. If None,
            plot is not saved to disk.

    Returns:
        matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all runs with transparency
    for i, rm in enumerate(run_metrics):
        if rm.val_history is None or not rm.val_history.epochs:
            continue

        val_epochs = [e.epoch for e in rm.val_history.epochs]
        val_scores = rm.val_history.best_scores()

        if i == best_run_idx:
            # Highlight best run with thicker line
            ax.plot(
                val_epochs,
                val_scores,
                color="orange",
                linewidth=3,
                alpha=1.0,
                label="Best Run",
                zorder=10,
            )
        else:
            ax.plot(
                val_epochs,
                val_scores,
                color="blue",
                linewidth=1,
                alpha=0.3,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1 Score")
    ax.set_title("All Runs Validation Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_hierarchical_progression(
    train_history: TrainingHistory,
    num_children: int,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Generate parent vs children score progression plot for hierarchical GP.

    Creates a plot showing the parent (root) population best score as a
    solid line and each child population's best score as dashed lines
    with distinct colors.

    Args:
        train_history: Training history containing children_best_scores
            in each epoch's metrics.
        num_children: Number of child populations.
        save_path: Optional path to save the plot as PNG. If None,
            plot is not saved to disk.

    Returns:
        matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract epochs and parent scores
    epochs = [e.epoch for e in train_history.epochs]
    parent_scores = train_history.best_scores()

    # Plot parent (solid line, prominent)
    ax.plot(
        epochs,
        parent_scores,
        color="black",
        linewidth=2.5,
        linestyle="-",
        label="Parent",
    )

    # Color palette for children
    child_colors = plt.cm.tab10.colors

    # Extract and plot children scores
    for child_idx in range(num_children):
        child_scores = []
        for epoch_metrics in train_history.epochs:
            if epoch_metrics.children_best_scores is not None and child_idx < len(
                epoch_metrics.children_best_scores
            ):
                child_scores.append(epoch_metrics.children_best_scores[child_idx])
            else:
                child_scores.append(None)

        # Filter out None values for plotting
        valid_epochs = [e for e, s in zip(epochs, child_scores) if s is not None]
        valid_scores = [s for s in child_scores if s is not None]

        if valid_scores:
            color = child_colors[child_idx % len(child_colors)]
            ax.plot(
                valid_epochs,
                valid_scores,
                color=color,
                linewidth=1.5,
                linestyle="--",
                label=f"Child {child_idx + 1}",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("Hierarchical GP: Parent vs Children Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
