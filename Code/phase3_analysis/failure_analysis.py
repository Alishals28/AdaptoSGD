"""Extended failure-window analysis for Phase 3 reporting."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import CommunicationMode, SystemConfig, SystemConfig

from ._style import COLORS, LABELS, apply_academic_style, resolve_output_path, safe_savefig


def _baseline_loss_pre_failure(history, failure_starts_at: int, tail: int = 12) -> float:
    pre = [m.loss for m in history if m.iteration < failure_starts_at]
    if not pre:
        return 0.0
    return float(np.mean(pre[-tail:]))


def _recovery_iterations_to_baseline(
    history, cfg: SystemConfig, tol: float = 0.04
) -> Optional[int]:
    baseline = _baseline_loss_pre_failure(history, cfg.failure_starts_at)
    for m in history:
        if m.iteration <= cfg.failure_ends_at:
            continue
        if m.loss <= baseline + tol:
            return m.iteration - cfg.failure_ends_at
    return None


def generate_extended_failure_visualizations(
    results_df: pd.DataFrame,
    output_dir: str,
    show_plots: bool = False,
) -> None:
    apply_academic_style()
    subset = results_df[results_df["condition"] == "worker_failure"]
    if subset.empty:
        print("Phase 3 failure analysis: no worker_failure rows in results_df.")
        return

    systems = (
        CommunicationMode.RING_ALLREDUCE,
        CommunicationMode.PARAMETER_SERVER,
        CommunicationMode.ADAPTIVE,
    )

    cfg0: SystemConfig = subset.iloc[0]["config"]

    fig = plt.figure(figsize=(18, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.85, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    for mode in systems:
        run_data = subset[subset["mode"] == mode]
        if run_data.empty:
            continue
        history = run_data.iloc[0]["history"]
        ax1.plot(
            [m.iteration for m in history],
            [m.loss for m in history],
            label=LABELS[mode],
            color=COLORS[mode],
            linewidth=2.2,
        )

    ax1.axvspan(
        cfg0.failure_starts_at,
        cfg0.failure_ends_at,
        alpha=0.22,
        color="0.35",
        label="Failure window",
    )
    ax1.set_xlabel("Iteration", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Loss stability during failure", fontweight="bold", loc="left")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    recovery: Dict[str, Optional[int]] = {}
    for mode in systems:
        run_data = subset[subset["mode"] == mode]
        if run_data.empty:
            recovery[LABELS[mode]] = None
            continue
        history = run_data.iloc[0]["history"]
        cfg = run_data.iloc[0]["config"]
        recovery[LABELS[mode]] = _recovery_iterations_to_baseline(history, cfg)

    labels_list = [LABELS[m] for m in systems]
    values = [recovery[l] if recovery.get(l) is not None else 0 for l in labels_list]
    colors_list = [COLORS[m] for m in systems]
    bars = ax2.bar(labels_list, values, color=colors_list, alpha=0.88, edgecolor="0.25")
    ax2.set_ylabel("Iterations after failure clears", fontweight="bold")
    ax2.set_title("Recovery to pre-failure loss band", fontweight="bold", loc="left")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, lab in zip(bars, labels_list):
        h = bar.get_height()
        note = f"{int(h)} iters" if recovery.get(lab) is not None else "n/a"
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.3, note, ha="center", fontsize=9)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_title("Mechanistic comparison (dropped worker)", fontweight="bold", loc="left")

    def _box(ax, x, y, w, h, text, fc):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="0.25",
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    _box(ax3, 0.05, 0.62, 0.38, 0.2, "Parameter Server\n(update rule waits on\nslow/missing workers)", "#d6eaf8")
    _box(ax3, 0.57, 0.62, 0.38, 0.2, "AdaptoSGD\n(re-weights active workers;\ncontinues with partial sync)", "#d5f4e6")

    arrow_ps = FancyArrowPatch(
        (0.52, 0.5), (0.52, 0.22), arrowstyle="->", mutation_scale=16, linewidth=1.4, color="0.35"
    )
    arrow_ad = FancyArrowPatch(
        (0.76, 0.6), (0.76, 0.32), arrowstyle="->", mutation_scale=16, linewidth=1.4, color="0.35"
    )
    ax3.add_patch(arrow_ps)
    ax3.add_patch(arrow_ad)
    ax3.text(0.52, 0.15, "Barrier / straggler stall", ha="center", va="top", fontsize=10, style="italic")
    ax3.text(0.76, 0.2, "Monitor detects fault →\nPS-style updates with\nSSP + re-weighting", ha="center", va="top", fontsize=9)

    fig.suptitle("Phase 3 — failure scenario", fontsize=14, fontweight="bold", y=1.02)

    out = resolve_output_path(output_dir, "phase3_failure.png")
    safe_savefig(out, show_plots)
