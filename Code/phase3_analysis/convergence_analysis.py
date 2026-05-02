"""Iterations-to-threshold grouped comparison across systems and conditions."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CommunicationMode

from ._style import COLORS, LABELS, apply_academic_style, resolve_output_path, safe_savefig

LOSS_THRESHOLD = 0.1


def _first_convergence_iter(history: List[MetricsSnapshot], threshold: float) -> float:
    if not history:
        return float("nan")
    for m in history:
        if m.loss < threshold:
            return float(m.iteration)
    return float(len(history))


def _normalize_mode(results_df: pd.DataFrame) -> pd.Series:
    return results_df["mode"].apply(lambda m: m.value if hasattr(m, "value") else m)


def generate_convergence_comparison(
    results_df: pd.DataFrame,
    output_dir: str,
    show_plots: bool = False,
    loss_threshold: float = LOSS_THRESHOLD,
) -> None:
    apply_academic_style()
    mode_col = _normalize_mode(results_df)
    all_conds = [
        ("homogeneous", "Homogeneous"),
        ("static", "Straggler"),
        ("worker_failure", "Failure"),
    ]
    conditions = [(c, t) for c, t in all_conds if (results_df["condition"] == c).any()]
    if not conditions:
        print("Phase 3 convergence: no matching conditions in results_df.")
        return
    systems = (
        CommunicationMode.ADAPTIVE,
        CommunicationMode.RING_ALLREDUCE,
        CommunicationMode.PARAMETER_SERVER,
    )
    mode_key = {
        CommunicationMode.ADAPTIVE: "adaptive",
        CommunicationMode.RING_ALLREDUCE: "ring_allreduce",
        CommunicationMode.PARAMETER_SERVER: "parameter_server",
    }

    x = np.arange(len(conditions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, system in enumerate(systems):
        means: List[float] = []
        stds: List[float] = []
        for cond, _ in conditions:
            subset = results_df[(mode_col == mode_key[system]) & (results_df["condition"] == cond)]
            vals: List[float] = []
            for _, row in subset.iterrows():
                history = row["history"]
                if isinstance(history, list) and history:
                    vals.append(_first_convergence_iter(history, loss_threshold))
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            else:
                means.append(float("nan"))
                stds.append(0.0)
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=LABELS[system],
            color=COLORS[system],
            alpha=0.88,
            capsize=4,
            ecolor="0.35",
        )

    ax.set_xlabel("Condition", fontweight="bold")
    ax.set_ylabel(f"Iterations to loss < {loss_threshold}", fontweight="bold")
    ax.set_title(
        "Iterations to convergence — lower is better",
        fontweight="bold",
        loc="left",
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([t[1] for t in conditions])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    out = resolve_output_path(output_dir, "phase3_convergence.png")
    safe_savefig(out, show_plots)
