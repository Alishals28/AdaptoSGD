"""Amdahl's law vs observed speedup for three communication strategies."""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.config import CommunicationMode, SystemConfig

from ._data import throughput_grid_worker_counts
from ._style import COLORS, LABELS, apply_academic_style, resolve_output_path, safe_savefig


def _amdahl_speedup(n: np.ndarray, f: float) -> np.ndarray:
    denom = f + (1.0 - f) / n
    return 1.0 / denom


def _implied_serial_fraction(speedup: float, n: int) -> float:
    if n <= 1 or speedup <= 0:
        return float("nan")
    return max(0.0, (n / speedup - 1.0) / (n - 1.0))


def generate_amdahl_plot(
    base_config: SystemConfig,
    output_dir: str,
    show_plots: bool = False,
    worker_counts: Tuple[int, ...] = (1, 2, 4, 8),
) -> None:
    apply_academic_style()
    modes = (
        CommunicationMode.ADAPTIVE,
        CommunicationMode.RING_ALLREDUCE,
        CommunicationMode.PARAMETER_SERVER,
    )

    grid = throughput_grid_worker_counts(
        base_config, list(worker_counts), modes, "homogeneous"
    )

    speedups: Dict[CommunicationMode, Dict[int, float]] = {m: {} for m in modes}
    for mode in modes:
        t1 = grid[mode].get(1)
        if not t1 or t1 <= 0:
            continue
        for n in worker_counts:
            if n in grid[mode]:
                speedups[mode][n] = grid[mode][n] / t1

    n_theory = np.arange(1, 17, dtype=float)
    f_values = (0.05, 0.10, 0.20)

    fig, ax = plt.subplots(figsize=(10, 6))
    styles = ["-", "--", ":"]
    for f, ls in zip(f_values, styles):
        ax.plot(
            n_theory,
            _amdahl_speedup(n_theory, f),
            linestyle=ls,
            linewidth=2.2,
            color="0.35",
            label=f"Amdahl $f$={f:.2f}",
        )

    markers = ["o", "s", "^"]
    for mode, mk in zip(modes, markers):
        xs: List[int] = []
        ys: List[float] = []
        for n in worker_counts:
            if n in speedups.get(mode, {}):
                xs.append(n)
                ys.append(speedups[mode][n])
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=120,
            color=COLORS[mode],
            marker=mk,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=f"{LABELS[mode]} (observed)",
        )
        for n, s in zip(xs, ys):
            f_hat = _implied_serial_fraction(s, n)
            if np.isfinite(f_hat):
                ax.annotate(
                    rf"$\hat f$={f_hat:.2f}",
                    (n, s),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    color=COLORS[mode],
                )

    ax.set_xlabel("Workers $N$", fontweight="bold")
    ax.set_ylabel("Speedup $S(N)$", fontweight="bold")
    ax.set_title("Amdahl bound vs. empirical speedup (homogeneous)", fontweight="bold", loc="left")
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="lower right", framealpha=0.95)

    out = resolve_output_path(output_dir, "phase3_amdahl.png")
    safe_savefig(out, show_plots)
