"""Speedup and parallel efficiency across worker counts."""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.config import CommunicationMode, SystemConfig

from ._data import throughput_grid_worker_counts
from ._style import COLORS, LABELS, apply_academic_style, resolve_output_path, safe_savefig


def _speedup_efficiency(
    throughput_by_n: Dict[int, float], worker_counts: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1 = throughput_by_n.get(1)
    if not t1 or t1 <= 0:
        return np.array([]), np.array([]), np.array([])
    n_arr = np.array([n for n in worker_counts if n in throughput_by_n], dtype=float)
    s = np.array([throughput_by_n[int(n)] / t1 for n in n_arr])
    eff = s / n_arr
    return n_arr, s, eff


def generate_speedup_efficiency_plot(
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
    ws = list(worker_counts)

    grid_hom = throughput_grid_worker_counts(base_config, ws, modes, "homogeneous")
    grid_straggler = throughput_grid_worker_counts(base_config, ws, modes, "static")

    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(14, 5.5))

    n_ideal = np.array(ws, dtype=float)
    ax_s.plot(
        n_ideal,
        n_ideal,
        color="0.45",
        linestyle="--",
        linewidth=2,
        label="Ideal $S=N$",
    )

    for mode in modes:
        n_h, s_h, _ = _speedup_efficiency(grid_hom[mode], ws)
        if len(n_h):
            ax_s.plot(
                n_h,
                s_h,
                marker="o",
                linewidth=2.2,
                markersize=8,
                color=COLORS[mode],
                label=LABELS[mode],
            )

    ax_s.set_xlabel("Workers $N$", fontweight="bold")
    ax_s.set_ylabel("Speedup", fontweight="bold")
    ax_s.set_title("Strong scaling — speedup (homogeneous)", fontweight="bold", loc="left")
    ax_s.legend(loc="upper left", fontsize=9)
    ax_s.set_xticks(ws)

    for mode in modes:
        n_h, _, eff_h = _speedup_efficiency(grid_hom[mode], ws)
        n_st, _, eff_st = _speedup_efficiency(grid_straggler[mode], ws)
        if len(n_h):
            ax_e.plot(
                n_h,
                eff_h,
                marker="o",
                linewidth=2,
                color=COLORS[mode],
                linestyle="-",
                label=f"{LABELS[mode]} homog.",
            )
        if len(n_st):
            ax_e.plot(
                n_st,
                eff_st,
                marker="^",
                linewidth=2,
                color=COLORS[mode],
                linestyle="--",
                alpha=0.9,
                label=f"{LABELS[mode]} straggler",
            )

    ax_e.axhline(1.0, color="0.5", linestyle=":", linewidth=1.5)

    ax_e.annotate(
        "Straggler: synchronous AllReduce\nbars on the slowest worker;\nAdaptoSGD adapts and retains\nhigher parallel efficiency.",
        xy=(0.56, 0.42),
        xycoords="axes fraction",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.9),
    )

    ax_e.set_xlabel("Workers $N$", fontweight="bold")
    ax_e.set_ylabel("Parallel efficiency $S/N$", fontweight="bold")
    ax_e.set_title("Parallel efficiency", fontweight="bold", loc="left")
    handles, labels = ax_e.get_legend_handles_labels()
    seen: Dict[str, object] = {}
    for h, lab in zip(handles, labels):
        seen[lab] = h
    ax_e.legend(seen.values(), seen.keys(), loc="upper right", fontsize=7)
    ax_e.set_xticks(ws)
    ax_e.set_ylim(0.0, None)

    out = resolve_output_path(output_dir, "phase3_scalability.png")
    safe_savefig(out, show_plots)
