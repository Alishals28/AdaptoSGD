"""Homogeneous monitoring overhead for AdaptoSGD vs pure AllReduce."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ._style import apply_academic_style, resolve_output_path, safe_savefig


def compute_monitoring_overhead(
    results_df: pd.DataFrame,
    output_dir: str,
    show_plots: bool = False,
    negligible_threshold_pct: float = 5.0,
) -> float:
    """
    Overhead % = (AllReduce_throughput - AdaptoSGD_throughput) / AllReduce_throughput * 100
    using aggregated runs under the homogeneous condition only.
    """
    apply_academic_style()
    mode_col = results_df["mode"].apply(lambda m: m.value if hasattr(m, "value") else m)
    hom = results_df["condition"] == "homogeneous"

    ar_tp = results_df[(mode_col == "ring_allreduce") & hom]["avg_throughput"].mean()
    ad_tp = results_df[(mode_col == "adaptive") & hom]["avg_throughput"].mean()

    if pd.isna(ar_tp) or ar_tp == 0:
        overhead_pct = 0.0
    else:
        overhead_pct = float((ar_tp - ad_tp) / ar_tp * 100.0)

    if overhead_pct < negligible_threshold_pct:
        print(
            f"Monitoring overhead is negligible ({overhead_pct:.2f}%) — "
            "AdaptoSGD pays near-zero cost for adaptability under homogeneous conditions"
        )
    else:
        print(f"Monitoring overhead: {overhead_pct:.2f}% (AllReduce vs AdaptoSGD, homogeneous).")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    systems = ["AllReduce", "AdaptoSGD"]
    tps = [float(ar_tp) if pd.notna(ar_tp) else 0.0, float(ad_tp) if pd.notna(ad_tp) else 0.0]
    bar_colors = ["#E74C3C", "#2ECC71"]
    ax.bar(systems, tps, color=bar_colors, alpha=0.88, edgecolor="0.25", linewidth=0.8)
    ax.set_ylabel("Mean throughput (iter/s)", fontweight="bold")
    ax.set_title("Homogeneous throughput — monitoring cost", fontweight="bold", loc="left")
    ax.grid(True, axis="y", alpha=0.3)

    out = resolve_output_path(output_dir, "phase3_overhead.png")
    safe_savefig(out, show_plots)

    return overhead_pct
