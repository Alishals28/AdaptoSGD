from typing import Tuple

import pandas as pd

def compute_monitoring_overhead(results_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculates the performance overhead of the adaptive monitoring system.
    
    This is done by comparing the throughput of AdaptoSGD against the optimal
    baseline (Ring AllReduce) in a homogeneous environment where no adaptation
    is necessary. The difference represents the "cost of being adaptive."
    
    Returns:
        A tuple containing:
        - The absolute overhead in iterations/second.
        - The relative overhead as a percentage.
    """
    
    # Throughput of pure Ring AllReduce in the ideal (homogeneous) case
    mode_values = results_df["mode"].apply(
        lambda mode: mode.value if hasattr(mode, "value") else mode
    )
    allreduce_ideal_tp = results_df[
        (mode_values == "ring_allreduce")
        & (results_df["condition"] == "homogeneous")
    ]["avg_throughput"].mean()
    
    # Throughput of AdaptoSGD in the same ideal case
    adaptive_ideal_tp = results_df[
        (mode_values == "adaptive")
        & (results_df["condition"] == "homogeneous")
    ]["avg_throughput"].mean()
    
    if allreduce_ideal_tp == 0 or pd.isna(allreduce_ideal_tp):
        return 0.0, 0.0
        
    # The overhead is the performance lost by running the monitoring logic
    absolute_overhead = allreduce_ideal_tp - adaptive_ideal_tp
    
    # Express overhead as a percentage of the optimal performance
    relative_overhead_pct = (absolute_overhead / allreduce_ideal_tp) * 100
    
    return absolute_overhead, relative_overhead_pct
