"""
Phase 3 analysis utilities: scalability bounds, overhead, and convergence figures.
"""

from .amdahl_analysis import generate_amdahl_plot
from .convergence_analysis import generate_convergence_comparison
from .failure_analysis import generate_extended_failure_visualizations
from .overhead_analysis import compute_monitoring_overhead
from .scalability_analysis import generate_speedup_efficiency_plot

__all__ = [
    "generate_amdahl_plot",
    "generate_speedup_efficiency_plot",
    "generate_convergence_comparison",
    "compute_monitoring_overhead",
    "generate_extended_failure_visualizations",
]
