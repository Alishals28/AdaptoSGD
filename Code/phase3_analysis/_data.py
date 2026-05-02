"""Collect supplemental throughput measurements for phase-3 plots."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.analysis import compute_monitoring_overhead as baseline_overhead_pair
from src.config import CommunicationMode, SystemConfig

_CODE_DIR = Path(__file__).resolve().parent.parent
_MAIN_MOD = None


def _main_module():
    global _MAIN_MOD
    if _MAIN_MOD is None:
        import __main__ as boot

        if hasattr(boot, "ExperimentRunner"):
            _MAIN_MOD = boot
        else:
            spec = importlib.util.spec_from_file_location("adaptosgd_main", _CODE_DIR / "main.py")
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            _MAIN_MOD = mod
    return _MAIN_MOD


def _scaling_iterations(base: SystemConfig) -> int:
    cap = 28 if base.num_iterations <= 55 else 100
    return max(15, min(base.num_iterations, cap))


def throughput_grid_worker_counts(
    base_config: SystemConfig,
    worker_counts: List[int],
    modes: Tuple[CommunicationMode, ...],
    condition: str,
) -> Dict[CommunicationMode, Dict[int, float]]:
    ExperimentRunner = _main_module().ExperimentRunner
    out: Dict[CommunicationMode, Dict[int, float]] = {m: {} for m in modes}
    for n in worker_counts:
        for mode in modes:
            cfg = dataclasses.replace(
                base_config,
                num_workers=n,
                num_iterations=_scaling_iterations(base_config),
            )
            runner = ExperimentRunner(cfg)
            result = runner.run_experiment(mode, condition)
            out[mode][n] = float(result.avg_throughput)
    return out


def homogeneous_overhead_from_results_df(results_df: pd.DataFrame) -> Tuple[float, float]:
    return baseline_overhead_pair(results_df)
