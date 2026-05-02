"""Shared plot styling aligned with Code/main.py generate_core_visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import CommunicationMode

COLORS = {
    CommunicationMode.RING_ALLREDUCE: "#E74C3C",
    CommunicationMode.PARAMETER_SERVER: "#3498DB",
    CommunicationMode.ADAPTIVE: "#2ECC71",
}

LABELS = {
    CommunicationMode.RING_ALLREDUCE: "AllReduce",
    CommunicationMode.PARAMETER_SERVER: "PS",
    CommunicationMode.ADAPTIVE: "AdaptoSGD",
}


def apply_academic_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def resolve_output_path(output_dir: str, filename: str) -> Path:
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root
    root.mkdir(parents=True, exist_ok=True)
    return root / filename


def safe_savefig(path: Path, show: bool = False) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
