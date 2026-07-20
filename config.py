"""Centralized configuration for the integrated portfolio solver and benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SolverConfig:
    """Parameters for the proximal ALM solver."""

    max_outer: int = 200
    max_inner: int = 200
    restart: bool = True
    outer_tol: float = 1e-7
    inner_tol: float = 1e-7
    beta: float = 0.7
    xi_rho: float = 2.0
    rho0: float = 1.0
    eta0: float = 1.0
    alpha_growth: float = 1.2
    delta_eta: float = 0.1


@dataclass
class BenchmarkConfig:
    """Parameters for the benchmark runner."""

    n_values: List[int] = field(default_factory=lambda: [8, 12])
    instances_per_setting: int = 1
    base_seed: int = 20260617
    m_constraints: Optional[int] = None
    use_scaling_k: bool = False
    csv_path: Optional[Path] = None
    quiet: bool = False
    solver: SolverConfig = field(default_factory=SolverConfig)
