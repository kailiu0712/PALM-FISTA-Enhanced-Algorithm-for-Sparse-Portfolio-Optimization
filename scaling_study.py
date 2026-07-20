"""Scaling study: find the problem size n at which the integrated solver
overtakes the CVXPY oracle in wall-clock time, and record the trend.

Runs N_TRIALS_PER_N independent random instances at each n for robustness
against run-to-run timing noise (see raw per-trial output and the
mean +/- std error bars in the plot).

Run from within this directory: python scaling_study.py
Writes:
  results/scaling_results_raw.csv     one row per (n, trial)
  results/scaling_results.csv         one row per n, aggregated over trials
  results/scaling_plot.png            mean speedup vs n, with std error bars

NOTE: with N_TRIALS_PER_N=10 trials this is ~10x the wall time of a single
pass. Based on a single-trial run, the full n=8..5000 grid took ~350s total,
so expect roughly 45-60 minutes for the whole 10-trial sweep (dominated by
the largest n, where a single CVXPY oracle solve alone takes 60-100+ seconds).
Both CSVs are written incrementally (flushed after every trial / every n) so
partial results are always on disk if you interrupt early.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

try:
    from .config import SolverConfig
    from .core import build_random_feasible_instance, exposure_violation
    from .solver import integrated_palm_solve, oracle_solve
except ImportError:  # pragma: no cover - script mode
    from config import SolverConfig
    from core import build_random_feasible_instance, exposure_violation
    from solver import integrated_palm_solve, oracle_solve

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_CSV_PATH = RESULTS_DIR / "scaling_results_raw.csv"
SUMMARY_CSV_PATH = RESULTS_DIR / "scaling_results.csv"
PLOT_PATH = RESULTS_DIR / "scaling_plot.png"

# n grows roughly geometrically from 8 to 5000; k is a fixed 5% sparsity level (min 2).
N_GRID = [8, 12, 20, 30, 45, 65, 100, 150, 225, 350, 500, 700, 1000, 1500, 2200, 3200, 4200, 5000]
N_TRIALS_PER_N = 10
BASE_SEED = 20260704
RESTART = True
SOLVER_CFG = SolverConfig(restart=RESTART)

RAW_FIELDNAMES = [
    "n", "trial", "seed", "k", "m",
    "oracle_time_sec", "integrated_time_sec", "speedup_vs_oracle",
    "objective_error", "relative_gap", "converged",
    "simplex_residual", "exposure_violation",
    "outer_iterations", "inner_iterations", "breakpoint_events",
]
SUMMARY_FIELDNAMES = [
    "n", "k", "m", "n_trials",
    "oracle_time_mean", "oracle_time_std",
    "integrated_time_mean", "integrated_time_std",
    "speedup_mean", "speedup_std", "speedup_min", "speedup_max",
    "speedup_of_means",
    "max_objective_error", "max_relative_gap",
    "max_simplex_residual", "max_exposure_violation",
    "converged_rate",
]


def run_one(n: int, trial: int, seed: int) -> Dict[str, object]:
    k = max(2, round(0.05 * n))
    instance = build_random_feasible_instance(instance_id=trial, seed=seed, n=n, k=k)

    t0 = time.perf_counter()
    oracle = oracle_solve(instance)
    oracle_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    integrated = integrated_palm_solve(instance, config=SOLVER_CFG)
    integrated_time = time.perf_counter() - t0

    obj_err = abs(integrated.objective - oracle.objective)
    return {
        "n": n,
        "trial": trial,
        "seed": seed,
        "k": k,
        "m": instance.m,
        "oracle_time_sec": oracle_time,
        "integrated_time_sec": integrated_time,
        "speedup_vs_oracle": oracle_time / max(integrated_time, 1e-12),
        "objective_error": obj_err,
        "relative_gap": obj_err / max(1.0, abs(oracle.objective)),
        "converged": integrated.converged,
        "simplex_residual": integrated.simplex_residual,
        "exposure_violation": exposure_violation(integrated.x, instance),
        "outer_iterations": integrated.outer_iterations,
        "inner_iterations": integrated.inner_iterations,
        "breakpoint_events": integrated.breakpoint_events,
    }


def summarize(n: int, trials: List[Dict[str, object]]) -> Dict[str, object]:
    oracle_times = [t["oracle_time_sec"] for t in trials]
    integrated_times = [t["integrated_time_sec"] for t in trials]
    speedups = [t["speedup_vs_oracle"] for t in trials]
    oracle_mean = statistics.fmean(oracle_times)
    integrated_mean = statistics.fmean(integrated_times)
    return {
        "n": n,
        "k": trials[0]["k"],
        "m": trials[0]["m"],
        "n_trials": len(trials),
        "oracle_time_mean": oracle_mean,
        "oracle_time_std": statistics.pstdev(oracle_times) if len(oracle_times) > 1 else 0.0,
        "integrated_time_mean": integrated_mean,
        "integrated_time_std": statistics.pstdev(integrated_times) if len(integrated_times) > 1 else 0.0,
        "speedup_mean": statistics.fmean(speedups),
        "speedup_std": statistics.pstdev(speedups) if len(speedups) > 1 else 0.0,
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "speedup_of_means": oracle_mean / max(integrated_mean, 1e-12),
        "max_objective_error": max(t["objective_error"] for t in trials),
        "max_relative_gap": max(t["relative_gap"] for t in trials),
        "max_simplex_residual": max(t["simplex_residual"] for t in trials),
        "max_exposure_violation": max(t["exposure_violation"] for t in trials),
        "converged_rate": sum(1 for t in trials if t["converged"]) / len(trials),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trials_by_n: Dict[int, List[Dict[str, object]]] = {n: [] for n in N_GRID}
    summary_rows: List[Dict[str, object]] = []

    jobs = [(n, trial) for n in N_GRID for trial in range(N_TRIALS_PER_N)]

    with RAW_CSV_PATH.open("w", newline="", encoding="utf-8") as raw_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=RAW_FIELDNAMES)
        raw_writer.writeheader()

        pbar = tqdm(jobs, unit="trial", desc="scaling study", ascii=True)
        for n, trial in pbar:
            pbar.set_description(f"n={n} trial {trial + 1}/{N_TRIALS_PER_N}")
            seed = BASE_SEED + 1_000_000 * n + trial

            row = run_one(n, trial, seed)
            trials_by_n[n].append(row)

            raw_writer.writerow(row)
            raw_handle.flush()
            pbar.set_postfix(
                speedup=f"{row['speedup_vs_oracle']:.2f}x",
                oracle=f"{row['oracle_time_sec']:.2f}s",
                integrated=f"{row['integrated_time_sec']:.2f}s",
            )

            if trial == N_TRIALS_PER_N - 1:
                summary_rows.append(summarize(n, trials_by_n[n]))
                write_summary_csv(summary_rows)
                make_plot(summary_rows)

    print(f"\nSaved per-trial table to {RAW_CSV_PATH}")
    print(f"Saved summary table to {SUMMARY_CSV_PATH}")
    print(f"Saved plot to {PLOT_PATH}")
    print_summary_table(summary_rows)


def write_summary_csv(summary_rows: List[Dict[str, object]]) -> None:
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)


def print_summary_table(summary_rows: List[Dict[str, object]]) -> None:
    print(f"\n{'n':>6}{'trials':>8}{'oracle(s)':>14}{'integrated(s)':>16}{'speedup':>12}{'converged':>12}")
    for row in summary_rows:
        print(
            f"{row['n']:>6}{row['n_trials']:>8}"
            f"{row['oracle_time_mean']:>9.3f}+/-{row['oracle_time_std']:<6.3f}"
            f"{row['integrated_time_mean']:>11.3f}+/-{row['integrated_time_std']:<6.3f}"
            f"{row['speedup_mean']:>9.2f}x"
            f"{row['converged_rate']:>11.0%}"
        )


def make_plot(summary_rows: List[Dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_vals = np.array([r["n"] for r in summary_rows], dtype=float)
    speedup_mean = np.array([r["speedup_mean"] for r in summary_rows], dtype=float)
    speedup_std = np.array([r["speedup_std"] for r in summary_rows], dtype=float)
    lower = np.clip(speedup_mean - speedup_std, 1e-3, None)
    upper = speedup_mean + speedup_std

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)

    ax.axhline(1.0, color="#9aa0a6", linewidth=1.5, linestyle="--", zorder=1)
    ax.text(
        n_vals[0], 1.0, "  breakeven (equal speed)", color="#5f6368", fontsize=9,
        va="bottom", ha="left",
    )

    ax.fill_between(n_vals, lower, upper, color="#3b6fd6", alpha=0.15, zorder=2, linewidth=0)
    ax.plot(n_vals, speedup_mean, color="#3b6fd6", linewidth=2, marker="o", markersize=5, zorder=3)
    ax.errorbar(
        n_vals, speedup_mean, yerr=speedup_std, fmt="none",
        ecolor="#3b6fd6", elinewidth=1, capsize=3, alpha=0.6, zorder=3,
    )

    crossover_idx = np.argmax(speedup_mean >= 1.0) if np.any(speedup_mean >= 1.0) else None
    if crossover_idx is not None and speedup_mean[crossover_idx] >= 1.0:
        ax.scatter(
            [n_vals[crossover_idx]], [speedup_mean[crossover_idx]],
            color="#1a7f37", s=70, zorder=4, label="first crossover (mean)",
        )
        ax.annotate(
            f"n={int(n_vals[crossover_idx])}",
            (n_vals[crossover_idx], speedup_mean[crossover_idx]),
            textcoords="offset points", xytext=(8, 8), fontsize=9, color="#1a7f37",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of assets n")
    ax.set_ylabel("Speedup vs CVXPY oracle (oracle time / integrated time)")
    ax.set_title(
        f"Integrated breakpoint-walk solver vs. CVXPY oracle\n"
        f"mean speedup ± std over {summary_rows[0]['n_trials']} random trials per n"
    )
    ax.grid(True, which="both", linewidth=0.5, alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)


if __name__ == "__main__":
    main()
