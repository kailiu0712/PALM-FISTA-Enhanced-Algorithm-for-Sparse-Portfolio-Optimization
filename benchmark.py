from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from .config import BenchmarkConfig, SolverConfig
    from .core import PortfolioInstance, build_random_feasible_instance
    from .solver import integrated_palm_solve, oracle_solve
except ImportError:  # pragma: no cover - script mode
    from config import BenchmarkConfig, SolverConfig
    from core import PortfolioInstance, build_random_feasible_instance
    from solver import integrated_palm_solve, oracle_solve

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONVERGENCE_PLOT_PATH = RESULTS_DIR / "convergence_plot.png"
FROZEN_INNER_GAP_PLOT_PATH = RESULTS_DIR / "frozen_inner_objective_gap.png"
FROZEN_INNER_VALUE_PLOT_PATH = RESULTS_DIR / "frozen_inner_objective_values.png"


@dataclass
class BenchmarkRow:
    """One benchmark comparison row."""

    instance_id: int
    seed: int
    n: int
    m: int
    k: int
    oracle_objective: float
    integrated_objective: float
    objective_error: float
    relative_gap: float
    x_error: float
    simplex_residual: float
    exposure_violation: float
    nonnegativity_violation: float
    budget_residual: float
    outer_iterations: int
    inner_iterations: int
    oracle_time_sec: float
    integrated_time_sec: float
    speedup_vs_oracle: float
    converged: bool
    breakpoint_events: int
    merge_events: int
    regime_changes: int


def summarize_metric(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return max, mean, median."""

    return float(max(values)), float(np.mean(values)), float(statistics.median(values))


def write_results_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    """Write benchmark rows to CSV."""

    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_convergence_plot(
    instance: PortfolioInstance,
    objective_trace: Sequence[float],
    outer_trace_indices: Sequence[int],
    oracle_objective: float,
    oracle_solver: str,
    path: Path = CONVERGENCE_PLOT_PATH,
) -> None:
    """Save one convergence plot against the CVXPY oracle optimum."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace = np.asarray(objective_trace, dtype=float)
    if trace.size == 0:
        return

    gap = np.abs(trace - oracle_objective) / max(1.0, abs(oracle_objective))
    gap = np.maximum(gap, 1e-16)
    iterations = np.arange(trace.size, dtype=int)
    outer_idx = np.asarray(outer_trace_indices, dtype=int)
    outer_idx = outer_idx[(outer_idx >= 0) & (outer_idx < gap.size)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    ax.plot(
        iterations,
        gap,
        color="#2f6db3",
        linewidth=1.6,
        label="Integrated inner iterations",
    )
    if outer_idx.size > 0:
        ax.scatter(
            outer_idx,
            gap[outer_idx],
            color="#c2410c",
            s=26,
            zorder=4,
            label="Outer iteration endpoints",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative inner FISTA iteration")
    ax.set_ylabel("Relative objective gap to CVXPY optimum")
    ax.set_title(
        "Iteration vs. Optimality Gap\n"
        f"instance={instance.instance_id}, n={instance.n}, m={instance.m}, k={instance.k}"
    )
    ax.grid(True, which="both", linewidth=0.5, alpha=0.3)
    ax.legend()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _relative_gap_to_reference(values: np.ndarray, reference: float) -> np.ndarray:
    """Compute a stable relative gap to a finite reference value."""

    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.isfinite(reference):
        return out
    scale = max(1.0, abs(reference))
    out[finite] = np.maximum((values[finite] - reference) / scale, 1e-16)
    return out


def write_frozen_inner_objective_plots(
    instance: PortfolioInstance,
    x_objective_trace: Sequence[float],
    y_objective_trace: Sequence[float],
    gap_path: Path = FROZEN_INNER_GAP_PLOT_PATH,
    value_path: Path = FROZEN_INNER_VALUE_PLOT_PATH,
) -> None:
    """Save paper-style diagnostics for one frozen outer subproblem."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = np.asarray(x_objective_trace, dtype=float)
    y_values = np.asarray(y_objective_trace, dtype=float)
    if x_values.size == 0:
        return

    finite_values = np.concatenate([x_values[np.isfinite(x_values)], y_values[np.isfinite(y_values)]])
    if finite_values.size == 0:
        return
    reference = float(np.min(finite_values))

    x_iters = np.arange(x_values.size, dtype=int)
    y_iters = np.arange(y_values.size, dtype=int)
    x_gap = _relative_gap_to_reference(x_values, reference)
    y_gap = _relative_gap_to_reference(y_values, reference)
    finite_x = np.isfinite(x_values)
    finite_y = np.isfinite(y_values)
    finite_x_gap = np.isfinite(x_gap)
    finite_y_gap = np.isfinite(y_gap)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    ax.plot(
        x_iters[finite_x_gap],
        x_gap[finite_x_gap],
        color="#2563eb",
        linewidth=1.8,
        label="Prox iterates $x^{(t)}$",
    )
    ax.plot(
        y_iters[finite_y_gap],
        y_gap[finite_y_gap],
        color="#dc2626",
        linewidth=1.3,
        alpha=0.9,
        label="Extrapolated points $y^{(t)}$",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Inner FISTA iteration on frozen outer subproblem")
    ax.set_ylabel("Relative gap to best observed frozen-subproblem objective")
    ax.set_title(
        "Frozen Outer Subproblem: Objective Gap\n"
        f"instance={instance.instance_id}, n={instance.n}, m={instance.m}, k={instance.k}, outer=1"
    )
    ax.grid(True, which="both", linewidth=0.5, alpha=0.3)
    ax.legend()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(gap_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    ax.plot(
        x_iters[finite_x],
        x_values[finite_x],
        color="#2563eb",
        linewidth=1.8,
        label="Prox iterates $x^{(t)}$",
    )
    ax.plot(
        y_iters[finite_y],
        y_values[finite_y],
        color="#dc2626",
        linewidth=1.3,
        alpha=0.9,
        label="Extrapolated points $y^{(t)}$",
    )
    ax.axhline(reference, color="#6b7280", linewidth=1.2, linestyle="--", label="Best observed value")
    ax.set_xlabel("Inner FISTA iteration on frozen outer subproblem")
    ax.set_ylabel("Frozen-subproblem objective value")
    ax.set_title(
        "Frozen Outer Subproblem: Objective Values\n"
        f"instance={instance.instance_id}, n={instance.n}, m={instance.m}, k={instance.k}, outer=1"
    )
    ax.grid(True, which="both", linewidth=0.5, alpha=0.3)
    ax.legend()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(value_path)
    plt.close(fig)


def run_benchmark(cfg: BenchmarkConfig) -> List[BenchmarkRow]:
    """Run the integrated solver against the CVXPY oracle."""

    rows: List[BenchmarkRow] = []
    last_plot_payload: Tuple[
        PortfolioInstance,
        Sequence[float],
        Sequence[int],
        float,
        str,
        Sequence[float],
        Sequence[float],
    ] | None = None
    for n in cfg.n_values:
        if cfg.use_scaling_k:
            k_values = [max(2, round(0.05 * n))]
        else:
            max_k = min(5, n - 1)
            k_values = list(range(1, max_k + 1))
        for k in k_values:
            for repeat in range(cfg.instances_per_setting):
                instance_id = len(rows)
                seed = cfg.base_seed + 1009 * instance_id + repeat
                instance = build_random_feasible_instance(
                    instance_id=instance_id,
                    seed=seed,
                    n=n,
                    k=k,
                    m=cfg.m_constraints,
                )
                oracle = oracle_solve(instance, verbose=False)
                integrated = integrated_palm_solve(
                    instance,
                    config=cfg.solver,
                )
                last_plot_payload = (
                    instance,
                    integrated.objective_trace,
                    integrated.outer_trace_indices,
                    oracle.objective,
                    oracle.solver,
                    integrated.frozen_inner_x_objective_trace,
                    integrated.frozen_inner_y_objective_trace,
                )

                objective_error = abs(integrated.objective - oracle.objective)
                relative_gap = objective_error / max(1.0, abs(oracle.objective))
                x_error = float(np.linalg.norm(integrated.x - oracle.x))
                speedup = oracle.solve_time_sec / max(integrated.solve_time_sec, 1e-12)
                row = BenchmarkRow(
                    instance_id=instance.instance_id,
                    seed=instance.seed,
                    n=instance.n,
                    m=instance.m,
                    k=instance.k,
                    oracle_objective=oracle.objective,
                    integrated_objective=integrated.objective,
                    objective_error=objective_error,
                    relative_gap=relative_gap,
                    x_error=x_error,
                    simplex_residual=integrated.simplex_residual,
                    exposure_violation=integrated.exposure_violation,
                    nonnegativity_violation=integrated.nonnegativity_violation,
                    budget_residual=integrated.budget_residual,
                    outer_iterations=integrated.outer_iterations,
                    inner_iterations=integrated.inner_iterations,
                    oracle_time_sec=oracle.solve_time_sec,
                    integrated_time_sec=integrated.solve_time_sec,
                    speedup_vs_oracle=speedup,
                    converged=integrated.converged,
                    breakpoint_events=integrated.breakpoint_events,
                    merge_events=integrated.merge_events,
                    regime_changes=integrated.regime_changes,
                )
                rows.append(row)

                if not cfg.quiet:
                    print_instance_result(instance, row)

    if last_plot_payload is not None:
        write_convergence_plot(*last_plot_payload[:5])
        write_frozen_inner_objective_plots(
            last_plot_payload[0],
            last_plot_payload[5],
            last_plot_payload[6],
        )
        print(f"Saved convergence plot to {CONVERGENCE_PLOT_PATH}")
        print(f"Saved frozen-inner gap plot to {FROZEN_INNER_GAP_PLOT_PATH}")
        print(f"Saved frozen-inner value plot to {FROZEN_INNER_VALUE_PLOT_PATH}")

    if cfg.csv_path is not None:
        write_results_csv(cfg.csv_path, [asdict(row) for row in rows])
        print(f"\nSaved benchmark results to {cfg.csv_path}")

    print_summary(rows)
    return rows


def print_instance_result(instance: PortfolioInstance, row: BenchmarkRow) -> None:
    """Print one compact result line."""

    print(
        f"instance={instance.instance_id} seed={instance.seed} n={instance.n} m={instance.m} k={instance.k} "
        f"obj_err={row.objective_error:.3e} x_err={row.x_error:.3e} "
        f"exp_violation={row.exposure_violation:.3e} speedup={row.speedup_vs_oracle:.3f} "
        f"outer={row.outer_iterations} inner={row.inner_iterations} converged={row.converged}"
    )


def print_summary(rows: Sequence[BenchmarkRow]) -> None:
    """Print aggregate summary."""

    objective_errors = [row.objective_error for row in rows]
    relative_gaps = [row.relative_gap for row in rows]
    x_errors = [row.x_error for row in rows]
    simplex = [row.simplex_residual for row in rows]
    exposure = [row.exposure_violation for row in rows]
    nonnegativity = [row.nonnegativity_violation for row in rows]
    budget = [row.budget_residual for row in rows]
    oracle_time = [row.oracle_time_sec for row in rows]
    integrated_time = [row.integrated_time_sec for row in rows]
    speedups = [row.speedup_vs_oracle for row in rows]
    breakpoint_events = [row.breakpoint_events for row in rows]
    merges = [row.merge_events for row in rows]
    regime_changes = [row.regime_changes for row in rows]
    converged_rate = sum(1 for row in rows if row.converged) / max(len(rows), 1)

    print("\nIntegrated solver benchmark summary")
    print("-" * 88)
    print(f"{'metric':<28}{'max':>16}{'mean':>16}{'median':>16}")
    for name, values in [
        ("objective_error", objective_errors),
        ("relative_gap", relative_gaps),
        ("x_error", x_errors),
        ("simplex_residual", simplex),
        ("exposure_violation", exposure),
        ("nonnegativity", nonnegativity),
        ("budget_residual", budget),
        ("oracle_time_sec", oracle_time),
        ("integrated_time_sec", integrated_time),
        ("speedup_vs_oracle", speedups),
        ("breakpoint_events", [float(v) for v in breakpoint_events]),
        ("merge_events", [float(v) for v in merges]),
        ("regime_changes", [float(v) for v in regime_changes]),
    ]:
        v_max, v_mean, v_median = summarize_metric(values)
        print(f"{name:<28}{v_max:>16.3e}{v_mean:>16.3e}{v_median:>16.3e}")
    print(f"{'converged_rate':<28}{converged_rate:>16.3%}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", type=int, nargs="+", default=[8, 12], help="Problem sizes.")
    parser.add_argument(
        "--instances-per-setting",
        type=int,
        default=1,
        help="Instances per (n, k) setting.",
    )
    parser.add_argument("--base-seed", type=int, default=20260617, help="Base RNG seed.")
    parser.add_argument(
        "--m-constraints",
        type=int,
        default=None,
        help="Number of rows in A. Default picks a modest value based on n.",
    )
    parser.add_argument(
        "--use-scaling-k",
        action="store_true",
        help="Use the scaling-study sparsity rule k=max(2, round(0.05*n)) instead of looping over k=1..min(5,n-1).",
    )
    parser.add_argument("--csv-path", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--quiet", action="store_true", help="Print only summary.")
    parser.add_argument("--max-outer", type=int, default=200, help="Maximum PALM outer iterations.")
    parser.add_argument("--max-inner", type=int, default=200, help="Maximum FISTA inner iterations.")
    parser.add_argument("--restart", dest="restart", action="store_true", help="Enable value-based restart in the inner FISTA loop.")
    parser.add_argument("--no-restart", dest="restart", action="store_false", help="Disable value-based restart in the inner FISTA loop.")
    parser.set_defaults(restart=True)
    parser.add_argument("--outer-tol", type=float, default=1e-7, help="Outer stopping tolerance.")
    parser.add_argument("--inner-tol", type=float, default=1e-7, help="Inner FISTA stopping tolerance.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    solver_cfg = SolverConfig(
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        restart=args.restart,
        outer_tol=args.outer_tol,
        inner_tol=args.inner_tol,
    )
    cfg = BenchmarkConfig(
        n_values=args.n_values,
        instances_per_setting=args.instances_per_setting,
        base_seed=args.base_seed,
        m_constraints=args.m_constraints,
        use_scaling_k=args.use_scaling_k,
        csv_path=args.csv_path,
        quiet=args.quiet,
        solver=solver_cfg,
    )
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
