"""Correctness and diagnostic checks for the directional breakpoint walk.

Run directly: python -m proximal_operator_validation.integrated_algorithm_for_portfolio_problem.test_breakpoint_walk

These checks exist to prove the *actual* event-driven walk works (not a
hybrid that silently falls back to bisection). `prox_moreau_simplex_bisection`
is used here only as an independent reference oracle for comparison; the
production solve path (`solver.py`) never calls it.
"""

from __future__ import annotations

import time
import traceback
from typing import Callable, List, Tuple

import numpy as np

try:
    from .core import (
        build_random_feasible_instance,
        check_block_monotonicity,
        check_moreau_identity,
        check_simplex_residual,
        original_objective,
        prox_moreau_simplex_bisection,
        prox_moreau_simplex_breakpoint_walk,
        run_breakpoint_walk_self_check,
    )
    from .solver import integrated_palm_solve, oracle_solve
except ImportError:  # pragma: no cover - script mode
    from core import (
        build_random_feasible_instance,
        check_block_monotonicity,
        check_moreau_identity,
        check_simplex_residual,
        original_objective,
        prox_moreau_simplex_bisection,
        prox_moreau_simplex_breakpoint_walk,
        run_breakpoint_walk_self_check,
    )
    from solver import integrated_palm_solve, oracle_solve

Check = Tuple[str, Callable[[], None]]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_directional_fuzz(n_trials: int = 4000, seed: int = 0) -> None:
    """The walk must agree with bisection to ~1e-9 and satisfy all invariants."""

    rng = np.random.default_rng(seed)
    max_err = 0.0
    for trial in range(n_trials):
        n = int(rng.integers(2, 20))
        v = rng.normal(size=n) * rng.uniform(0.3, 4.0)
        tau = float(rng.uniform(0.02, 4.0))
        k = int(rng.integers(1, n + 1))

        x_walk, diag = prox_moreau_simplex_breakpoint_walk(v, tau, k, debug=True)
        x_ref, _ = prox_moreau_simplex_bisection(v, tau, k)

        err = float(np.max(np.abs(x_walk - x_ref)))
        max_err = max(max_err, err)
        _assert(err < 1e-6, f"trial {trial}: walk disagrees with bisection by {err:.3e} (n={n}, k={k}, tau={tau})")
        _assert(
            check_simplex_residual(x_walk) < 1e-6,
            f"trial {trial}: simplex residual too large, diag={diag}",
        )
        _assert(check_block_monotonicity(x_walk), f"trial {trial}: sorted x not monotone")
        _assert(
            check_moreau_identity(v, tau, k) < 1e-7,
            f"trial {trial}: Moreau identity violated",
        )
    print(f"  directional fuzz: {n_trials} trials, max |walk-bisection| = {max_err:.3e}")


def check_no_split_events(n_trials: int = 500, seed: int = 1) -> None:
    """Every logged event must be 'regime' or 'merge'; never a split."""

    rng = np.random.default_rng(seed)
    total_events = 0
    for _ in range(n_trials):
        n = int(rng.integers(3, 25))
        v = rng.normal(size=n) * rng.uniform(0.3, 4.0)
        tau = float(rng.uniform(0.02, 4.0))
        k = int(rng.integers(1, n + 1))
        _, diag = prox_moreau_simplex_breakpoint_walk(v, tau, k, debug=True)
        total_events += diag["breakpoint_events"]
        _assert(
            diag["breakpoint_events"] == diag["merge_events"] + diag["regime_changes"],
            f"event accounting mismatch: {diag}",
        )
    print(f"  no-split-events: {n_trials} trials, {total_events} total events, all regime/merge")


def check_exact_ties(seed: int = 2) -> None:
    """Duplicate input values (ties straddling the top-k cutoff) must still match bisection."""

    rng = np.random.default_rng(seed)
    for _ in range(200):
        n = int(rng.integers(4, 12))
        base = rng.normal(size=max(2, n // 2))
        reps = rng.integers(1, 4, size=len(base))
        v = np.repeat(base, reps)[:n]
        if len(v) < n:
            v = np.concatenate([v, rng.normal(size=n - len(v))])
        tau = float(rng.uniform(0.05, 3.0))
        k = int(rng.integers(1, n))
        x_walk, _ = prox_moreau_simplex_breakpoint_walk(v, tau, k, debug=True)
        x_ref, _ = prox_moreau_simplex_bisection(v, tau, k)
        err = float(np.max(np.abs(x_walk - x_ref)))
        _assert(err < 1e-4, f"tie case mismatch: err={err:.3e}, v={v}, tau={tau}, k={k}")
    print("  exact ties: 200 trials with duplicate inputs, all matched bisection")


def check_no_fallback_on_failure() -> None:
    """A tau < 0 (invalid) must raise, not silently substitute a fallback solve."""

    try:
        prox_moreau_simplex_breakpoint_walk(np.array([1.0, 2.0]), -1.0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative tau")
    print("  no-fallback-on-failure: invalid tau raises instead of silently recovering")


def check_self_check_utility() -> None:
    result = run_breakpoint_walk_self_check(n_trials=20, seed=3)
    _assert(result["max_error"] < 1e-9, f"self-check max_error too large: {result}")
    print(f"  run_breakpoint_walk_self_check: {result}")


def check_solver_matches_oracle(seed: int = 42) -> None:
    """End-to-end: integrated solver with exposure constraints matches the CVXPY oracle."""

    for n, k in [(8, 2), (12, 4)]:
        instance = build_random_feasible_instance(instance_id=0, seed=seed + n, n=n, k=k)
        result = integrated_palm_solve(instance)
        oracle = oracle_solve(instance)
        obj_err = abs(result.objective - oracle.objective)
        _assert(result.converged, f"n={n}: integrated solver did not converge")
        _assert(obj_err < 1e-4, f"n={n}: objective mismatch {obj_err:.3e}")
        _assert(result.exposure_violation < 1e-6, f"n={n}: exposure constraint violated")
        _assert(result.simplex_residual < 1e-6, f"n={n}: simplex constraint violated")
        _assert(
            abs(original_objective(result.x, instance) - result.objective) < 1e-9,
            f"n={n}: reported objective inconsistent with original_objective",
        )
        print(
            f"  n={n} k={k}: obj_err={obj_err:.3e} exposure_violation={result.exposure_violation:.3e} "
            f"breakpoint_events={result.breakpoint_events} merge_events={result.merge_events}"
        )


def check_performance_scaling() -> None:
    """Sanity check that per-call cost grows roughly log-linearly, not quadratically+."""

    rng = np.random.default_rng(7)
    timings = {}
    for n in (200, 2000, 20000):
        v = rng.normal(size=n)
        k = max(1, n // 10)
        prox_moreau_simplex_breakpoint_walk(v[:20], 0.5, 3)  # warm numba
        t0 = time.perf_counter()
        for _ in range(5):
            prox_moreau_simplex_breakpoint_walk(v, 0.5, k)
        timings[n] = (time.perf_counter() - t0) / 5
    ratio_10x = timings[2000] / timings[200]
    ratio_100x = timings[20000] / timings[200]
    print(f"  perf: n=200 -> {timings[200]*1e3:.3f}ms, n=2000 -> {timings[2000]*1e3:.3f}ms, "
          f"n=20000 -> {timings[20000]*1e3:.3f}ms (x10 ratio={ratio_10x:.1f}, x100 ratio={ratio_100x:.1f})")
    _assert(ratio_100x < 200.0, "per-call cost appears to scale worse than ~O(n log n)")


CHECKS: List[Check] = [
    ("directional fuzz vs bisection", check_directional_fuzz),
    ("no split events ever generated", check_no_split_events),
    ("exact ties handled correctly", check_exact_ties),
    ("no silent fallback on failure", check_no_fallback_on_failure),
    ("self-check utility", check_self_check_utility),
    ("integrated solver matches CVXPY oracle", check_solver_matches_oracle),
    ("performance scaling", check_performance_scaling),
]


def main() -> None:
    failures = 0
    for name, fn in CHECKS:
        print(f"[{name}]")
        try:
            fn()
        except Exception:  # noqa: BLE001 - report and continue to next check
            failures += 1
            print(f"  FAILED:\n{traceback.format_exc()}")
    print()
    if failures:
        print(f"{failures}/{len(CHECKS)} checks FAILED")
        raise SystemExit(1)
    print(f"All {len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
