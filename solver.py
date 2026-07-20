from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np

try:
    from .config import SolverConfig
    from .core import (
        OracleResult,
        PortfolioInstance,
        SolverResult,
        compute_g_and_z,
        exposure_violation,
        original_objective,
        prox_moreau_simplex_breakpoint_walk,
    )
except ImportError:  # pragma: no cover - script mode
    from config import SolverConfig
    from core import (
        OracleResult,
        PortfolioInstance,
        SolverResult,
        compute_g_and_z,
        exposure_violation,
        original_objective,
        prox_moreau_simplex_breakpoint_walk,
    )

SOLVER_FALLBACK = ("CLARABEL", "ECOS", "SCS")


def solve_with_fallback(problem: cp.Problem, verbose: bool = False) -> str:
    """Solve a CVXPY problem with a robust fallback order."""

    installed = set(cp.installed_solvers())
    last_error: Optional[Exception] = None
    for solver in SOLVER_FALLBACK:
        if solver not in installed:
            continue
        try:
            if solver == "SCS":
                problem.solve(solver=solver, verbose=verbose, eps=1e-8, max_iters=200_000)
            else:
                problem.solve(solver=solver, verbose=verbose)
        except Exception as exc:  # pragma: no cover - solver-dependent
            last_error = exc
            continue
        if problem.status in {"optimal", "optimal_inaccurate"}:
            return solver
    raise RuntimeError(
        f"All CVXPY solvers failed. Installed={sorted(installed)} last_error={last_error}"
    )


def oracle_solve(instance: PortfolioInstance, verbose: bool = False) -> OracleResult:
    """Solve the full lifted convex problem with CVXPY."""

    n = instance.n
    x = cp.Variable(n)
    z = cp.Variable(n)
    objective = (
        instance.sigma ** 2 * cp.quad_form(x, cp.psd_wrap(instance.Sigma))
        - instance.mu @ x
        + (0.5 / instance.gamma) * cp.sum([cp.quad_over_lin(x[i], z[i]) for i in range(n)])
    )
    constraints = [
        x >= 0.0,
        cp.sum(x) == 1.0,
        z >= x,
        z <= 1.0,
        cp.sum(z) <= instance.k,
        instance.A @ x >= instance.l,
        instance.A @ x <= instance.u,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    tic = time.perf_counter()
    solver = solve_with_fallback(problem, verbose=verbose)
    solve_time_sec = time.perf_counter() - tic
    if x.value is None or z.value is None:
        raise RuntimeError(f"Oracle solve failed with status={problem.status}")
    return OracleResult(
        x=np.asarray(x.value, dtype=float),
        z=np.asarray(z.value, dtype=float),
        objective=float(problem.value),
        status=problem.status,
        solver=solver,
        solve_time_sec=solve_time_sec,
    )


def reduced_objective_unchecked(x: np.ndarray, instance: PortfolioInstance) -> float:
    """Evaluate the portfolio loss without enforcing exposure feasibility."""

    x = np.asarray(x, dtype=float)
    g_value, _ = compute_g_and_z(x, instance.k)
    if not np.isfinite(g_value):
        return math.inf
    return (
        instance.sigma ** 2 * float(x @ instance.Sigma @ x)
        - float(instance.mu @ x)
        + g_value / instance.gamma
    )


def frozen_inner_objective(
    x: np.ndarray,
    x_center: np.ndarray,
    p: np.ndarray,
    rho: float,
    eta: float,
    instance: PortfolioInstance,
) -> float:
    """Evaluate the fixed outer-subproblem objective used by the inner FISTA loop."""

    base_value = reduced_objective_unchecked(x, instance)
    if not np.isfinite(base_value):
        return math.inf
    violation = p + rho * (instance.B @ np.asarray(x, dtype=float) - instance.b)
    penalty = 0.5 / rho * (
        float(np.dot(np.maximum(violation, 0.0), np.maximum(violation, 0.0))) - float(np.dot(p, p))
    )
    prox_term = 0.5 / eta * float(np.linalg.norm(np.asarray(x, dtype=float) - x_center) ** 2)
    return base_value + penalty + prox_term


def augmented_gradient(
    x: np.ndarray,
    x_center: np.ndarray,
    p: np.ndarray,
    rho: float,
    eta: float,
    instance: PortfolioInstance,
) -> np.ndarray:
    """Gradient of the smooth part of the proximal augmented Lagrangian subproblem."""

    c = instance.B @ x - instance.b
    penalty_term = np.maximum(p + rho * c, 0.0)
    return (
        2.0 * instance.sigma ** 2 * (instance.Sigma @ x)
        - instance.mu
        + instance.B.T @ penalty_term
        + (x - x_center) / eta
    )


def inner_subproblem_fista(
    instance: PortfolioInstance,
    x_center: np.ndarray,
    p: np.ndarray,
    rho: float,
    eta: float,
    restart: bool = True,
    objective_trace: Optional[list[float]] = None,
    frozen_inner_x_objective_trace: Optional[list[float]] = None,
    frozen_inner_y_objective_trace: Optional[list[float]] = None,
    warm_start: Optional[np.ndarray] = None,
    nu_warm_start: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Solve the proximal ALM subproblem (objective) by FISTA."""

    L = instance.sigma_lipschitz + rho * instance.B_norm_sq + 1.0 / eta
    tau = 1.0 / (L * instance.gamma) # lipnitz curvature for GD lr

    x = np.asarray(warm_start if warm_start is not None else x_center, dtype=float).copy()
    y = x.copy()
    t = 1.0
    breakpoint_events = 0
    merge_events = 0
    regime_changes = 0
    restart_events = 0
    nu_guess = nu_warm_start # warm start nu
    previous_value = frozen_inner_objective(x, x_center, p, rho, eta, instance)

    for iteration in range(1, max_iter + 1): # run one FISTA process
        grad = augmented_gradient(y, x_center, p, rho, eta, instance)
        v = y - grad / L
        x_next, prox_diag = prox_moreau_simplex_breakpoint_walk( # PAVA + breakpoint walk
            v,
            tau,
            instance.k,
            nu_init=nu_guess,
        )
        breakpoint_events += int(prox_diag.get("breakpoint_events", 0))
        merge_events += int(prox_diag.get("merge_events", 0))
        regime_changes += int(prox_diag.get("regime_changes", 0))
        nu_guess = prox_diag.get("nu", nu_guess)
        current_value = frozen_inner_objective(x_next, x_center, p, rho, eta, instance)
        if objective_trace is not None:
            objective_trace.append(reduced_objective_unchecked(x_next, instance))
        if frozen_inner_x_objective_trace is not None:
            frozen_inner_x_objective_trace.append(current_value)
        residual = float(np.linalg.norm(x_next - x))
        if residual <= tol: # criteria for inner loop convergence: x_next close enough to x
            return x_next, {
                "iterations": iteration,
                "stationarity": residual,
                "breakpoint_events": breakpoint_events,
                "merge_events": merge_events,
                "regime_changes": regime_changes,
                "restart_events": restart_events,
                "L": L,
                "last_nu": float(nu_guess) if nu_guess is not None else math.nan,
            }
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t)) # momentum
        if restart and current_value > previous_value + 1e-12 * max(1.0, abs(previous_value)):
            restart_events += 1  # restart momentum if objective increased
            t_next = 1.0
            y = x_next.copy()
        else:
            y = x_next + ((t - 1.0) / t_next) * (x_next - x)
        if frozen_inner_y_objective_trace is not None:
            frozen_inner_y_objective_trace.append(
                frozen_inner_objective(y, x_center, p, rho, eta, instance)
            )
        x = x_next
        t = t_next
        previous_value = current_value

    return x, {
        "iterations": max_iter,
        "stationarity": float(np.linalg.norm(augmented_gradient(x, x_center, p, rho, eta, instance))),
        "breakpoint_events": breakpoint_events,
        "merge_events": merge_events,
        "regime_changes": regime_changes,
        "restart_events": restart_events,
        "L": L,
        "last_nu": float(nu_guess) if nu_guess is not None else math.nan,
    }


def integrated_palm_solve(
    instance: PortfolioInstance,
    config: Optional[SolverConfig] = None,
    max_outer: int = 200,
    max_inner: int = 200,
    restart: bool = True,
    outer_tol: float = 1e-7,
    inner_tol: float = 1e-7,
    beta: float = 0.7,
    xi_rho: float = 2.0,
    rho0: float = 1.0,
    eta0: float = 1.0,
    alpha_growth: float = 1.2,
    delta_eta: float = 0.1,
) -> SolverResult:
    """Solve the full problem with linear exposure constraints by proximal ALM.

    If *config* is provided, its values override the keyword defaults.
    Individual keyword arguments still override the config when explicitly passed.
    """

    if config is not None:
        max_outer = max_outer if max_outer != 200 else config.max_outer
        max_inner = max_inner if max_inner != 200 else config.max_inner
        restart = restart if restart is not True else config.restart
        outer_tol = outer_tol if outer_tol != 1e-7 else config.outer_tol
        inner_tol = inner_tol if inner_tol != 1e-7 else config.inner_tol
        beta = beta if beta != 0.7 else config.beta
        xi_rho = xi_rho if xi_rho != 2.0 else config.xi_rho
        rho0 = rho0 if rho0 != 1.0 else config.rho0
        eta0 = eta0 if eta0 != 1.0 else config.eta0
        alpha_growth = alpha_growth if alpha_growth != 1.2 else config.alpha_growth
        delta_eta = delta_eta if delta_eta != 0.1 else config.delta_eta

    tic = time.perf_counter()
    x = instance.x0.copy()
    p = np.zeros(instance.B.shape[0], dtype=float)
    rho = float(rho0)
    eta = float(eta0)
    inner_iterations = 0
    breakpoint_events = 0
    merge_events = 0
    regime_changes = 0
    stationarity = math.inf
    converged = False
    nu_warm_start: Optional[float] = None
    objective_history = [reduced_objective_unchecked(x, instance)]
    objective_trace = [objective_history[0]]
    outer_trace_indices = [0]
    frozen_inner_x_objective_trace: Tuple[float, ...] = tuple()
    frozen_inner_y_objective_trace: Tuple[float, ...] = tuple()

    E_prev = np.minimum(instance.b - instance.B @ x, p)
    x_initial = x.copy()

    for outer_iteration in range(1, max_outer + 1):
        frozen_x_trace = None
        frozen_y_trace = None
        if outer_iteration == 1:
            initial_frozen_value = frozen_inner_objective(x, x, p, rho, eta, instance)
            frozen_x_trace = [initial_frozen_value]
            frozen_y_trace = [initial_frozen_value]
        x_next, inner_diag = inner_subproblem_fista(
            instance,
            x_center=x,
            p=p,
            rho=rho,
            eta=eta,
            restart=restart,
            objective_trace=objective_trace,
            frozen_inner_x_objective_trace=frozen_x_trace,
            frozen_inner_y_objective_trace=frozen_y_trace,
            warm_start=x,
            nu_warm_start=nu_warm_start,
            max_iter=max_inner,
            tol=inner_tol,
        )
        inner_iterations += int(inner_diag["iterations"])
        breakpoint_events += int(inner_diag.get("breakpoint_events", 0))
        merge_events += int(inner_diag.get("merge_events", 0))
        regime_changes += int(inner_diag.get("regime_changes", 0))
        if math.isfinite(inner_diag.get("last_nu", math.nan)):
            nu_warm_start = float(inner_diag["last_nu"])
        objective_history.append(reduced_objective_unchecked(x_next, instance))
        outer_trace_indices.append(len(objective_trace) - 1)
        if outer_iteration == 1 and frozen_x_trace is not None and frozen_y_trace is not None:
            frozen_inner_x_objective_trace = tuple(frozen_x_trace)
            frozen_inner_y_objective_trace = tuple(frozen_y_trace)

        c = instance.B @ x_next - instance.b
        p_next = np.maximum(p + rho * c, 0.0)  # PALM update
        E_next = np.minimum(instance.b - instance.B @ x_next, p / rho if rho > 0.0 else p)

        simplex_residual = float(abs(np.sum(x_next) - 1.0))
        exp_violation = exposure_violation(x_next, instance)
        stationarity = float(np.linalg.norm(x_next - x))

        if stationarity <= outer_tol and simplex_residual <= outer_tol and exp_violation <= outer_tol:
            x = x_next   # convergence criteria: x stationary, sum(x)=1 constraint small enough, exposure constraint small enough
            p = p_next
            converged = True
            break

        prev_norm = float(np.linalg.norm(E_prev, ord=np.inf))
        next_norm = float(np.linalg.norm(E_next, ord=np.inf))
        growth_floor = rho0 * ((outer_iteration + 1) ** alpha_growth) # Some PALM updates
        if prev_norm <= 1e-12:
            if next_norm > 1e-12:
                rho = max(xi_rho * rho, growth_floor)
        elif next_norm > beta * prev_norm:
            rho = max(xi_rho * rho, growth_floor)

        eta = max(delta_eta * float(np.linalg.norm(x_next - x_initial) ** 2), eta0 * ((outer_iteration + 1) ** alpha_growth))
        x = x_next
        p = p_next
        E_prev = E_next
    else:
        outer_iteration = max_outer

    solve_time_sec = time.perf_counter() - tic
    g_value, z = compute_g_and_z(x, instance.k)
    budget_residual = float(max(np.sum(z) - instance.k, 0.0)) if np.all(np.isfinite(z)) else math.inf
    return SolverResult(
        name="integrated_palm_breakpoint",
        x=x,
        objective=original_objective(x, instance),
        objective_history=tuple(objective_history),
        objective_trace=tuple(objective_trace),
        outer_trace_indices=tuple(outer_trace_indices),
        frozen_inner_x_objective_trace=frozen_inner_x_objective_trace,
        frozen_inner_y_objective_trace=frozen_inner_y_objective_trace,
        outer_iterations=outer_iteration,
        inner_iterations=inner_iterations,
        converged=converged,
        simplex_residual=float(abs(np.sum(x) - 1.0)),
        exposure_violation=exposure_violation(x, instance),
        nonnegativity_violation=float(max(0.0, -np.min(x))),
        budget_residual=budget_residual,
        stationarity_residual=stationarity,
        solve_time_sec=solve_time_sec,
        rho_final=rho,
        eta_final=eta,
        breakpoint_events=breakpoint_events,
        merge_events=merge_events,
        regime_changes=regime_changes,
    )
