from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit as _njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is an optional accelerator
    _HAS_NUMBA = False

    def _njit(*args, **kwargs):  # type: ignore[no-redef]
        """No-op stand-in for numba.njit when numba is unavailable."""

        def _decorator(func):
            return func

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _decorator


@dataclass(frozen=True)
class PortfolioInstance:
    """One feasible portfolio instance with linear exposure constraints."""

    instance_id: int
    seed: int
    n: int
    m: int
    k: int
    sigma: float
    gamma: float
    Sigma: np.ndarray
    mu: np.ndarray
    A: np.ndarray
    l: np.ndarray
    u: np.ndarray
    x0: np.ndarray
    B: np.ndarray
    b: np.ndarray
    sigma_lipschitz: float
    B_norm_sq: float


@dataclass
class OracleResult:
    """CVXPY oracle result."""

    x: np.ndarray
    z: np.ndarray
    objective: float
    status: str
    solver: str
    solve_time_sec: float


@dataclass
class SolverResult:
    """Integrated solver result."""

    name: str
    x: np.ndarray
    objective: float
    objective_history: Tuple[float, ...]
    objective_trace: Tuple[float, ...]
    outer_trace_indices: Tuple[int, ...]
    frozen_inner_x_objective_trace: Tuple[float, ...]
    frozen_inner_y_objective_trace: Tuple[float, ...]
    outer_iterations: int
    inner_iterations: int
    converged: bool
    simplex_residual: float
    exposure_violation: float
    nonnegativity_violation: float
    budget_residual: float
    stationarity_residual: float
    solve_time_sec: float
    rho_final: float
    eta_final: float
    breakpoint_events: int
    merge_events: int
    regime_changes: int


def make_B_b(A: np.ndarray, l: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode l <= A x <= u as Bx - b <= 0."""

    A = np.asarray(A, dtype=float)
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    B = np.vstack([A, -A])
    b = np.concatenate([u, -l])
    return B, b


def build_random_feasible_instance(
    instance_id: int,
    seed: int,
    n: int,
    k: int,
    m: Optional[int] = None,
) -> PortfolioInstance:
    """Build a random feasible instance with exposure bounds around a feasible x0."""

    rng = np.random.default_rng(seed)
    if m is None:
        m = min(max(2, n // 2), 8)

    A_rand = rng.normal(size=(n, n))
    Sigma = A_rand.T @ A_rand + 0.1 * np.eye(n)
    mu = rng.normal(size=n)
    sigma = float(rng.uniform(0.4, 1.6))
    gamma = float(rng.uniform(0.5, 2.5))

    x0 = rng.dirichlet(np.ones(n))
    A = rng.normal(size=(m, n))
    exposure_center = A @ x0
    slack = rng.uniform(0.05, 0.25, size=m) * (1.0 + np.linalg.norm(A, axis=1))
    l = exposure_center - slack
    u = exposure_center + slack
    B, b = make_B_b(A, l, u)

    sigma_lipschitz = max(2.0 * sigma ** 2 * float(np.max(np.linalg.eigvalsh(Sigma))), 1e-8)
    B_norm_sq = float(np.linalg.norm(B, 2) ** 2)

    return PortfolioInstance(
        instance_id=instance_id,
        seed=seed,
        n=n,
        m=m,
        k=k,
        sigma=sigma,
        gamma=gamma,
        Sigma=Sigma,
        mu=mu,
        A=A,
        l=l,
        u=u,
        x0=x0,
        B=B,
        b=b,
        sigma_lipschitz=sigma_lipschitz,
        B_norm_sq=B_norm_sq,
    )


def phi_piecewise(values: np.ndarray) -> np.ndarray:
    """Evaluate phi(a) in the conjugate formula."""

    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    mask_mid = (values > 0.0) & (values < 1.0)
    mask_hi = values >= 1.0
    out[mask_mid] = 0.5 * values[mask_mid] ** 2
    out[mask_hi] = values[mask_hi] - 0.5
    return out


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex."""

    v = np.asarray(v, dtype=float)
    if v.size == 1:
        return np.array([1.0], dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, v.size + 1)
    mask = u - cssv / ind > 0.0
    rho = ind[mask][-1]
    theta = cssv[mask][-1] / rho
    return np.maximum(v - theta, 0.0)


def prox_phi_scalar(u: float, lam: float) -> float:
    """Scalar prox for phi."""

    if lam <= 0.0:
        return float(u)
    if u <= 0.0:
        return float(u)
    if u <= 1.0 + lam:
        return float(u / (1.0 + lam))
    return float(u - lam)


def _pava_blocks_fast(
    u: np.ndarray,
    rho: float,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast PAVA for prox_{rho g*}(u) with a light block representation.

    Returns:
        solution: prox_{rho g*}(u)
        size_arr: block sizes
        a_arr: block a_B values
        value_arr: block prox values q_B
        regime_arr: block regimes encoded as 0=inactive, 1=quadratic, 2=linear
    """

    u = np.asarray(u, dtype=float)
    order = np.argsort(-u)
    sorted_u = u[order]

    starts: List[int] = []
    ends: List[int] = []
    sizes: List[float] = []
    sum_us: List[float] = []
    penalizeds: List[float] = []
    values: List[float] = []

    for index, value in enumerate(sorted_u):
        penalized = 1.0 if index < k else 0.0
        starts.append(index)
        ends.append(index + 1)
        sizes.append(1.0)
        sum_us.append(float(value))
        penalizeds.append(penalized)
        values.append(prox_phi_scalar(float(value), rho * penalized))

        while len(values) >= 2 and values[-2] < values[-1]:
            right_start = starts.pop()
            right_end = ends.pop()
            right_size = sizes.pop()
            right_sum = sum_us.pop()
            right_pen = penalizeds.pop()
            right_value = values.pop()

            left_start = starts.pop()
            left_end = ends.pop()
            left_size = sizes.pop()
            left_sum = sum_us.pop()
            left_pen = penalizeds.pop()
            left_value = values.pop()

            del right_value, left_value
            merged_size = left_size + right_size
            merged_sum = left_sum + right_sum
            merged_pen = left_pen + right_pen
            avg_u = merged_sum / merged_size
            merged_value = prox_phi_scalar(avg_u, rho * merged_pen / merged_size)

            starts.append(left_start)
            ends.append(right_end)
            sizes.append(merged_size)
            sum_us.append(merged_sum)
            penalizeds.append(merged_pen)
            values.append(merged_value)

    sorted_solution = np.empty_like(sorted_u)
    size_arr = np.empty(len(values), dtype=float)
    a_arr = np.empty(len(values), dtype=float)
    value_arr = np.empty(len(values), dtype=float)
    regime_arr = np.empty(len(values), dtype=np.int8)

    for idx, (start, end, size, sum_u, penalized, value) in enumerate(
        zip(starts, ends, sizes, sum_us, penalizeds, values)
    ):
        sorted_solution[start:end] = value
        avg_u = sum_u / size
        a_block = rho * penalized / size
        size_arr[idx] = size
        a_arr[idx] = a_block
        value_arr[idx] = value
        if avg_u <= 0.0:
            regime_arr[idx] = 0
        elif avg_u <= 1.0 + a_block:
            regime_arr[idx] = 1
        else:
            regime_arr[idx] = 2

    solution = np.empty_like(sorted_solution)
    solution[order] = sorted_solution
    return solution, size_arr, a_arr, value_arr, regime_arr


def compute_g_and_z(x: np.ndarray, k: int, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    """Evaluate g(x) and produce an optimal z via the exact threshold formula."""

    x = np.asarray(x, dtype=float)
    if np.any(x < -tol) or np.any(x > 1.0 + tol) or np.sum(x) > k + tol:
        return math.inf, np.full_like(x, np.nan)

    x = np.clip(x, 0.0, 1.0)
    positive = np.flatnonzero(x > tol)
    z = np.zeros_like(x)
    if positive.size <= k:
        z[positive] = 1.0
        return 0.5 * float(np.dot(x, x)), z

    order = np.argsort(-x)
    sorted_x = x[order]
    prefix = np.cumsum(sorted_x)
    total = float(prefix[-1])

    r_star = 0
    theta_star = total / k
    found = False
    for r in range(k):
        prefix_r = float(prefix[r - 1]) if r > 0 else 0.0
        suffix_sum = total - prefix_r
        theta = suffix_sum / (k - r)
        left = float(sorted_x[r - 1]) if r > 0 else math.inf
        right = float(sorted_x[r])
        if left > theta + tol and theta >= right - tol:
            r_star = r
            theta_star = theta
            found = True
            break

    if not found:
        r_star = k - 1
        theta_star = float(sorted_x[k - 1])

    z = np.minimum(1.0, x / max(theta_star, tol))
    top_sq = float(np.dot(sorted_x[:r_star], sorted_x[:r_star]))
    suffix_sum = total - (float(prefix[r_star - 1]) if r_star > 0 else 0.0)
    value = 0.5 * (top_sq + suffix_sum * suffix_sum / (k - r_star))
    return value, z


def original_objective(x: np.ndarray, instance: PortfolioInstance) -> float:
    """Evaluate the reduced objective with exposure constraints."""

    x = np.asarray(x, dtype=float)
    if np.any(x < -1e-10) or abs(np.sum(x) - 1.0) > 1e-7:
        return math.inf
    if exposure_violation(x, instance) > 1e-7:
        return math.inf
    g_value, _ = compute_g_and_z(x, instance.k)
    if not np.isfinite(g_value):
        return math.inf
    return (
        instance.sigma ** 2 * float(x @ instance.Sigma @ x)
        - float(instance.mu @ x)
        + g_value / instance.gamma
    )


def exposure_violation(x: np.ndarray, instance: PortfolioInstance) -> float:
    """Maximum violation of l <= Ax <= u."""

    exposure = instance.A @ np.asarray(x, dtype=float)
    low_violation = np.maximum(instance.l - exposure, 0.0)
    high_violation = np.maximum(exposure - instance.u, 0.0)
    return float(max(np.max(low_violation, initial=0.0), np.max(high_violation, initial=0.0)))


def prox_gstar_pava_detailed(
    u: np.ndarray,
    rho: float,
    k: int,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Compute prox_{rho g*}(u) and detailed PAVA blocks."""

    u = np.asarray(u, dtype=float)
    order = np.argsort(-u)
    sorted_u = u[order]

    blocks: List[Dict[str, float]] = []
    for index, value in enumerate(sorted_u):
        penalized = 1 if index < k else 0
        fitted = prox_phi_scalar(float(value), rho * penalized)
        blocks.append(
            {
                "start": int(index),
                "end": int(index + 1),
                "size": 1.0,
                "sum_u": float(value),
                "penalized": float(penalized),
                "value": float(fitted),
            }
        )
        while len(blocks) >= 2 and blocks[-2]["value"] < blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            size = left["size"] + right["size"]
            sum_u = left["sum_u"] + right["sum_u"]
            penalized_count = left["penalized"] + right["penalized"]
            avg_u = sum_u / size
            fitted = prox_phi_scalar(avg_u, rho * penalized_count / size)
            blocks.append(
                {
                    "start": left["start"],
                    "end": right["end"],
                    "size": size,
                    "sum_u": sum_u,
                    "penalized": penalized_count,
                    "value": float(fitted),
                }
            )

    sorted_solution = np.empty_like(sorted_u)
    enriched_blocks: List[Dict[str, float]] = []
    for block in blocks:
        start = int(block["start"])
        end = int(block["end"])
        size = float(block["size"])
        penalized = float(block["penalized"])
        avg_u = float(block["sum_u"] / size)
        a_block = float(rho * penalized / size)
        if avg_u <= 0.0:
            regime = "inactive"
        elif avg_u <= 1.0 + a_block:
            regime = "quadratic"
        else:
            regime = "linear"
        sorted_solution[start:end] = block["value"]
        enriched_blocks.append(
            {
                "start": start,
                "end": end,
                "size": size,
                "penalized": penalized,
                "avg_u": avg_u,
                "a_block": a_block,
                "value": float(block["value"]),
                "regime": regime,
            }
        )

    solution = np.empty_like(sorted_solution)
    solution[order] = sorted_solution
    return solution, enriched_blocks


def prox_g_via_moreau_detailed(
    v: np.ndarray,
    tau: float,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Compute prox_{tau g}(v) and detailed PAVA diagnostics."""

    if tau <= 0.0:
        x = np.asarray(v, dtype=float).copy()
        return x, np.zeros_like(x), []
    scaled = np.asarray(v, dtype=float) / tau
    prox_star, blocks = prox_gstar_pava_detailed(scaled, 1.0 / tau, k)
    prox_value = np.asarray(v, dtype=float) - tau * prox_star
    return prox_value, prox_star, blocks


def prox_g_via_moreau(v: np.ndarray, tau: float, k: int) -> np.ndarray:
    """Fast prox_{tau g}(v) via Moreau + PAVA."""

    if tau <= 0.0:
        return np.asarray(v, dtype=float).copy()
    scaled = np.asarray(v, dtype=float) / tau
    prox_star, _, _, _, _ = _pava_blocks_fast(scaled, 1.0 / tau, k)
    return np.asarray(v, dtype=float) - tau * prox_star


def _prox_g_via_moreau_blocks(
    v: np.ndarray,
    tau: float,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute prox_{tau g}(v) plus lightweight block data for breakpoint walk."""

    if tau <= 0.0:
        x = np.asarray(v, dtype=float).copy()
        empty = np.empty(0, dtype=float)
        empty_i = np.empty(0, dtype=np.int8)
        return x, empty, empty, empty, empty_i
    scaled = np.asarray(v, dtype=float) / tau
    prox_star, size_arr, a_arr, value_arr, regime_arr = _pava_blocks_fast(scaled, 1.0 / tau, k)
    prox_value = np.asarray(v, dtype=float) - tau * prox_star
    return prox_value, size_arr, a_arr, value_arr, regime_arr


def phi_simplex_sum(v: np.ndarray, tau: float, k: int, nu: float) -> float:
    """Evaluate Phi(nu) = 1^T prox_{tau g}(v - nu 1)."""

    if tau <= 0.0:
        return float(np.sum(prox_g_via_moreau(np.asarray(v, dtype=float) - nu, tau, k)))
    v = np.asarray(v, dtype=float)
    sorted_v = np.sort(v)[::-1]
    return float(_phi_sum_kernel(sorted_v, tau, k, nu, 1.0 / tau))


def find_simplex_multiplier_bracket(
    v: np.ndarray,
    tau: float,
    k: int,
    target_sum: float = 1.0,
    nu_hint: Optional[float] = None,
) -> Tuple[float, float]:
    """Find a bracket [lo, hi] with Phi(lo) >= target_sum >= Phi(hi)."""

    v = np.asarray(v, dtype=float)
    if tau > 0.0:
        sorted_v = np.sort(v)[::-1]
        rho = 1.0 / tau

        def _phi(nu: float) -> float:
            return float(_phi_sum_kernel(sorted_v, tau, k, nu, rho))

    else:

        def _phi(nu: float) -> float:
            return phi_simplex_sum(v, tau, k, nu)

    if nu_hint is not None and np.isfinite(nu_hint):
        lo = hi = float(nu_hint)
        phi_mid = _phi(nu_hint)
        radius = max(1e-3, 0.05 * (1.0 + float(np.max(np.abs(v))) + abs(float(nu_hint))))
        if phi_mid >= target_sum:
            phi_lo = phi_mid
            phi_hi = phi_mid
            expand = 0
            while phi_hi > target_sum and expand < 100:
                hi += radius
                phi_hi = _phi(hi)
                radius *= 2.0
                expand += 1
            if phi_lo >= target_sum and phi_hi <= target_sum:
                return lo, hi
        else:
            phi_lo = phi_mid
            phi_hi = phi_mid
            expand = 0
            while phi_lo < target_sum and expand < 100:
                lo -= radius
                phi_lo = _phi(lo)
                radius *= 2.0
                expand += 1
            if phi_lo >= target_sum and phi_hi <= target_sum:
                return lo, hi

    radius = max(1.0, float(np.max(np.abs(v))) + abs(tau) + 1.0)
    lo = float(np.min(v) - radius)
    hi = float(np.max(v) + radius)
    phi_lo = _phi(lo)
    phi_hi = _phi(hi)

    expand = 0
    while phi_lo < target_sum and expand < 100:
        radius *= 2.0
        lo -= radius
        phi_lo = _phi(lo)
        expand += 1

    expand = 0
    while phi_hi > target_sum and expand < 100:
        radius *= 2.0
        hi += radius
        phi_hi = _phi(hi)
        expand += 1

    if phi_lo < target_sum or phi_hi > target_sum:
        raise RuntimeError("Failed to bracket the simplex multiplier.")
    return lo, hi


# ---------------------------------------------------------------------------
# Directional finite breakpoint walk.
#
# Mathematical direction (see main.md, "No-splitting of the breakpoint
# path"): with lambda = tau, rho = 1/tau, and s = -nu/tau, the sorted
# shifted values obey mu_tilde_i(s) = mu_tilde_i(0) + s. The no-splitting
# theorem only holds while sweeping s upward, equivalently nu downward.
# The walk below always starts at a right bracket nu_R with Phi(nu_R) <= 1
# and only ever decreases nu. Only two event families are ever generated:
#   (1) scalar regime changes of a block (bar_mu_B(s) crosses 0 or 1+a_B),
#   (2) adjacent block merges (t_{B_L}(s) = t_{B_R}(s)).
# No split events are produced, and there is no bisection fallback: if the
# event enumeration cannot locate the root, a RuntimeError with full
# diagnostics is raised.
# ---------------------------------------------------------------------------


@_njit(cache=True)
def _scalar_prox_h_kernel(u: float, a: float) -> float:
    """Closed-form prox_{a h}(u) (Proposition 5)."""

    if a <= 0.0:
        return u
    if u <= 0.0:
        return u
    if u <= 1.0 + a:
        return u / (1.0 + a)
    return u - a


@_njit(cache=True)
def _stack_pava_kernel(
    sorted_scaled: np.ndarray,
    k: int,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack-based maximal-block PAVA partition at a fixed nu (one sort, O(n)).

    Operates purely on parallel NumPy arrays (no Python list/dict) so it can
    be numba-jitted. Returns, for each maximal block: start index, element
    count, sum of the (shifted) values, and count of penalized elements.
    """

    n = sorted_scaled.shape[0]
    b_start = np.empty(n, dtype=np.int64)
    b_count = np.empty(n, dtype=np.int64)
    b_sum = np.empty(n, dtype=np.float64)
    b_pen = np.empty(n, dtype=np.float64)
    b_value = np.empty(n, dtype=np.float64)

    top = -1
    for i in range(n):
        top += 1
        b_start[top] = i
        b_count[top] = 1
        b_sum[top] = sorted_scaled[i]
        b_pen[top] = 1.0 if i < k else 0.0
        a_block = rho * b_pen[top] / b_count[top]
        b_value[top] = _scalar_prox_h_kernel(b_sum[top], a_block)

        while top >= 1 and b_value[top - 1] < b_value[top]:
            b_count[top - 1] += b_count[top]
            b_sum[top - 1] += b_sum[top]
            b_pen[top - 1] += b_pen[top]
            avg = b_sum[top - 1] / b_count[top - 1]
            a_block = rho * b_pen[top - 1] / b_count[top - 1]
            b_value[top - 1] = _scalar_prox_h_kernel(avg, a_block)
            top -= 1

    m = top + 1
    return b_start[:m].copy(), b_count[:m].copy(), b_sum[:m].copy(), b_pen[:m].copy()


@_njit(cache=True)
def _phi_sum_kernel(sorted_v: np.ndarray, tau: float, k: int, nu: float, rho: float) -> float:
    """Fast O(n) evaluation of Phi(nu) = 1^T prox_{tau g}(sorted_v - nu*1) on presorted input.

    Used by the bracket search, which evaluates Phi at many candidate nu for
    the same input; this avoids the pure-Python PAVA path for every trial.
    """

    n = sorted_v.shape[0]
    b_count = np.empty(n, dtype=np.int64)
    b_sum = np.empty(n, dtype=np.float64)
    b_pen = np.empty(n, dtype=np.float64)
    b_value = np.empty(n, dtype=np.float64)

    top = -1
    for i in range(n):
        top += 1
        u = (sorted_v[i] - nu) / tau
        b_count[top] = 1
        b_sum[top] = u
        b_pen[top] = 1.0 if i < k else 0.0
        a_block = rho * b_pen[top]
        b_value[top] = _scalar_prox_h_kernel(u, a_block)

        while top >= 1 and b_value[top - 1] < b_value[top]:
            b_count[top - 1] += b_count[top]
            b_sum[top - 1] += b_sum[top]
            b_pen[top - 1] += b_pen[top]
            avg = b_sum[top - 1] / b_count[top - 1]
            a_block = rho * b_pen[top - 1] / b_count[top - 1]
            b_value[top - 1] = _scalar_prox_h_kernel(avg, a_block)
            top -= 1

    t_sum = 0.0
    total_v = 0.0
    for i in range(n):
        total_v += sorted_v[i]
    for j in range(top + 1):
        t_sum += b_count[j] * b_value[j]
    return total_v - n * nu - tau * t_sum


@_njit(cache=True)
def _reconstruct_x_kernel(
    sorted_v: np.ndarray,
    block_start: np.ndarray,
    block_count: np.ndarray,
    block_tvalue: np.ndarray,
    nu: float,
    tau: float,
) -> np.ndarray:
    """Expand block-level fitted values t_B back into the full sorted x vector."""

    n = sorted_v.shape[0]
    x_sorted = np.empty(n, dtype=np.float64)
    for bi in range(block_start.shape[0]):
        start = block_start[bi]
        count = block_count[bi]
        t_b = block_tvalue[bi]
        for j in range(start, start + count):
            x_sorted[j] = sorted_v[j] - nu - tau * t_b
    return x_sorted


def _block_state(
    count: float,
    sum_mu0: float,
    m_b: float,
    nu: float,
    tau: float,
    rho: float,
) -> Tuple[float, float, float, int, float]:
    """O(1) block state at a given nu: (a_B, bar_mu_B, t_B, regime, dt_B/ds).

    The comparisons are biased by a tiny epsilon toward the *later* regime
    (the one that applies for slightly larger s / smaller nu). Since the walk
    always moves in the increasing-s direction, a bar that lands within
    floating error of a threshold should be treated as having just crossed
    it; classifying it as "not yet crossed" would be an arbitrary artifact of
    rounding, and produces an inconsistent (and hard to detect) slope for any
    exact or near-exact tie in the input.
    """

    avg_mu0 = sum_mu0 / count
    a_block = rho * m_b / count
    bar = avg_mu0 - nu / tau
    eps = 1e-9 * (1.0 + abs(avg_mu0) + a_block)
    if bar <= -eps:
        return a_block, bar, bar, 0, 1.0
    if bar <= 1.0 + a_block - eps:
        return a_block, bar, bar / (1.0 + a_block), 1, 1.0 / (1.0 + a_block)
    return a_block, bar, bar - a_block, 2, 1.0


def _block_value_from_regime(
    count: float,
    sum_mu0: float,
    m_b: float,
    nu: float,
    tau: float,
    rho: float,
    regime: int,
) -> Tuple[float, float]:
    """O(1) block value (t_B, dt_B/ds) at nu, given the block's *authoritative* regime.

    Unlike `_block_state`, this never re-derives the regime from a bar<=threshold
    comparison. That comparison is numerically ambiguous exactly at the instant a
    block's regime has just flipped (bar is only zero up to floating error), so
    regime transitions must always be applied explicitly by the event loop and the
    resulting label trusted here, not re-inferred.
    """

    avg_mu0 = sum_mu0 / count
    a_block = rho * m_b / count
    bar = avg_mu0 - nu / tau
    if regime == 0:
        return bar, 1.0
    if regime == 1:
        return bar / (1.0 + a_block), 1.0 / (1.0 + a_block)
    return bar - a_block, 1.0


def validate_breakpoint_walk_diagnostics(
    v: np.ndarray,
    tau: float,
    k: int,
    nu_right: float,
    nu_root: float,
    event_log: Sequence[Tuple[str, float]],
    tol: float = 1e-9,
) -> None:
    """Sanity-check the breakpoint walk against the expected monotone direction.

    Verifies: Phi(nu_R) <= 1, the walk direction is decreasing nu (each
    logged event nu is <= the starting bracket and the sequence of event nu
    values is non-increasing), and only regime/merge events were produced
    (no splits).
    """

    v = np.asarray(v, dtype=float)
    phi_right = phi_simplex_sum(v, tau, k, nu_right)
    if phi_right > 1.0 + tol:
        raise AssertionError(f"Right bracket has Phi(nu_R)={phi_right:.6e} > 1.")
    if nu_root > nu_right + tol:
        raise AssertionError(
            f"Root lies to the right of the starting bracket: nu_root={nu_root}, nu_right={nu_right}."
        )

    previous_nu = nu_right
    for event_type, event_nu in event_log:
        if event_type not in {"regime", "merge"}:
            raise AssertionError(f"Unexpected event type (split events are forbidden): {event_type}")
        if event_nu > previous_nu + tol:
            raise AssertionError(
                f"Walk direction violated: event nu={event_nu:.6e} > previous nu={previous_nu:.6e}."
            )
        if event_nu > nu_right + tol:
            raise AssertionError(f"Event nu={event_nu:.6e} lies to the right of the bracket nu_R={nu_right:.6e}.")
        previous_nu = event_nu
    if nu_root > previous_nu + tol:
        raise AssertionError("Root does not lie beyond (at or left of) the last processed event.")


def prox_moreau_simplex_breakpoint_walk(
    v: np.ndarray,
    tau: float,
    k: int,
    tol: float = 1e-12,
    max_steps: int = 10_000,
    nu_init: Optional[float] = None,
    debug: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """This is where we get the proximal operator: Compute prox_{tau g + I_{1^T x=1}}(v) by the directional finite breakpoint walk.

    Starts at a right bracket nu_R with Phi(nu_R) <= 1 and walks toward
    smaller nu (equivalently increasing s = -nu/tau), maintaining a
    stack-based maximal-block partition. On every interval with fixed block
    partition and scalar regimes, Phi is affine in nu; the next event
    (a scalar regime change or an adjacent block merge -- never a split) is
    found in O(1) per block via a lazily-invalidated event heap, giving
    overall O(n log n) complexity (one sort, O(n) initial PAVA scan, O(n)
    total events each processed in O(log n)).

    There is no fallback. If the event enumeration cannot locate
    the root, a RuntimeError with full diagnostics is raised.
    """

    if tau < 0.0:
        raise ValueError("tau must be nonnegative.")
    v = np.asarray(v, dtype=float)
    n = v.size

    if tau == 0.0:
        x = project_simplex(v)
        return x, {
            "nu": math.nan,
            "steps": 0,
            "breakpoint_events": 0,
            "merge_events": 0,
            "regime_changes": 0,
            "phi_residual": 0.0,
        }

    rho = 1.0 / tau
    nu_lo, nu_hi = find_simplex_multiplier_bracket(v, tau, k, nu_hint=nu_init)
    nu_R = float(nu_hi)

    phi_right = phi_simplex_sum(v, tau, k, nu_R)
    if phi_right > 1.0 + 1e-7:
        raise RuntimeError(
            f"Invalid right bracket for the breakpoint walk: Phi(nu_R={nu_R:.6e})={phi_right:.6e} > 1. "
            "The no-splitting walk requires Phi(nu_R) <= 1 to start."
        )

    order = np.argsort(-v)
    sorted_v = v[order]
    scaled_at_R = (sorted_v - nu_R) / tau
    b_start0, b_count0, b_sum_shift0, b_pen0 = _stack_pava_kernel(scaled_at_R, int(k), rho)
    num_initial = int(b_start0.shape[0])

    b_sum_mu0_0 = b_sum_shift0 + b_count0.astype(np.float64) * (nu_R / tau)

    cap = max(num_initial, 1)
    b_start = np.zeros(cap, dtype=np.int64)
    b_count = np.zeros(cap, dtype=np.int64)
    b_sum_mu0 = np.zeros(cap, dtype=np.float64)
    b_m = np.zeros(cap, dtype=np.float64)
    b_active = np.zeros(cap, dtype=bool)
    b_version = np.zeros(cap, dtype=np.int64)
    b_slope_gen = np.zeros(cap, dtype=np.int64)
    b_prev = np.full(cap, -1, dtype=np.int64)
    b_next = np.full(cap, -1, dtype=np.int64)
    b_regime = np.zeros(cap, dtype=np.int64)
    b_a = np.zeros(cap, dtype=np.float64)
    b_t = np.zeros(cap, dtype=np.float64)

    for i in range(num_initial):
        b_start[i] = b_start0[i]
        b_count[i] = b_count0[i]
        b_sum_mu0[i] = b_sum_mu0_0[i]
        b_m[i] = b_pen0[i]
        b_active[i] = True
        b_prev[i] = i - 1
        b_next[i] = i + 1 if i + 1 < num_initial else -1

    # Heap entries are (-nu_event, kind, id1, id2, ver1, ver2, sgen1, sgen2). kind=0 is a
    # regime event (id2/ver2/sgen2 unused, set to -1); kind=1 is a merge event between
    # adjacent blocks id1 (left) and id2 (right). Two independent per-block counters
    # guard against staleness: `b_version` changes only when a block's identity/
    # composition changes (i.e. it is absorbed by, or absorbs, a neighbor in a merge),
    # and is what regime events are validated against -- a block's own two sequential
    # regime events (bar=0 then bar=1+a_B) must NOT invalidate each other. `b_slope_gen`
    # changes on *both* merges and regime changes (anything that can change a block's
    # slope), and is what merge events are validated against, since a merge-event's
    # nu_event was computed assuming both endpoints' slopes stay fixed until then.
    heap: List[Tuple[float, int, int, int, int, int, int, int]] = []

    def _push_regime_events(bi: int, nu_cur: float) -> None:
        count = b_count[bi]
        m_b = b_m[bi]
        avg_mu0 = b_sum_mu0[bi] / count
        a_block = rho * m_b / count
        if a_block <= 0.0:
            return
        nu0 = tau * avg_mu0
        nu1 = tau * (avg_mu0 - (1.0 + a_block))
        if nu0 < nu_cur - tol:
            heapq.heappush(heap, (-nu0, 0, int(bi), -1, int(b_version[bi]), -1, -1, -1))
        if nu1 < nu_cur - tol:
            heapq.heappush(heap, (-nu1, 0, int(bi), -1, int(b_version[bi]), -1, -1, -1))

    def _push_merge_event(li: int, ri: int, nu_cur: float) -> None:
        if li < 0 or ri < 0:
            return
        t_l, slope_l_s = _block_value_from_regime(
            b_count[li], b_sum_mu0[li], b_m[li], nu_cur, tau, rho, int(b_regime[li])
        )
        t_r, slope_r_s = _block_value_from_regime(
            b_count[ri], b_sum_mu0[ri], b_m[ri], nu_cur, tau, rho, int(b_regime[ri])
        )
        gap = t_l - t_r
        if gap < -tol or slope_r_s <= slope_l_s + tol:
            return
        # Under the maximal-block convention, adjacent blocks that are already tied (or
        # numerically indistinguishable) while the right block's slope exceeds the
        # left's must be merged immediately (nu_event == nu_cur), not scheduled for a
        # future crossing -- this matters when input ties place unequal a_B on
        # otherwise-identical values (e.g. an exact tie straddling the top-k cutoff).
        nu_event = nu_cur if gap <= tol else nu_cur - gap * tau / (slope_r_s - slope_l_s)
        if nu_event <= nu_cur + tol:
            heapq.heappush(
                heap,
                (
                    -nu_event,
                    1,
                    int(li),
                    int(ri),
                    int(b_version[li]),
                    int(b_version[ri]),
                    int(b_slope_gen[li]),
                    int(b_slope_gen[ri]),
                ),
            )

    # Phi(nu) = 1^T x(nu) = sum(v) - n*nu - tau * sum_B(count_B * t_B(nu)); see (A-17)-(A-18).
    # d(Phi)/dnu = -n + sum_B count_B * slope_s_B(nu)/... simplifies to 0 for non-quadratic
    # blocks (slope_s=1) and to -count_B*a_B/(1+a_B) for quadratic blocks, since the "-n" term
    # exactly cancels the count_B*1 contribution of every non-quadratic block.
    sum_v = float(np.sum(v))

    # Vectorized initial classification: with n up to several thousand, looping the
    # scalar `_block_state`/`_push_*_event` closures once per block (Python-level
    # call + heap push) dominates runtime. Only a_B>0 blocks (at most k of them, out
    # of num_initial) ever need a regime-event push, and only pairs where the right
    # block's slope exceeds the left's need a merge-event push, so both are computed
    # in bulk with NumPy and only the surviving candidates go through heapq.
    count_arr = b_count[:num_initial].astype(np.float64)
    avg_mu0_arr = b_sum_mu0[:num_initial] / count_arr
    a_arr = rho * b_m[:num_initial] / count_arr
    bar_arr = avg_mu0_arr - nu_R / tau
    eps_arr = 1e-9 * (1.0 + np.abs(avg_mu0_arr) + a_arr)
    regime_arr = np.where(
        bar_arr <= -eps_arr, 0, np.where(bar_arr <= 1.0 + a_arr - eps_arr, 1, 2)
    ).astype(np.int64)
    t_arr = np.where(
        regime_arr == 0, bar_arr, np.where(regime_arr == 1, bar_arr / (1.0 + a_arr), bar_arr - a_arr)
    )
    slope_s_arr = np.where(regime_arr == 1, 1.0 / (1.0 + a_arr), 1.0)

    b_a[:num_initial] = a_arr
    b_t[:num_initial] = t_arr
    b_regime[:num_initial] = regime_arr

    t_sum = float(np.sum(count_arr * t_arr))
    quad_mask = regime_arr == 1
    slope_cur = float(np.sum(-count_arr[quad_mask] * a_arr[quad_mask] / (1.0 + a_arr[quad_mask])))

    penalized_idx = np.flatnonzero(a_arr > 0.0)
    nu0_arr = tau * avg_mu0_arr
    nu1_arr = tau * (avg_mu0_arr - (1.0 + a_arr))
    for bi in penalized_idx:
        bi = int(bi)
        if nu0_arr[bi] < nu_R - tol:
            heapq.heappush(heap, (-float(nu0_arr[bi]), 0, bi, -1, int(b_version[bi]), -1, -1, -1))
        if nu1_arr[bi] < nu_R - tol:
            heapq.heappush(heap, (-float(nu1_arr[bi]), 0, bi, -1, int(b_version[bi]), -1, -1, -1))

    if num_initial > 1:
        gap_arr = t_arr[:-1] - t_arr[1:]
        merge_mask = (gap_arr >= -tol) & (slope_s_arr[1:] > slope_s_arr[:-1] + tol)
        merge_idx = np.flatnonzero(merge_mask)
        for li in merge_idx:
            li = int(li)
            _push_merge_event(li, li + 1, nu_R)

    Phi_cur = sum_v - n * nu_R - tau * t_sum

    nu_cur = nu_R
    breakpoint_events = 0
    merge_events = 0
    regime_changes = 0
    event_log: List[Tuple[str, float]] = []
    nu_root: Optional[float] = None

    for _step in range(1, max_steps + 1):
        while heap:
            neg_nu, kind, id1, id2, ver1, ver2, sgen1, sgen2 = heap[0]
            if kind == 0:
                ok = bool(b_active[id1]) and int(b_version[id1]) == ver1
            else:
                ok = (
                    id2 >= 0
                    and bool(b_active[id1])
                    and bool(b_active[id2])
                    and int(b_next[id1]) == id2
                    and int(b_slope_gen[id1]) == sgen1
                    and int(b_slope_gen[id2]) == sgen2
                )
            if ok:
                break
            heapq.heappop(heap)

        nu_event = -heap[0][0] if heap else -math.inf

        # An "immediate" event (nu_event >= nu_cur - tol) is a tie under the
        # maximal-block convention that must be resolved before the root can be
        # trusted: e.g. two adjacent blocks with unequal a_B that happen to already
        # share the same fitted value, where the right block's slope would exceed
        # the left's on any further move. Apply it first, without checking for a
        # root, so the block partition stays valid.
        if heap and nu_event >= nu_cur - tol:
            pass
        else:
            residual = Phi_cur - 1.0
            if abs(residual) <= tol:
                nu_root = nu_cur
                break

            if abs(slope_cur) > tol:
                candidate_root = nu_cur + (1.0 - Phi_cur) / slope_cur
                if nu_event - tol <= candidate_root <= nu_cur + tol:
                    nu_root = candidate_root
                    break

            if not heap:
                raise RuntimeError(
                    "Breakpoint walk exhausted all regime/merge events without locating the simplex "
                    f"multiplier: nu_cur={nu_cur:.6e}, Phi={Phi_cur:.6e}, residual={residual:.6e}, "
                    f"bracket=[{nu_lo:.6e}, {nu_hi:.6e}], breakpoint_events={breakpoint_events}, "
                    f"merge_events={merge_events}, regime_changes={regime_changes}. This indicates a "
                    "missing event in the no-splitting enumeration; no bisection fallback is used."
                )

        neg_nu, kind, id1, id2, _ver1, _ver2, _sgen1, _sgen2 = heapq.heappop(heap)
        nu_event = -neg_nu
        Phi_cur = Phi_cur + slope_cur * (nu_event - nu_cur)
        nu_cur = nu_event
        breakpoint_events += 1

        if kind == 0:
            regime_changes += 1
            event_log.append(("regime", nu_event))
            bi = id1
            old_regime = int(b_regime[bi])
            old_a = float(b_a[bi])
            old_count = int(b_count[bi])
            if old_regime == 1:
                slope_cur -= -old_count * old_a / (1.0 + old_a)
            # Regime events always advance exactly one step (0->1 at bar=0, 1->2 at
            # bar=1+a_B). At nu_cur == the threshold itself, re-deriving the regime
            # from the bar<=0 / bar<=1+a comparison is numerically ambiguous (bar is
            # only zero up to floating error), so the new regime is set explicitly
            # instead of recomputed from the boundary test.
            new_regime = old_regime + 1
            count = b_count[bi]
            avg_mu0 = b_sum_mu0[bi] / count
            a_block = rho * b_m[bi] / count
            bar = avg_mu0 - nu_cur / tau
            t_b = bar / (1.0 + a_block) if new_regime == 1 else bar - a_block
            b_a[bi] = a_block
            b_t[bi] = t_b
            b_regime[bi] = new_regime
            # A regime change alters this block's slope, which invalidates the nu_event
            # of any merge candidate previously computed against its old slope. Bumping
            # the slope generation here (distinct from the identity/composition version
            # used by regime events, which must survive a sibling regime event on the
            # same block) makes those stale merge-event heap entries fail the lazy
            # validity check on pop, instead of being silently reused.
            b_slope_gen[bi] += 1
            if new_regime == 1:
                slope_cur += -count * a_block / (1.0 + a_block)
            _push_merge_event(int(b_prev[bi]), bi, nu_cur)
            _push_merge_event(bi, int(b_next[bi]), nu_cur)
        else:
            merge_events += 1
            event_log.append(("merge", nu_event))
            li, ri = id1, id2
            for bidx in (li, ri):
                old_regime = int(b_regime[bidx])
                if old_regime == 1:
                    old_a = float(b_a[bidx])
                    old_count = int(b_count[bidx])
                    slope_cur -= -old_count * old_a / (1.0 + old_a)

            b_count[li] = b_count[li] + b_count[ri]
            b_sum_mu0[li] = b_sum_mu0[li] + b_sum_mu0[ri]
            b_m[li] = b_m[li] + b_m[ri]
            new_next = int(b_next[ri])
            b_next[li] = new_next
            if new_next >= 0:
                b_prev[new_next] = li
            b_active[ri] = False
            b_version[ri] += 1
            b_version[li] += 1
            b_slope_gen[ri] += 1
            b_slope_gen[li] += 1

            a_block, _bar, t_b, regime, _slope_s = _block_state(
                b_count[li], b_sum_mu0[li], b_m[li], nu_cur, tau, rho
            )
            b_a[li] = a_block
            b_t[li] = t_b
            b_regime[li] = regime
            if regime == 1:
                slope_cur += -b_count[li] * a_block / (1.0 + a_block)

            _push_regime_events(li, nu_cur)
            _push_merge_event(int(b_prev[li]), li, nu_cur)
            _push_merge_event(li, int(b_next[li]), nu_cur)
    else:
        nu_root = None

    if nu_root is None:
        raise RuntimeError(
            f"Breakpoint walk exceeded max_steps={max_steps} without locating the simplex "
            f"multiplier: nu_cur={nu_cur:.6e}, Phi={Phi_cur:.6e}, bracket=[{nu_lo:.6e}, {nu_hi:.6e}]."
        )

    # Walk the linked list once to collect the surviving block slots (cheap: just
    # integer bookkeeping), then evaluate all of their t_B values in one vectorized
    # NumPy pass rather than calling `_block_state` once per block in a Python loop.
    final_indices: List[int] = []
    bi = 0
    while bi != -1:
        final_indices.append(bi)
        bi = int(b_next[bi])

    final_idx_arr = np.asarray(final_indices, dtype=np.int64)
    count_f = b_count[final_idx_arr].astype(np.float64)
    avg_mu0_f = b_sum_mu0[final_idx_arr] / count_f
    a_f = rho * b_m[final_idx_arr] / count_f
    bar_f = avg_mu0_f - nu_root / tau
    eps_f = 1e-9 * (1.0 + np.abs(avg_mu0_f) + a_f)
    regime_f = np.where(bar_f <= -eps_f, 0, np.where(bar_f <= 1.0 + a_f - eps_f, 1, 2))
    final_tval_arr = np.where(
        regime_f == 0, bar_f, np.where(regime_f == 1, bar_f / (1.0 + a_f), bar_f - a_f)
    )
    final_start_arr = b_start[final_idx_arr]
    final_count_arr = b_count[final_idx_arr]
    x_sorted = _reconstruct_x_kernel(sorted_v, final_start_arr, final_count_arr, final_tval_arr, nu_root, tau)
    x = np.empty(n, dtype=float)
    x[order] = x_sorted

    phi_residual = float(np.sum(x_sorted) - 1.0)

    if debug:
        validate_breakpoint_walk_diagnostics(v, tau, k, nu_R, nu_root, event_log, tol=max(tol, 1e-9))

    return x, {
        "nu": float(nu_root),
        "steps": float(breakpoint_events),
        "breakpoint_events": breakpoint_events,
        "merge_events": merge_events,
        "regime_changes": regime_changes,
        "phi_residual": phi_residual,
        "num_blocks_final": len(final_indices),
        "num_blocks_initial": num_initial,
    }


def run_breakpoint_walk_self_check(
    n_trials: int = 8,
    seed: int = 0,
    tol: float = 1e-9,
) -> Dict[str, float]:
    """Debug/testing utility: compare the event walk against reference bisection.

    This function is for validation only; it is never called from the
    production solve path (which uses `prox_moreau_simplex_breakpoint_walk`
    exclusively and never falls back to bisection).
    """

    rng = np.random.default_rng(seed)
    max_error = 0.0
    for _ in range(n_trials):
        n = int(rng.integers(4, 12))
        v = rng.normal(size=n) * rng.uniform(0.5, 3.0)
        tau = float(rng.uniform(0.1, 2.0))
        k = int(rng.integers(1, n))
        x_walk, diag_walk = prox_moreau_simplex_breakpoint_walk(v, tau, k, tol=tol, debug=True)
        x_ref, _diag_ref = prox_moreau_simplex_bisection(v, tau, k, tol=tol)
        err = float(np.linalg.norm(x_walk - x_ref, ord=np.inf))
        max_error = max(max_error, err)
        if err > 1e-9:
            raise AssertionError(
                f"Breakpoint walk disagrees with bisection: err={err:.3e}, diag={diag_walk}"
            )
    return {"max_error": max_error, "trials": float(n_trials)}


def prox_moreau_simplex_bisection(
    v: np.ndarray,
    tau: float,
    k: int,
    tol: float = 1e-12,
    max_iter: int = 80,
    nu_init: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Reference simplex-constrained prox using scalar bisection on nu.

    DEBUG/TEST ONLY. This helper exists solely for cross-checking the
    breakpoint walk on small random instances (see
    `run_breakpoint_walk_self_check`). It must never be called from
    production solve paths (`solver.py`) and there is no automatic fallback
    to it anywhere in the breakpoint walk.
    """

    if tau < 0.0:
        raise ValueError("tau must be nonnegative.")
    if tau == 0.0:
        x = project_simplex(np.asarray(v, dtype=float))
        return x, {"nu": math.nan}

    v = np.asarray(v, dtype=float)
    nu_lo, nu_hi = find_simplex_multiplier_bracket(v, tau, k, nu_hint=nu_init)

    def residual(nu: float) -> float:
        return phi_simplex_sum(v, tau, k, nu) - 1.0

    try:
        from scipy.optimize import brentq as _brentq
    except Exception:  # pragma: no cover - optional acceleration
        _brentq = None

    if _brentq is not None:
        nu_star = _brentq(residual, nu_lo, nu_hi, xtol=tol, rtol=tol, maxiter=200)
    else:
        for _ in range(max_iter):
            nu_mid = 0.5 * (nu_lo + nu_hi)
            if residual(nu_mid) > 0.0:
                nu_lo = nu_mid
            else:
                nu_hi = nu_mid
        nu_star = 0.5 * (nu_lo + nu_hi)
    x_star = prox_g_via_moreau(v - nu_star, tau, k)
    return x_star, {"nu": float(nu_star)}


def check_block_monotonicity(x: np.ndarray, tol: float = 1e-9) -> bool:
    """Diagnostic: the prox output, sorted descending, must be non-increasing."""

    sorted_x = np.sort(np.asarray(x, dtype=float))[::-1]
    diffs = np.diff(sorted_x)
    return bool(np.all(diffs <= tol))


def check_moreau_identity(v: np.ndarray, tau: float, k: int, tol: float = 1e-8) -> float:
    """Diagnostic: max violation of prox_{tau g}(v) + tau*prox_{g*/tau}(v/tau) == v."""

    v = np.asarray(v, dtype=float)
    prox_g = prox_g_via_moreau(v, tau, k)
    scaled = v / tau
    prox_star, _, _, _, _ = _pava_blocks_fast(scaled, 1.0 / tau, k)
    residual = prox_g + tau * prox_star - v
    return float(np.max(np.abs(residual)))


def check_simplex_residual(x: np.ndarray, tol: float = 1e-8) -> float:
    """Diagnostic: |1^T x - 1|."""

    return float(abs(np.sum(np.asarray(x, dtype=float)) - 1.0))
