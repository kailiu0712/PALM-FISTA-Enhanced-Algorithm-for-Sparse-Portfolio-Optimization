The paper’s oscillation is typically shown for a single fixed composite problem, plotting the FISTA objective itself over iterations:
F(beta^t) + 2 lambda g(beta^t)
or a primal-dual gap for that same fixed problem.

Current code is different in three important ways:
It is not plotting the inner subproblem objective.
Ins olver.py:156, the trace stores reduced_objective_unchecked(x_next, instance).
That is the original portfolio objective, not the PALM inner objective being optimized at that moment.

The inner solver is actually minimizing an augmented objective that depends on p, rho, eta, and x_center; see the gradient in solver.py:97. So the plotted curve is already a different diagnostic.

The trace concatenates several different inner solves.
integrated_palm_solve runs multiple outer PALM iterations, and each one changes the subproblem through p, rho, eta, and x_center. So "inner trace" is not one long FISTA run on one fixed objective. It is several warm-started FISTA runs on changing objectives. That suppresses the classic repeated oscillation pattern.

You are plotting the prox iterates x_next, not the extrapolated points.
The overshoot in FISTA comes from the momentum/extrapolation step (solver.py:168 and solver.py:169). But the trace records only the post-prox point x_next. That sequence can look much smoother than the raw momentum trajectory.


What the new plots suggest:
The prox-iterate objective x^(t) drops very quickly at the start, then keeps decreasing in a fairly regular way.
On the log-gap plot, the curve is close to a straight descending line for a long stretch. That usually means steady geometric-like improvement on this instance.
The extrapolated-point objective y^(t) is slightly worse than the prox-iterate objective early on, but it does not blow up or bounce strongly. It tracks x^(t) closely after the transient.
The best observed value is essentially reached by the prox iterates, not by the extrapolated points. That is typical: y^(t) is the aggressive momentum point, x^(t+1) is the corrected prox point.
The final sharp drop in the blue gap curve is just because the best observed objective is attained at the last prox iterate, so its gap to the best observed value is numerically near zero.

Why it is smooth here:
This frozen subproblem is strongly stabilized by several terms, not just the original objective:
sigma^2 x^T Sigma x, the augmented-Lagrangian penalty, and the proximal term (1/(2 eta)) ||x - x_center||^2.
The step size uses a conservative Lipschitz constant (solver.py:132), which tends to damp oscillation.
The prox step is exact and quite strong here, so momentum does not get much chance to overshoot badly.
This particular instance may simply be well-conditioned for the inner problem.
On this portfolio subproblem and this instance, the frozen inner objective is smooth enough that plain FISTA looks well behaved. 

Difference of 'restart' is instance-dependent:
on benchmark-seed n=3000 case, the value-based restart condition was never met (restart=True had restart_events = 0). It is obereved though, on smaller instances, restart makes improvements. By contrast, for the scaling-study seed family, the n=3000 got:
restart=True: iterations = 119, restart_events = 3
restart=False: iterations = 200, hit the max-inner limit on that frozen subproblem
So restart does matter on some instances; it just did not matter on the specific benchmark instance you inspected.

Why restart seems to help smaller n more than larger n
Using the old no-restart summary and the current restart summary:
no-restart first mean crossover was around n = 700
restart first mean crossover is now around n = 350
That left-shift does suggest restart helped the integrated solver more at moderate sizes than at the largest sizes.
The likely reasons are:
Restart mainly saves inner iterations, not asymptotic oracle cost.
It only helps when plain FISTA overshoots enough to waste steps. If the non-restart inner loop is already smooth, restart does nothing.

At moderate n, wasted acceleration steps are a bigger fraction of total integrated time. For smaller/moderate problems, the integrated method is often close to the CVXPY oracle in runtime, so cutting inner iterations has a visible multiplicative effect on speedup. At large n, the integrated solver is already winning strongly without restart.
Once n is large, CVXPY’s oracle time grows very quickly. Even if restart still helps, the speedup ratio is already dominated by oracle growth, so the extra relative gain from restart is less dramatic.


There is also runtime noise and randomness effects In the scaling summaries, CVXPY times at large n have substantial variance. So some of the apparent crossover shift can absolutely be due to random variation, solver variability, and compilation overhead noise.