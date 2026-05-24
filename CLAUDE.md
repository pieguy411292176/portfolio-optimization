# Portfolio Optimizer

## Project Purpose
Mean-variance portfolio optimization (Markowitz). Implements:
1. `max_return.py` — maximize return subject to a risk budget (section 2.4.1)
2. `efficient_frontier.py` — sweep the efficient frontier via delta parameter (section 2.4.2)

## Solver
Use **cvxpy** (not MOSEK Fusion). Prefer conic form with Cholesky decomposition to mirror
the MOSEK formulation exactly. Use `cp.Parameter` for quantities swept in a loop (delta).

## Conventions
- Input arrays: numpy, named `m` (expected returns) and `S` (covariance matrix)
- Cholesky factor: `G = np.linalg.cholesky(S)` (lower triangular)
- No short-selling: `x >= 0` constraint, not bounds
- Results stored in a pandas DataFrame with columns: delta, obj, return, risk, + ticker names
- Plots use matplotlib; save to `plots/` subfolder

## Subagents
When implementing features in parallel (e.g., optimization logic vs. plotting), spawn subagents:
- Give each agent a self-contained prompt with all relevant context
- Merge results in the parent after both complete
- Useful here: one agent for the optimizer, one for the visualization code

## Testing
Run `python max_return.py` to check part 1. Expected: portfolio_return ≈ 0.429 (stock 5 dominant).
Run `python efficient_frontier.py` to check part 2. Expected: 20-point frontier plot saved to `plots/`.
