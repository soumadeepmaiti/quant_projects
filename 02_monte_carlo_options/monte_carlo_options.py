"""
Monte Carlo Option Pricing & Variance Reduction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# 1. BLACK-SCHOLES ANALYTIC (BENCHMARK)
# ─────────────────────────────────────────────

def bs_price(S: float, K: float, r: float, T: float, sigma: float,
             option: str = "call") -> float:
    """Black-Scholes closed-form price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if option == "call":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * np.sqrt(T) * norm.pdf(d1)
    return {"delta": delta, "gamma": gamma, "vega": vega}


# ─────────────────────────────────────────────
# 2. GBM PATH GENERATION
# ─────────────────────────────────────────────

def gbm_terminal(S0: float, r: float, sigma: float, T: float,
                  N: int, antithetic: bool = False, seed: int = 42) -> np.ndarray:
    """
    Exact GBM terminal prices:
        S_T = S0 · exp( (r - σ²/2)T + σ√T · Z )
    Antithetic: use Z and -Z for variance reduction.
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        Z = rng.standard_normal(N // 2)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(N)
    return S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)


def gbm_paths(S0: float, r: float, sigma: float, T: float,
               N: int, M: int, seed: int = 42) -> np.ndarray:
    """
    Full GBM path matrix: shape (N, M+1).
    Uses exact discretisation: S_{t+dt} = S_t · exp((r-σ²/2)dt + σ√dt·Z).
    """
    rng = np.random.default_rng(seed)
    dt  = T / M
    Z   = rng.standard_normal((N, M))
    log_incr = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_S    = np.log(S0) + np.cumsum(log_incr, axis=1)
    paths    = np.empty((N, M + 1))
    paths[:, 0] = S0
    paths[:, 1:] = np.exp(log_S)
    return paths


# ─────────────────────────────────────────────
# 3. EUROPEAN OPTIONS
# ─────────────────────────────────────────────

@dataclass
class MCResult:
    price: float
    std_err: float
    ci_95: tuple
    n_paths: int

    def __str__(self):
        lo, hi = self.ci_95
        return (f"Price={self.price:.4f}  SE={self.std_err:.5f}  "
                f"95%CI=[{lo:.4f}, {hi:.4f}]  N={self.n_paths:,}")


def mc_european(S0, K, r, T, sigma, N=200_000,
                antithetic=True, control_variate=False,
                option="call", seed=42) -> MCResult:
    """
    Monte Carlo European option with optional variance reduction.
    Control variate: use E[S_T] = S0·e^{rT} (known analytically).
    """
    ST = gbm_terminal(S0, r, sigma, T, N, antithetic=antithetic, seed=seed)
    disc = np.exp(-r * T)

    if option == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)

    if control_variate:
        # CV: h(S_T) = S_T,  E[h] = S0·e^{rT}
        expected_ST = S0 * np.exp(r * T)
        beta = np.cov(payoffs, ST)[0, 1] / np.var(ST)
        payoffs = payoffs - beta * (ST - expected_ST)

    payoffs_disc = disc * payoffs
    price   = float(np.mean(payoffs_disc))
    std_err = float(np.std(payoffs_disc, ddof=1) / np.sqrt(len(payoffs_disc)))
    return MCResult(price, std_err, (price - 1.96 * std_err, price + 1.96 * std_err), N)


# ─────────────────────────────────────────────
# 4. ASIAN OPTIONS (ARITHMETIC AVERAGE)
# ─────────────────────────────────────────────

def mc_asian(S0, K, r, T, sigma, N=100_000, M=252, seed=42) -> MCResult:
    """
    Arithmetic average Asian call:
        Payoff = max( (1/M)·Σ S_k  −  K,  0 )
    """
    paths   = gbm_paths(S0, r, sigma, T, N, M, seed)
    avg     = paths[:, 1:].mean(axis=1)
    disc    = np.exp(-r * T)
    payoffs = disc * np.maximum(avg - K, 0.0)
    price   = float(np.mean(payoffs))
    se      = float(np.std(payoffs, ddof=1) / np.sqrt(N))
    return MCResult(price, se, (price - 1.96 * se, price + 1.96 * se), N)


def geometric_asian_analytic(S0, K, r, T, sigma, M):
    """
    Closed-form geometric Asian call (used as control variate benchmark).
    Under GBM, ln G_T ~ N(μ_G, σ_G²).
    """
    sigma_g = sigma * np.sqrt((M + 1) * (2 * M + 1) / (6 * M * M))
    mu_g    = (r - 0.5 * sigma ** 2) * (M + 1) / (2 * M)
    d1 = (np.log(S0 / K) + (mu_g + 0.5 * sigma_g ** 2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)
    disc = np.exp(-r * T)
    return disc * (S0 * np.exp(mu_g * T) * norm.cdf(d1) - K * norm.cdf(d2))


# ─────────────────────────────────────────────
# 5. BARRIER OPTIONS (DOWN-AND-OUT CALL)
# ─────────────────────────────────────────────

def mc_barrier_dao(S0, K, B, r, T, sigma, N=100_000, M=252,
                    brownian_bridge=True, seed=42) -> MCResult:
    """
    Down-and-out call with optional Brownian bridge bias correction.

    Discretisation bias: P(hit) is underestimated when using discrete steps.
    Brownian bridge correction:
        P(cross barrier between t_k and t_{k+1} | S_k, S_{k+1} > B) =
            exp( -2 ln(S_k/B) ln(S_{k+1}/B) / (σ² Δt) )
    """
    paths = gbm_paths(S0, r, sigma, T, N, M, seed)
    disc  = np.exp(-r * T)

    if brownian_bridge:
        dt      = T / M
        S_left  = paths[:, :-1]   # S_{k}
        S_right = paths[:, 1:]    # S_{k+1}
        # Both endpoints above barrier
        both_above = (S_left > B) & (S_right > B)
        # Bridge probability of crossing
        with np.errstate(divide="ignore", invalid="ignore"):
            p_cross = np.exp(
                -2 * np.log(np.maximum(S_left / B, 1e-12)) *
                   np.log(np.maximum(S_right / B, 1e-12)) /
                (sigma ** 2 * dt)
            )
        p_cross = np.where(both_above, p_cross, 0.0)
        # Path is knocked out with probability prod(1 - p_cross)
        log_surv = np.log(np.maximum(1 - p_cross, 1e-15)).sum(axis=1)
        discrete_hit = (paths[:, 1:].min(axis=1) <= B)
        alive = (~discrete_hit) * np.exp(log_surv)
    else:
        alive = (paths[:, 1:].min(axis=1) > B).astype(float)

    payoffs = disc * np.maximum(paths[:, -1] - K, 0.0) * alive
    price   = float(np.mean(payoffs))
    se      = float(np.std(payoffs, ddof=1) / np.sqrt(N))
    return MCResult(price, se, (price - 1.96 * se, price + 1.96 * se), N)


# ─────────────────────────────────────────────
# 6. EXCHANGE OPTION (MARGRABE)
# ─────────────────────────────────────────────

def margrabe_price(S1, S2, sigma1, sigma2, rho, T) -> float:
    """
    Margrabe (1978) closed-form for exchange option:
        Payoff = max(S1_T - S2_T, 0)
        σ_spread = √(σ1² + σ2² - 2ρσ1σ2)
        C = S1·N(d1) - S2·N(d2)
    """
    sigma_sp = np.sqrt(sigma1 ** 2 + sigma2 ** 2 - 2 * rho * sigma1 * sigma2)
    if sigma_sp < 1e-10:
        return max(S1 - S2, 0.0)
    d1 = (np.log(S1 / S2) + 0.5 * sigma_sp ** 2 * T) / (sigma_sp * np.sqrt(T))
    d2 = d1 - sigma_sp * np.sqrt(T)
    return S1 * norm.cdf(d1) - S2 * norm.cdf(d2)


def mc_exchange(S1, S2, sigma1, sigma2, rho, r, T,
                 N=200_000, M=50, seed=42) -> MCResult:
    """
    MC exchange option using Cholesky-correlated GBM.
    Correlated normals:
        Z1 = L11·ε1
        Z2 = L21·ε1 + L22·ε2
    where L is the Cholesky factor of the correlation matrix.
    """
    rng = np.random.default_rng(seed)
    dt  = T / M
    # Cholesky factor
    L11 = sigma1
    L21 = rho * sigma2
    L22 = sigma2 * np.sqrt(max(1 - rho ** 2, 0))

    S1_t = np.full(N, S1)
    S2_t = np.full(N, S2)

    for _ in range(M):
        e1 = rng.standard_normal(N)
        e2 = rng.standard_normal(N)
        Z1 = L11 * e1
        Z2 = L21 * e1 + L22 * e2
        S1_t *= np.exp((r - 0.5 * sigma1 ** 2) * dt + np.sqrt(dt) * Z1)
        S2_t *= np.exp((r - 0.5 * sigma2 ** 2) * dt + np.sqrt(dt) * Z2)

    disc    = np.exp(-r * T)
    payoffs = disc * np.maximum(S1_t - S2_t, 0.0)
    price   = float(np.mean(payoffs))
    se      = float(np.std(payoffs, ddof=1) / np.sqrt(N))
    return MCResult(price, se, (price - 1.96 * se, price + 1.96 * se), N)


# ─────────────────────────────────────────────
# 7. CONVERGENCE ANALYSIS  O(1/√N)
# ─────────────────────────────────────────────

def convergence_study(S0, K, r, T, sigma, max_N=500_000):
    """Empirically confirm O(1/√N) convergence rate."""
    ns = np.array([1_000, 5_000, 10_000, 50_000, 100_000, 500_000])
    ns = ns[ns <= max_N]
    prices, errors = [], []
    true_price = bs_price(S0, K, r, T, sigma, "call")

    for n in ns:
        res = mc_european(S0, K, r, T, sigma, N=n, antithetic=False, seed=0)
        prices.append(res.price)
        errors.append(abs(res.price - true_price))

    slope, intercept = np.polyfit(np.log(ns), np.log(errors), 1)
    return ns, np.array(prices), np.array(errors), slope, true_price


# ─────────────────────────────────────────────
# 8. VARIANCE REDUCTION REPORT
# ─────────────────────────────────────────────

def variance_reduction_report(S0, K, r, T, sigma, N=100_000):
    true = bs_price(S0, K, r, T, sigma, "call")
    print(f"  Black-Scholes benchmark: {true:.4f}\n")

    configs = [
        ("Naive MC",               dict(antithetic=False, control_variate=False)),
        ("Antithetic Variates",    dict(antithetic=True,  control_variate=False)),
        ("Control Variate",        dict(antithetic=False, control_variate=True)),
        ("Antithetic + CV",        dict(antithetic=True,  control_variate=True)),
    ]

    base_var = None
    rows = []
    for name, cfg in configs:
        res = mc_european(S0, K, r, T, sigma, N=N, seed=42, **cfg)
        var = res.std_err ** 2 * N
        if base_var is None:
            base_var = var
        reduction = 1 - var / base_var if base_var > 0 else 0
        rows.append({"Method": name, "Price": f"{res.price:.4f}",
                     "Std Err": f"{res.std_err:.6f}",
                     "Var Reduction": f"{reduction*100:.1f}%",
                     "Error vs BS": f"{abs(res.price-true):.5f}"})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────
# 9. VISUALISATION
# ─────────────────────────────────────────────

def plot_convergence(ns, errors, slope, true_price):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("MC Convergence Analysis", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.loglog(ns, errors, "o-", label="MC Error", linewidth=2)
    ref_line = errors[0] * np.sqrt(ns[0] / ns)
    ax.loglog(ns, ref_line, "--", label="O(1/√N)", color="red")
    ax.set_xlabel("N paths")
    ax.set_ylabel("|MC price − BS price|")
    ax.set_title(f"Convergence Rate ≈ N^{slope:.3f}  (theory: −0.5)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax2 = axes[1]
    ax2.semilogx(ns, errors / true_price * 100, "s-", color="navy")
    ax2.axhline(0.1, color="red", linestyle="--", label="0.1% threshold")
    ax2.set_xlabel("N paths")
    ax2.set_ylabel("Relative Error (%)")
    ax2.set_title("Relative Error vs N")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mc_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_paths_and_payoff(S0, K, r, T, sigma, N=5000, M=252):
    paths = gbm_paths(S0, r, sigma, T, N, M)
    t     = np.linspace(0, T, M + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GBM Simulation", fontsize=13, fontweight="bold")

    ax = axes[0]
    for i in range(min(100, N)):
        ax.plot(t, paths[i], alpha=0.15, linewidth=0.5, color="steelblue")
    ax.axhline(K, color="red", linestyle="--", linewidth=1.5, label=f"Strike K={K}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("S_t")
    ax.set_title("100 GBM Sample Paths")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ST = paths[:, -1]
    ax2.hist(ST, bins=80, density=True, alpha=0.6, label="MC Terminal Prices")
    s_grid = np.linspace(ST.min(), ST.max(), 300)
    mu     = (r - 0.5 * sigma ** 2) * T
    sig_ln = sigma * np.sqrt(T)
    pdf    = np.exp(-0.5 * ((np.log(s_grid / S0) - mu) / sig_ln) ** 2) / \
             (s_grid * sig_ln * np.sqrt(2 * np.pi))
    ax2.plot(s_grid, pdf, "r-", linewidth=2, label="Lognormal PDF")
    ax2.axvline(K, color="navy", linestyle="--", label=f"K={K}")
    ax2.set_xlabel("S_T")
    ax2.set_title("Terminal Price Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gbm_paths.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    S0, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.20

    print("=" * 60)
    print("PROJECT 2 — MONTE CARLO OPTION PRICING")
    print("=" * 60)

    # European options
    print("\n--- European Call ---")
    print(f"  BS price: {bs_price(S0, K, r, T, sigma):.4f}")
    print("\nVariance Reduction Report:")
    variance_reduction_report(S0, K, r, T, sigma)

    # Asian option
    print("\n--- Asian (Arithmetic) Call ---")
    res_asian = mc_asian(S0, K, r, T, sigma)
    geo       = geometric_asian_analytic(S0, K, r, T, sigma, M=252)
    print(f"  MC Arithmetic:  {res_asian}")
    print(f"  Geometric (analytic benchmark): {geo:.4f}")

    # Barrier option
    print("\n--- Down-and-Out Call  (B=90) ---")
    res_bar_naive  = mc_barrier_dao(S0, K, 90, r, T, sigma, brownian_bridge=False)
    res_bar_bridge = mc_barrier_dao(S0, K, 90, r, T, sigma, brownian_bridge=True)
    print(f"  Naive (discrete):          {res_bar_naive}")
    print(f"  Brownian Bridge corrected: {res_bar_bridge}")

    # Exchange option
    print("\n--- Exchange Option (Margrabe) ---")
    S1, S2, s1, s2, rho = 100.0, 105.0, 0.20, 0.25, 0.50
    analytic  = margrabe_price(S1, S2, s1, s2, rho, T)
    res_exch  = mc_exchange(S1, S2, s1, s2, rho, r, T)
    print(f"  Margrabe closed-form: {analytic:.4f}")
    print(f"  MC:                   {res_exch}")
    print(f"  Error: {abs(res_exch.price - analytic):.5f}")

    # Convergence
    print("\n--- Convergence O(1/√N) ---")
    ns, prices, errors, slope, true_p = convergence_study(S0, K, r, T, sigma)
    print(f"  Empirical convergence exponent: {slope:.4f}  (theory: -0.50)")
    plot_convergence(ns, errors, slope, true_p)
    plot_paths_and_payoff(S0, K, r, T, sigma)
