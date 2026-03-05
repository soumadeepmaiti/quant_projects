"""
Multi-Asset Monte Carlo — Variance Reduction & Correlated Pricing

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from dataclasses import dataclass


# ─────────────────────────────────────────────
# 1. CORRELATED GBM VIA CHOLESKY
# ─────────────────────────────────────────────

def cholesky_correlated_gbm(S0_vec: np.ndarray,
                              r: float,
                              sigma_vec: np.ndarray,
                              corr_matrix: np.ndarray,
                              T: float, M: int, N: int,
                              seed: int = 42) -> np.ndarray:
    """
    Simulate n_assets correlated GBM paths.

    Cholesky decomposition:
        Σ = σ · ρ · σ'  (covariance)
        L = chol(Σ)     (lower triangular)
        Z_corr = L · Z_indep,   Z_indep ~ N(0, I)

    Returns: array (N, n_assets, M+1)
    """
    rng     = np.random.default_rng(seed)
    n       = len(S0_vec)
    dt      = T / M

    # Build covariance and Cholesky factor
    Sigma = np.outer(sigma_vec, sigma_vec) * corr_matrix
    L     = np.linalg.cholesky(Sigma)

    # Independent standard normals: (N, n, M)
    Z_indep  = rng.standard_normal((N, n, M))
    # Correlated: Z_corr[i,k] = L @ Z_indep[i,:,k]
    Z_corr   = np.einsum("jk,ilk->ilj", L, Z_indep)  # shape (N, M, n)
    Z_corr   = Z_corr.transpose(0, 2, 1)              # shape (N, n, M)

    # GBM paths
    drift = (r - 0.5 * sigma_vec ** 2) * dt
    paths = np.empty((N, n, M + 1))
    paths[:, :, 0] = S0_vec

    for k in range(M):
        log_incr         = drift + np.sqrt(dt) * Z_corr[:, :, k]
        paths[:, :, k+1] = paths[:, :, k] * np.exp(log_incr)

    return paths


# ─────────────────────────────────────────────
# 2. EXCHANGE OPTION — MARGRABE BENCHMARK
# ─────────────────────────────────────────────

def margrabe_price(S1, S2, sigma1, sigma2, rho, T):
    """
    Margrabe (1978):
        C = S1·N(d1) - S2·N(d2)
        σ_spread = √(σ1² + σ2² - 2ρσ1σ2)
        d1 = (ln(S1/S2) + ½σ²_spread·T) / (σ_spread·√T)
    """
    sigma_sp = np.sqrt(max(sigma1**2 + sigma2**2 - 2*rho*sigma1*sigma2, 1e-12))
    d1 = (np.log(S1/S2) + 0.5*sigma_sp**2*T) / (sigma_sp*np.sqrt(T))
    d2 = d1 - sigma_sp*np.sqrt(T)
    return S1*norm.cdf(d1) - S2*norm.cdf(d2)


# ─────────────────────────────────────────────
# 3. VARIANCE REDUCTION METHODS
# ─────────────────────────────────────────────

class VarianceReduction:
    """
    Implement and compare variance reduction techniques:
      1. Naive Monte Carlo
      2. Antithetic Variates
      3. Control Variates
      4. Stratified Sampling
    """

    def __init__(self, S0, K, r, T, sigma, N=100_000):
        self.S0    = S0
        self.K     = K
        self.r     = r
        self.T     = T
        self.sigma = sigma
        self.N     = N
        self.disc  = np.exp(-r * T)
        self.drift = (r - 0.5*sigma**2)*T
        self.diff  = sigma*np.sqrt(T)
        # BS benchmark
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        self.bs_price = self.disc * (S0*norm.cdf(d1) - K*np.exp(r*T)*norm.cdf(d2))
        # Correct BS price
        self.bs_price = S0*norm.cdf(d1) - K*self.disc*norm.cdf(d2)

    def _terminal_prices(self, Z):
        return self.S0 * np.exp(self.drift + self.diff * Z)

    def _payoff(self, ST):
        return np.maximum(ST - self.K, 0.0)

    def naive(self, seed=0):
        rng = np.random.default_rng(seed)
        Z   = rng.standard_normal(self.N)
        ST  = self._terminal_prices(Z)
        p   = self.disc * self._payoff(ST)
        return p.mean(), p.std(ddof=1), p.var(ddof=1)

    def antithetic(self, seed=0):
        """
        Antithetic variates: E[(f(Z) + f(-Z))/2]
        Variance reduction: Var_AV = Var(naive) · (1 + ρ) / 2
        where ρ = Corr(f(Z), f(-Z)) < 0 for monotone payoffs → VR > 1.
        """
        rng = np.random.default_rng(seed)
        Z   = rng.standard_normal(self.N // 2)
        ST1 = self._terminal_prices(Z)
        ST2 = self._terminal_prices(-Z)
        p1  = self._payoff(ST1)
        p2  = self._payoff(ST2)
        avg = 0.5 * (p1 + p2)
        p   = self.disc * avg
        return p.mean(), p.std(ddof=1), p.var(ddof=1)

    def control_variate(self, seed=0):
        """
        Control variate: use E[S_T] = S0·e^{rT} (known).
        Adjusted estimator: ĝ = payoff - β·(S_T - E[S_T])
        Optimal β = Cov(payoff, S_T) / Var(S_T)
        Variance: Var_CV = Var(payoff) · (1 - ρ²)
        """
        rng = np.random.default_rng(seed)
        Z   = rng.standard_normal(self.N)
        ST  = self._terminal_prices(Z)
        p   = self._payoff(ST)

        E_ST = self.S0 * np.exp(self.r * self.T)
        beta = np.cov(p, ST)[0, 1] / np.var(ST)
        p_cv = p - beta * (ST - E_ST)
        p_cv_d = self.disc * p_cv
        return p_cv_d.mean(), p_cv_d.std(ddof=1), p_cv_d.var(ddof=1)

    def stratified(self, n_strata=100, seed=0):
        """
        Stratified sampling: divide [0,1] into n_strata equal strata.
        Sample one uniform U(k/n, (k+1)/n) per stratum → better coverage.
        Variance: Var_strat ≤ Var_naive (always, by ANOVA decomposition).
        """
        rng    = np.random.default_rng(seed)
        k      = np.arange(n_strata)
        n_each = self.N // n_strata
        us     = (k[:, None] + rng.uniform(0, 1, (n_strata, n_each))) / n_strata
        Z      = norm.ppf(us.ravel())
        ST     = self._terminal_prices(Z)
        p      = self.disc * self._payoff(ST)
        return p.mean(), p.std(ddof=1), p.var(ddof=1)

    def report(self) -> pd.DataFrame:
        naive_m,  naive_s,  naive_v  = self.naive()
        anti_m,   anti_s,   anti_v   = self.antithetic()
        cv_m,     cv_s,     cv_v     = self.control_variate()
        strat_m,  strat_s,  strat_v  = self.stratified()

        rows = [
            ("Naive MC",           naive_m,  naive_s,  1.0,                naive_v / naive_v),
            ("Antithetic",         anti_m,   anti_s,   naive_v / anti_v,   anti_v  / naive_v),
            ("Control Variate",    cv_m,     cv_s,     naive_v / cv_v,     cv_v    / naive_v),
            ("Stratified",         strat_m,  strat_s,  naive_v / strat_v,  strat_v / naive_v),
        ]
        df = pd.DataFrame(rows, columns=["Method", "Price", "Std Err",
                                          "VR Ratio", "Relative Var"])
        df["Error vs BS"] = (df["Price"] - self.bs_price).abs()
        return df


# ─────────────────────────────────────────────
# 4. BASKET OPTION
# ─────────────────────────────────────────────

def mc_basket_call(S0_vec, weights, K, r, sigma_vec, corr_matrix,
                    T, N=100_000, M=50, seed=42):
    """
    Basket call on weighted sum of n assets:
        Basket_T = Σ w_i · S_i(T)
        Payoff   = max(Basket_T - K, 0)
    """
    paths = cholesky_correlated_gbm(S0_vec, r, sigma_vec, corr_matrix, T, M, N, seed)
    ST    = paths[:, :, -1]  # (N, n)
    basket = ST @ weights
    disc   = np.exp(-r * T)
    payoffs = disc * np.maximum(basket - K, 0.0)
    price   = payoffs.mean()
    se      = payoffs.std(ddof=1) / np.sqrt(N)
    return price, se


# ─────────────────────────────────────────────
# 5. CONVERGENCE ANALYSIS
# ─────────────────────────────────────────────

def empirical_convergence(S0, K, r, T, sigma, N_max=500_000):
    """Show O(1/√N) convergence empirically."""
    from scipy.stats import norm as sp_norm
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs = S0*sp_norm.cdf(d1) - K*np.exp(-r*T)*sp_norm.cdf(d2)

    ns     = np.array([500, 1000, 5000, 10000, 50000, 100000, 500000])
    ns     = ns[ns <= N_max]
    errors = []

    for n in ns:
        rng = np.random.default_rng(0)
        Z   = rng.standard_normal(n)
        ST  = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        p   = np.exp(-r*T) * np.maximum(ST - K, 0).mean()
        errors.append(abs(p - bs))

    slope, _ = np.polyfit(np.log(ns), np.log(errors), 1)
    return ns, np.array(errors), slope, bs


# ─────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────

def plot_variance_reduction(report_df, bs_price):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Variance Reduction Techniques", fontsize=13, fontweight="bold")

    colors = ["steelblue", "crimson", "darkorange", "forestgreen"]
    ax = axes[0]
    bars = ax.bar(report_df["Method"], report_df["Std Err"],
                   color=colors, alpha=0.8)
    ax.axhline(report_df.loc[0, "Std Err"], color="black", linestyle="--",
                label="Naive benchmark")
    ax.set_ylabel("Standard Error")
    ax.set_title("Standard Error by Method")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, report_df["Std Err"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f"{val:.5f}", ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    bars2 = ax2.bar(report_df["Method"], report_df["VR Ratio"],
                     color=colors, alpha=0.8)
    ax2.axhline(1, color="black", linestyle="--")
    ax2.set_ylabel("Variance Reduction Ratio")
    ax2.set_title("Variance Reduction vs Naive")
    ax2.grid(True, alpha=0.3)
    for bar, val in zip(bars2, report_df["VR Ratio"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.1f}×", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("variance_reduction.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_sensitivity(S1, S2, sigma1, sigma2, T):
    rhos   = np.linspace(-0.95, 0.95, 50)
    prices = [margrabe_price(S1, S2, sigma1, sigma2, rho, T) for rho in rhos]

    plt.figure(figsize=(9, 5))
    plt.plot(rhos, prices, linewidth=2.5, color="steelblue")
    plt.xlabel("Correlation  ρ")
    plt.ylabel("Exchange Option Price")
    plt.title("Exchange Option Price vs Correlation  (Margrabe's Formula)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("exchange_option_corr.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT 7 — MULTI-ASSET MONTE CARLO")
    print("=" * 60)

    S0, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.20

    # ── Variance reduction comparison
    print("\n--- Variance Reduction Report ---")
    vr = VarianceReduction(S0, K, r, T, sigma, N=200_000)
    df = vr.report()
    print(df.to_string(index=False))
    naive_vr = df.loc[df["Method"] == "Antithetic", "VR Ratio"].values[0]
    print(f"\n  Antithetic VR:       {df[df['Method']=='Antithetic']['VR Ratio'].values[0]:.1f}×")
    print(f"  Control Variate VR:  {df[df['Method']=='Control Variate']['VR Ratio'].values[0]:.1f}×")
    plot_variance_reduction(df, vr.bs_price)

    # ── Exchange option (Margrabe validation)
    print("\n--- Exchange Option vs Margrabe ---")
    S1, S2   = 100.0, 105.0
    s1, s2   = 0.20, 0.25
    rho      = 0.50
    analytic = margrabe_price(S1, S2, s1, s2, rho, T)

    corr2    = np.array([[1.0, rho], [rho, 1.0]])
    paths    = cholesky_correlated_gbm(
        np.array([S1, S2]), r, np.array([s1, s2]), corr2, T, M=50, N=200_000)
    ST       = paths[:, :, -1]
    payoffs  = np.exp(-r*T) * np.maximum(ST[:, 0] - ST[:, 1], 0)
    mc_price = payoffs.mean()
    mc_se    = payoffs.std(ddof=1) / np.sqrt(len(payoffs))

    print(f"  Margrabe (closed-form): {analytic:.4f}")
    print(f"  MC estimate:            {mc_price:.4f} ± {mc_se:.5f}")
    print(f"  Absolute error:         {abs(mc_price - analytic):.5f}")
    plot_correlation_sensitivity(S1, S2, s1, s2, T)

    # ── Basket option
    print("\n--- Basket Call (3 assets) ---")
    S0_v = np.array([100.0, 95.0, 105.0])
    wts  = np.array([0.4, 0.3, 0.3])
    sigs = np.array([0.20, 0.25, 0.18])
    corr = np.array([[1.0, 0.6, 0.4],
                     [0.6, 1.0, 0.5],
                     [0.4, 0.5, 1.0]])
    K_b = np.dot(wts, S0_v)
    bprice, bse = mc_basket_call(S0_v, wts, K_b, r, sigs, corr, T)
    print(f"  Basket Call Price: {bprice:.4f} ± {bse:.5f}")

    # ── Convergence
    print("\n--- Convergence O(1/√N) ---")
    ns, errors, slope, bs_p = empirical_convergence(S0, K, r, T, sigma)
    print(f"  Empirical slope: {slope:.4f}  (theory: -0.50)")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(ns, errors, "o-", linewidth=2, label="MC Error")
    ax.loglog(ns, errors[0]*np.sqrt(ns[0]/ns), "--r", label="O(1/√N)")
    ax.set_xlabel("N paths")
    ax.set_ylabel("|MC − BS|")
    ax.set_title(f"Convergence: slope ≈ {slope:.3f}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence_multiasset.png", dpi=150, bbox_inches="tight")
    plt.show()
