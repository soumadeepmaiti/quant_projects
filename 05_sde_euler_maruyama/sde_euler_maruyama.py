"""
SDE Simulation — Euler–Maruyama & Convergence Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable


# ─────────────────────────────────────────────
# 1. GENERAL EULER-MARUYAMA SOLVER
# ─────────────────────────────────────────────

class EulerMaruyama:
    """
    Euler–Maruyama discretisation of the general SDE:
        dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t

    Discretisation:
        X_{t+Δt} = X_t + μ(X_t, t)·Δt + σ(X_t, t)·√Δt·Z,   Z ~ N(0,1)

    Strong convergence order:  γ = 0.5  (pathwise)
    Weak convergence order:    β = 1.0  (distributional)
    """

    def __init__(self,
                 mu:    Callable[[float, float], float],
                 sigma: Callable[[float, float], float],
                 X0: float, T: float):
        self.mu    = mu
        self.sigma = sigma
        self.X0    = X0
        self.T     = T

    def simulate(self, steps: int, seed: int = 0) -> tuple:
        """Returns (t_arr, X_arr) for a single path."""
        dt  = self.T / steps
        rng = np.random.default_rng(seed)
        Z   = rng.standard_normal(steps)
        t   = np.linspace(0, self.T, steps + 1)
        X   = np.empty(steps + 1)
        X[0] = self.X0

        for i in range(steps):
            X[i + 1] = (X[i]
                        + self.mu(X[i], t[i]) * dt
                        + self.sigma(X[i], t[i]) * np.sqrt(dt) * Z[i])
        return t, X

    def simulate_paths(self, steps: int, n_paths: int, seed: int = 0) -> np.ndarray:
        """Simulate n_paths paths. Returns array (n_paths, steps+1)."""
        dt  = self.T / steps
        rng = np.random.default_rng(seed)
        Z   = rng.standard_normal((n_paths, steps))
        t   = np.linspace(0, self.T, steps + 1)

        X    = np.empty((n_paths, steps + 1))
        X[:, 0] = self.X0

        for i in range(steps):
            mu_v    = np.array([self.mu(X[k, i], t[i]) for k in range(n_paths)])
            sigma_v = np.array([self.sigma(X[k, i], t[i]) for k in range(n_paths)])
            X[:, i + 1] = X[:, i] + mu_v * dt + sigma_v * np.sqrt(dt) * Z[:, i]
        return X


# ─────────────────────────────────────────────
# 2. EXACT SOLUTIONS (GROUND TRUTH FOR CONVERGENCE TESTS)
# ─────────────────────────────────────────────

def gbm_exact(X0, r, sigma, T, steps, seed=0):
    """
    Exact GBM solution via Itô's lemma:
        d(ln S) = (r - σ²/2)dt + σ dW
    →  S_T = S_0 · exp((r - σ²/2)T + σ W_T)
    """
    dt  = T / steps
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(steps)
    t   = np.linspace(0, T, steps + 1)
    S   = np.empty(steps + 1)
    S[0] = X0
    for i in range(steps):
        S[i + 1] = S[i] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[i])
    return t, S


def ou_exact(X0, kappa, theta, sigma, T, N_paths=5000, seed=0):
    """
    Exact Ornstein–Uhlenbeck terminal distribution:
        X_T | X_0 ~ N(μ_T, σ²_T)
        μ_T = X_0·e^{-κT} + θ(1 - e^{-κT})
        σ²_T = σ²(1 - e^{-2κT}) / (2κ)
    Returns (mean, variance) analytically.
    """
    mu_T    = X0 * np.exp(-kappa * T) + theta * (1 - np.exp(-kappa * T))
    var_T   = sigma ** 2 * (1 - np.exp(-2 * kappa * T)) / (2 * kappa)
    return mu_T, var_T


# ─────────────────────────────────────────────
# 3. CONVERGENCE ANALYSIS
# ─────────────────────────────────────────────

def strong_convergence_study(solver: EulerMaruyama,
                              exact_fn: Callable,
                              steps_grid: list,
                              n_paths: int = 2000) -> pd.DataFrame:
    """
    Strong error:
        E[|X^{EM}_T - X^{exact}_T|] ≤ C · Δt^{1/2}

    We use the SAME Brownian path increments for EM and exact.
    """
    results = []
    for steps in steps_grid:
        errors = []
        dt = solver.T / steps
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((n_paths, steps))

        for k in range(n_paths):
            # EM path
            X_em = solver.X0
            for i in range(steps):
                t_i = i * dt
                X_em += (solver.mu(X_em, t_i) * dt
                         + solver.sigma(X_em, t_i) * np.sqrt(dt) * Z[k, i])

            # Exact terminal value (using same Z)
            X_exact = exact_fn(Z[k], dt)
            errors.append(abs(X_em - X_exact))

        results.append({
            "Steps":         steps,
            "dt":            dt,
            "Strong Error":  np.mean(errors),
            "Std":           np.std(errors),
        })

    df = pd.DataFrame(results)
    slope, _ = np.polyfit(np.log(df["dt"]), np.log(df["Strong Error"]), 1)
    df["Convergence Rate"] = slope
    return df, slope


def weak_convergence_study(solver: EulerMaruyama,
                            exact_mean: float,
                            exact_var: float,
                            steps_grid: list,
                            n_paths: int = 5000) -> pd.DataFrame:
    """
    Weak error (error in expectation):
        |E[X^{EM}_T] - E[X^{exact}_T]| ≤ C · Δt
    """
    results = []
    for steps in steps_grid:
        X_paths = solver.simulate_paths(steps, n_paths, seed=42)
        em_mean = X_paths[:, -1].mean()
        em_var  = X_paths[:, -1].var()

        results.append({
            "Steps":      steps,
            "dt":         solver.T / steps,
            "EM Mean":    em_mean,
            "Exact Mean": exact_mean,
            "Weak Error": abs(em_mean - exact_mean),
            "Var Error":  abs(em_var - exact_var),
        })

    df = pd.DataFrame(results)
    slope, _ = np.polyfit(np.log(df["dt"]), np.log(df["Weak Error"] + 1e-15), 1)
    return df, slope


# ─────────────────────────────────────────────
# 4. PROCESS IMPLEMENTATIONS
# ─────────────────────────────────────────────

def build_ou_process(kappa, theta, sigma, X0, T):
    """Ornstein–Uhlenbeck:  dX = κ(θ - X)dt + σ dW"""
    return EulerMaruyama(
        mu    = lambda X, t: kappa * (theta - X),
        sigma = lambda X, t: sigma,
        X0=X0, T=T
    )


def build_gbm_process(r, sigma, S0, T):
    """Geometric Brownian Motion:  dS = r·S dt + σ·S dW"""
    return EulerMaruyama(
        mu    = lambda S, t: r * S,
        sigma = lambda S, t: sigma * S,
        X0=S0, T=T
    )


def build_cir_process(kappa, theta, sigma, X0, T):
    """
    Cox–Ingersoll–Ross:  dr = κ(θ - r)dt + σ√r dW
    Positivity condition: 2κθ ≥ σ²  (Feller condition)
    Note: EM can produce negative values → clamp to zero.
    """
    def cir_mu(r, t):    return kappa * (theta - r)
    def cir_sigma(r, t): return sigma * np.sqrt(max(r, 0))
    return EulerMaruyama(mu=cir_mu, sigma=cir_sigma, X0=X0, T=T)


def build_heston_vol(kappa, theta, xi, rho, v0, T):
    """
    Heston stochastic volatility (variance process):
        dv_t = κ(θ - v_t) dt + ξ√v_t dW_t
    """
    def heston_mu(v, t):    return kappa * (theta - v)
    def heston_sigma(v, t): return xi * np.sqrt(max(v, 0))
    return EulerMaruyama(mu=heston_mu, sigma=heston_sigma, X0=v0, T=T)


# ─────────────────────────────────────────────
# 5. BROWNIAN MOTION PROPERTIES
# ─────────────────────────────────────────────

def demonstrate_brownian_motion(T=1.0, steps=1000, n_paths=5, seed=0):
    """
    Demonstrate key Brownian motion properties:
      1. Markov property: W_t ~ N(0, t) — only depends on current state
      2. Martingale:      E[W_t | F_s] = W_s  for t > s
      3. Quadratic variation: [W]_T = T  (almost surely)
    """
    rng = np.random.default_rng(seed)
    dt  = T / steps
    Z   = rng.standard_normal((n_paths, steps))
    t   = np.linspace(0, T, steps + 1)
    W   = np.zeros((n_paths, steps + 1))
    W[:, 1:] = np.cumsum(np.sqrt(dt) * Z, axis=1)

    # Quadratic variation ≈ T
    qv = np.sum(np.diff(W, axis=1) ** 2, axis=1)

    return t, W, qv


# ─────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────

def plot_convergence(strong_df, weak_df, strong_rate, weak_rate):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Euler–Maruyama Convergence Analysis", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.loglog(strong_df["dt"], strong_df["Strong Error"],
               "o-", linewidth=2, label="Strong Error")
    ref = strong_df["Strong Error"].iloc[0] * np.sqrt(
        strong_df["dt"] / strong_df["dt"].iloc[0])
    ax.loglog(strong_df["dt"], ref, "--r", label="O(√Δt)")
    ax.set_xlabel("Δt")
    ax.set_ylabel("E[|X^EM − X^exact|]")
    ax.set_title(f"Strong Convergence  (slope ≈ {strong_rate:.2f}, theory: 0.50)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax2 = axes[1]
    ax2.loglog(weak_df["dt"], weak_df["Weak Error"] + 1e-12,
                "s-", linewidth=2, color="crimson", label="Weak Error")
    ref2 = weak_df["Weak Error"].iloc[0] * (weak_df["dt"] / weak_df["dt"].iloc[0])
    ax2.loglog(weak_df["dt"], ref2, "--k", label="O(Δt)")
    ax2.set_xlabel("Δt")
    ax2.set_ylabel("|E[X^EM] − E[X^exact]|")
    ax2.set_title(f"Weak Convergence  (slope ≈ {weak_rate:.2f}, theory: 1.00)")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("sde_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sde_paths():
    T = 1.0
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("SDE Process Simulations", fontsize=13, fontweight="bold")

    # GBM
    solver_gbm = build_gbm_process(r=0.05, sigma=0.20, S0=100, T=T)
    for seed in range(8):
        t, X = solver_gbm.simulate(steps=252, seed=seed)
        axes[0, 0].plot(t, X, alpha=0.6, linewidth=0.9)
    axes[0, 0].set_title("GBM:  dS = 0.05·S dt + 0.20·S dW")
    axes[0, 0].set_ylabel("S_t")
    axes[0, 0].grid(True, alpha=0.3)

    # OU
    solver_ou = build_ou_process(kappa=2.0, theta=1.0, sigma=0.3, X0=0.5, T=T)
    for seed in range(8):
        t, X = solver_ou.simulate(steps=500, seed=seed)
        axes[0, 1].plot(t, X, alpha=0.6, linewidth=0.9)
    axes[0, 1].axhline(1.0, color="red", linestyle="--", linewidth=1.5,
                        label="θ = 1.0 (long-run mean)")
    axes[0, 1].set_title("Ornstein–Uhlenbeck:  κ=2, θ=1, σ=0.3")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # CIR
    solver_cir = build_cir_process(kappa=0.8, theta=0.05, sigma=0.10, X0=0.03, T=T)
    for seed in range(8):
        t, X = solver_cir.simulate(steps=500, seed=seed)
        axes[1, 0].plot(t, X * 100, alpha=0.6, linewidth=0.9)
    axes[1, 0].axhline(0.05 * 100, color="red", linestyle="--",
                        linewidth=1.5, label="θ = 5%")
    axes[1, 0].set_title("CIR (Interest Rate):  κ=0.8, θ=5%, σ=10%")
    axes[1, 0].set_ylabel("Rate (%)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Heston vol
    solver_heston = build_heston_vol(kappa=1.5, theta=0.04, xi=0.3, rho=-0.7,
                                      v0=0.04, T=T)
    for seed in range(8):
        t, X = solver_heston.simulate(steps=500, seed=seed)
        axes[1, 1].plot(t, np.sqrt(np.maximum(X, 0)) * 100, alpha=0.6, linewidth=0.9)
    axes[1, 1].axhline(np.sqrt(0.04) * 100, color="red", linestyle="--",
                        linewidth=1.5, label="√θ = 20%")
    axes[1, 1].set_title("Heston Variance Process:  κ=1.5, θ=4%, ξ=0.3")
    axes[1, 1].set_ylabel("Instantaneous Vol (%)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flatten():
        ax.set_xlabel("Time (years)")

    plt.tight_layout()
    plt.savefig("sde_paths.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_brownian_motion(T=1.0, steps=1000, n_paths=8):
    t, W, qv = demonstrate_brownian_motion(T, steps, n_paths)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Brownian Motion Properties", fontsize=13, fontweight="bold")

    for i in range(n_paths):
        axes[0].plot(t, W[i], alpha=0.7, linewidth=0.9)
    axes[0].fill_between(t, -2 * np.sqrt(t), 2 * np.sqrt(t), alpha=0.1,
                          color="red", label="±2σ = ±2√t")
    axes[0].set_title("Sample Paths  W_t")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Distribution at t=T: should be N(0, T)
    W_T = W[:, -1]
    axes[1].hist(W_T, bins=20, density=True, alpha=0.7, label="W_T histogram")
    x = np.linspace(W_T.min() * 1.5, W_T.max() * 1.5, 100)
    from scipy.stats import norm
    axes[1].plot(x, norm.pdf(x, 0, np.sqrt(T)), "r-", linewidth=2, label=f"N(0,{T})")
    axes[1].set_title(f"Distribution at T={T}: N(0,T)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Quadratic variation → T almost surely
    axes[2].bar(range(n_paths), qv, color="steelblue", alpha=0.8)
    axes[2].axhline(T, color="red", linestyle="--", linewidth=2, label=f"[W]_T = T = {T}")
    axes[2].set_xlabel("Path index")
    axes[2].set_ylabel("Quadratic Variation")
    axes[2].set_title("Quadratic Variation ≈ T  (a.s.)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("brownian_properties.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT 5 — EULER–MARUYAMA SDE SIMULATION")
    print("=" * 60)

    # ── GBM convergence (exact solution available via Itô)
    r0, sigma0, S0, T = 0.05, 0.20, 100.0, 1.0
    solver_gbm = build_gbm_process(r0, sigma0, S0, T)

    def gbm_exact_terminal(Z, dt):
        """Exact GBM terminal from the same Z increments."""
        log_incr = (r0 - 0.5 * sigma0 ** 2) * dt + sigma0 * np.sqrt(dt) * Z
        return S0 * np.exp(np.sum(log_incr))

    steps_grid = [10, 25, 50, 100, 250, 500]
    print("\nStrong convergence study (GBM):")
    strong_df, strong_rate = strong_convergence_study(
        solver_gbm, gbm_exact_terminal, steps_grid, n_paths=1000)
    print(strong_df[["Steps", "dt", "Strong Error"]].to_string(index=False))
    print(f"  → Empirical convergence rate: {strong_rate:.3f}  (theory: 0.50)")

    # ── OU weak convergence
    kappa, theta, sig_ou, X0_ou = 2.0, 1.0, 0.3, 0.5
    solver_ou = build_ou_process(kappa, theta, sig_ou, X0_ou, T)
    exact_mu_ou, exact_var_ou = ou_exact(X0_ou, kappa, theta, sig_ou, T)
    print(f"\nOU process  exact terminal: mean={exact_mu_ou:.4f}, var={exact_var_ou:.4f}")
    weak_df, weak_rate = weak_convergence_study(
        solver_ou, exact_mu_ou, exact_var_ou, steps_grid)
    print("\nWeak convergence study (OU):")
    print(weak_df[["Steps", "dt", "EM Mean", "Exact Mean", "Weak Error"]].to_string(index=False))
    print(f"  → Empirical convergence rate: {weak_rate:.3f}  (theory: 1.00)")

    plot_convergence(strong_df, weak_df, strong_rate, weak_rate)
    plot_sde_paths()
    plot_brownian_motion()
