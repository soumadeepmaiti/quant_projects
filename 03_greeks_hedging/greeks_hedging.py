"""
Greeks Computation & Delta-Gamma Hedging / PnL Attribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# ─────────────────────────────────────────────
# 1. BLACK-SCHOLES ANALYTIC GREEKS (BENCHMARK)
# ─────────────────────────────────────────────

def bs_price(S, K, r, T, sigma, option="call"):
    if T <= 1e-8:
        return max(S - K, 0) if option == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if option == "call":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def analytic_greeks(S, K, r, T, sigma, option="call"):
    """
    Black-Scholes Greeks:
        Δ = N(d1)
        Γ = N'(d1) / (S σ √T)
        Vega = S √T N'(d1)
        Θ = -Sσ N'(d1)/(2√T) - rKe^{-rT}N(d2)
        ρ = KTe^{-rT} N(d2)
    """
    if T <= 1e-8:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
    d1   = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    nd1  = norm.pdf(d1)

    if option == "call":
        delta = norm.cdf(d1)
        rho_g = K * T * disc * norm.cdf(d2)
        theta = (-S * sigma * nd1 / (2 * np.sqrt(T)) - r * K * disc * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        rho_g = -K * T * disc * norm.cdf(-d2)
        theta = (-S * sigma * nd1 / (2 * np.sqrt(T)) + r * K * disc * norm.cdf(-d2)) / 365

    return {
        "delta": delta,
        "gamma": nd1 / (S * sigma * np.sqrt(T)),
        "vega":  S * np.sqrt(T) * nd1 / 100,   # per 1 vol point
        "theta": theta,
        "rho":   rho_g / 100,                   # per 1bp
    }


# ─────────────────────────────────────────────
# 2. NUMERICAL GREEKS — CENTRAL FINITE DIFFERENCES
# ─────────────────────────────────────────────

class NumericalGreeks:
    """
    Central-difference Greeks for arbitrary pricing function f(S, K, r, T, sigma).
    Error analysis:
        Truncation error: O(h²)
        Rounding error:   O(ε_mach / h)
        Optimal h ≈ (ε_mach)^{1/3} ≈ 1e-5 for double precision
    """

    def __init__(self, pricing_fn, S, K, r, T, sigma, option="call"):
        self.f     = pricing_fn
        self.S     = S
        self.K     = K
        self.r     = r
        self.T     = T
        self.sigma = sigma
        self.opt   = option
        self._base = pricing_fn(S, K, r, T, sigma, option)

    def delta(self, h: float = 1e-4) -> float:
        """∂V/∂S  — central diff, O(h²)"""
        up   = self.f(self.S + h, self.K, self.r, self.T, self.sigma, self.opt)
        down = self.f(self.S - h, self.K, self.r, self.T, self.sigma, self.opt)
        return (up - down) / (2 * h)

    def gamma(self, h: float = 1e-4) -> float:
        """∂²V/∂S²  — requires 3 evaluations"""
        up   = self.f(self.S + h, self.K, self.r, self.T, self.sigma, self.opt)
        down = self.f(self.S - h, self.K, self.r, self.T, self.sigma, self.opt)
        return (up - 2 * self._base + down) / h ** 2

    def vega(self, h: float = 1e-4) -> float:
        """∂V/∂σ  (per 1 vol point → h=0.01)"""
        h_v  = 0.01
        up   = self.f(self.S, self.K, self.r, self.T, self.sigma + h_v, self.opt)
        down = self.f(self.S, self.K, self.r, self.T, self.sigma - h_v, self.opt)
        return (up - down) / (2 * h_v) * 0.01   # normalise to 1 vol point

    def theta(self, dt: float = 1 / 365) -> float:
        """∂V/∂t  (one calendar day)"""
        if self.T <= dt:
            return 0.0
        later = self.f(self.S, self.K, self.r, self.T - dt, self.sigma, self.opt)
        return (later - self._base)   # negative for long options

    def rho(self, dr: float = 1e-4) -> float:
        """∂V/∂r  (per 1bp)"""
        up   = self.f(self.S, self.K, self.r + dr, self.T, self.sigma, self.opt)
        down = self.f(self.S, self.K, self.r - dr, self.T, self.sigma, self.opt)
        return (up - down) / (2 * dr) * 1e-4  # normalise to 1bp

    def all(self, h: float = 1e-4) -> dict:
        return {
            "delta": self.delta(h),
            "gamma": self.gamma(h),
            "vega":  self.vega(),
            "theta": self.theta(),
            "rho":   self.rho(),
        }

    def step_size_error(self, greek: str, h_grid, true_val: float):
        """Sweep h and measure error — shows trunction vs rounding trade-off."""
        errors = []
        for h in h_grid:
            ng_temp = NumericalGreeks(self.f, self.S, self.K, self.r, self.T,
                                       self.sigma, self.opt)
            if greek == "delta":
                val = ng_temp.delta(h)
            elif greek == "gamma":
                val = ng_temp.gamma(h)
            else:
                val = true_val
            errors.append(abs(val - true_val))
        return np.array(errors)


# ─────────────────────────────────────────────
# 3. DELTA-HEDGING SIMULATION & PNL ATTRIBUTION
# ─────────────────────────────────────────────

def simulate_delta_hedge(S0, K, r, T, sigma, rebal_freq: int = 21,
                          N_paths: int = 1, seed: int = 0):
    """
    Simulate discrete delta-hedging of a short call.

    At each rebalancing date:
        1. Recompute Δ = N(d1)
        2. Adjust stock holding: trade (Δ_new - Δ_old) shares
        3. Finance trades from/to cash account

    PnL components:
        Total PnL  ≈  Δ·ΔS + ½Γ(ΔS)² + Θ·Δt
                     ↑ delta PnL   ↑ gamma PnL  ↑ theta PnL

    Residual PnL (unexplained by delta) ≈ ½Γ(ΔS)²
    """
    rng = np.random.default_rng(seed)
    M   = 252  # daily grid
    dt  = T / M
    rebal_dates = list(range(0, M, rebal_freq)) + [M]

    # Simulate single path
    Z  = rng.standard_normal(M)
    log_S = np.log(S0) + np.cumsum(
        (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    )
    S = np.concatenate([[S0], np.exp(log_S)])
    t_arr = np.linspace(0, T, M + 1)

    # Option short position initial premium received
    premium = bs_price(S0, K, r, T, sigma, "call")

    # Hedging book
    cash  = premium  # start: received premium
    delta_old = 0.0
    stock_pos = 0.0

    pnl_daily   = []
    delta_pnl   = []
    gamma_pnl   = []
    theta_pnl   = []

    prev_V = premium

    for i in range(M):
        T_rem = T - t_arr[i]
        if T_rem < 1e-8:
            break

        # Rebalance at rebal_dates
        if i in rebal_dates:
            delta_new = analytic_greeks(S[i], K, r, T_rem, sigma)["delta"]
            trade     = delta_new - stock_pos
            cash     -= trade * S[i]          # buy/sell shares
            stock_pos = delta_new

        # Carry cash at risk-free rate
        cash *= np.exp(r * dt)

        # PnL attribution for this step
        dS = S[i + 1] - S[i]
        gamma_i = analytic_greeks(S[i], K, r, T_rem, sigma)["gamma"]
        theta_i = analytic_greeks(S[i], K, r, T_rem, sigma)["theta"]  # per day

        d_pnl = stock_pos * dS          # delta component
        g_pnl = 0.5 * gamma_i * dS ** 2
        t_pnl = theta_i                 # theta per day

        new_V = bs_price(S[i + 1], K, r, max(T - t_arr[i + 1], 1e-8), sigma, "call")
        actual_pnl = -(new_V - prev_V) + stock_pos * dS + cash * (np.exp(r * dt) - 1)

        pnl_daily.append(actual_pnl)
        delta_pnl.append(d_pnl)
        gamma_pnl.append(g_pnl)
        theta_pnl.append(t_pnl)
        prev_V = new_V

    # Terminal: close stock position, option expires
    ST      = S[M]
    payoff  = max(ST - K, 0)
    port_val = stock_pos * ST + cash - payoff  # net P&L
    pnl_daily.append(port_val - sum(pnl_daily))

    return {
        "S_path":       S,
        "total_pnl":    np.array(pnl_daily),
        "delta_pnl":    np.array(delta_pnl),
        "gamma_pnl":    np.array(gamma_pnl),
        "theta_pnl":    np.array(theta_pnl),
        "terminal_pnl": port_val,
    }


def rebalancing_frequency_study(S0, K, r, T, sigma,
                                 freqs=(1, 5, 10, 21, 63),
                                 N_paths=500):
    """
    Measure hedge error (std of terminal PnL) vs rebalancing frequency.
    Theory: hedge error ∝ σ·√(Δt) → more rebalancing → smaller error.
    """
    rng = np.random.default_rng(42)
    results = []
    for freq in freqs:
        pnls = []
        for s in range(N_paths):
            res = simulate_delta_hedge(S0, K, r, T, sigma,
                                        rebal_freq=freq, seed=s)
            pnls.append(res["terminal_pnl"])
        pnls = np.array(pnls)
        results.append({
            "Rebal Freq (days)": freq,
            "Mean PnL":         np.mean(pnls),
            "Std PnL":          np.std(pnls),
            "RMSE":             np.sqrt(np.mean(pnls ** 2)),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 4. GREEKS SURFACE PLOTS
# ─────────────────────────────────────────────

def plot_greeks_surface(K=100, r=0.05, T=1.0, sigma=0.20):
    S_range = np.linspace(60, 140, 80)
    greeks_surface = {g: [] for g in ["delta", "gamma", "vega", "theta"]}
    for S in S_range:
        g = analytic_greeks(S, K, r, T, sigma, "call")
        for key in greeks_surface:
            greeks_surface[key].append(g[key])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Black-Scholes Greeks — Call Option (K=100, T=1y, σ=20%)",
                  fontsize=13, fontweight="bold")
    labels = {"delta": "Δ  (∂V/∂S)", "gamma": "Γ  (∂²V/∂S²)",
              "vega":  "Vega (per 1% σ)", "theta": "Θ (per day)"}
    colors = ["steelblue", "crimson", "darkorange", "forestgreen"]

    for ax, (key, vals), col in zip(axes.flatten(),
                                     greeks_surface.items(), colors):
        ax.plot(S_range, vals, linewidth=2.5, color=col, label=labels[key])
        ax.axvline(K, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Spot Price S")
        ax.set_title(labels[key])
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("greeks_surface.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_delta_hedge_pnl(S0=100, K=100, r=0.05, T=1.0, sigma=0.20):
    res = simulate_delta_hedge(S0, K, r, T, sigma, rebal_freq=5)
    t   = np.arange(len(res["S_path"]))

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    fig.suptitle("Delta-Hedging P&L Attribution  (rebal every 5 days)",
                  fontsize=13, fontweight="bold")

    axes[0].plot(t, res["S_path"], color="steelblue", linewidth=1.5)
    axes[0].axhline(K, color="red", linestyle="--", label=f"K={K}")
    axes[0].set_ylabel("Stock Price")
    axes[0].set_title("Simulated GBM Path")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    cumulative = np.cumsum(res["total_pnl"][:-1])
    axes[1].plot(cumulative, color="navy", linewidth=1.5, label="Cumulative PnL")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("P&L")
    axes[1].set_title("Cumulative Hedging P&L")
    axes[1].grid(True, alpha=0.3)

    gamma_cum = np.cumsum(res["gamma_pnl"])
    theta_cum = np.cumsum(res["theta_pnl"])
    axes[2].plot(gamma_cum, label="½Γ(ΔS)²  — Gamma P&L", color="crimson")
    axes[2].plot(theta_cum, label="Θ·Δt  — Theta P&L",   color="darkorange", linestyle="--")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Trading Days")
    axes[2].set_ylabel("Cumulative P&L")
    axes[2].set_title("PnL Attribution: Gamma vs Theta")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("delta_hedge_pnl.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    S0, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.20

    print("=" * 60)
    print("PROJECT 3 — GREEKS & DELTA-GAMMA HEDGING")
    print("=" * 60)

    # Analytic Greeks
    ag = analytic_greeks(S0, K, r, T, sigma)
    print("\nAnalytic Greeks (Black-Scholes):")
    for k, v in ag.items():
        print(f"  {k:<8}: {v:.6f}")

    # Numerical Greeks
    ng = NumericalGreeks(bs_price, S0, K, r, T, sigma)
    fd = ng.all()
    print("\nNumerical Greeks (Central FD, h=1e-4):")
    for k, v in fd.items():
        print(f"  {k:<8}: {v:.6f}  |  error vs analytic: {abs(v - ag[k]):.2e}")

    # h sweep
    print("\nDelta error vs step size h:")
    h_grid = np.logspace(-7, -1, 15)
    errs   = ng.step_size_error("delta", h_grid, ag["delta"])
    opt_h  = h_grid[np.argmin(errs)]
    print(f"  Optimal h ≈ {opt_h:.1e}  (error={min(errs):.2e})")

    # Greeks surface plot
    plot_greeks_surface()

    # Delta hedge
    print("\n--- Delta Hedging PnL Attribution ---")
    res = simulate_delta_hedge(S0, K, r, T, sigma, rebal_freq=5, seed=7)
    print(f"  Terminal PnL: {res['terminal_pnl']:.4f}")
    print(f"  Gamma P&L (cumulative): {res['gamma_pnl'].sum():.4f}")
    print(f"  Theta P&L (cumulative): {res['theta_pnl'].sum():.4f}")
    plot_delta_hedge_pnl()

    # Rebalancing frequency study
    print("\n--- Rebalancing Frequency Study ---")
    df = rebalancing_frequency_study(S0, K, r, T, sigma)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["Rebal Freq (days)"], df["Std PnL"], "o-", linewidth=2, color="steelblue")
    ax.set_xlabel("Rebalancing Frequency (days)")
    ax.set_ylabel("Std of Terminal PnL")
    ax.set_title("Hedge Error vs Rebalancing Frequency")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rebal_frequency.png", dpi=150, bbox_inches="tight")
    plt.show()
