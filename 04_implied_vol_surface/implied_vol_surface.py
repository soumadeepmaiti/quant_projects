"""
Implied Volatility Surface & Arbitrage Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
import warnings


# ─────────────────────────────────────────────
# 1. BLACK-SCHOLES PRICE & VEGA
# ─────────────────────────────────────────────

def bs_price(S: float, K: float, r: float, T: float, sigma: float,
             option: str = "call") -> float:
    if T <= 1e-8 or sigma <= 1e-8:
        return max(S - K, 0) if option == "call" else max(K - S, 0)
    d1   = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    if option == "call":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """∂C/∂σ = S√T N'(d1)"""
    if T <= 1e-8 or sigma <= 1e-8:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


# ─────────────────────────────────────────────
# 2. IMPLIED VOLATILITY — NEWTON-RAPHSON + BISECTION FALLBACK
# ─────────────────────────────────────────────

def implied_vol(C_mkt: float, S: float, K: float, r: float, T: float,
                option: str = "call", sigma0: float = 0.20,
                tol: float = 1e-8, max_iter: int = 100) -> float:
    """
    Solve C_BS(σ) = C_mkt for σ via Newton-Raphson:
        σ_{n+1} = σ_n − (C_BS(σ_n) − C_mkt) / Vega(σ_n)

    Quadratic convergence when Vega is large (ATM).
    Falls back to bisection if Vega < ε (deep ITM/OTM).

    Arbitrage bounds check:
        call: max(Se^{-qT} - Ke^{-rT}, 0) ≤ C ≤ Se^{-qT}
    """
    # Intrinsic value bounds
    disc = np.exp(-r * T)
    lb   = max(S - K * disc, 0) if option == "call" else max(K * disc - S, 0)
    ub   = S if option == "call" else K * disc

    if C_mkt < lb - 1e-4 or C_mkt > ub + 1e-4:
        return np.nan  # arbitrage violation

    sigma = sigma0

    for _ in range(max_iter):
        price  = bs_price(S, K, r, T, sigma, option)
        vega   = bs_vega(S, K, r, T, sigma)
        err    = price - C_mkt

        if abs(err) < tol:
            return sigma

        if vega < 1e-10:
            # Fallback: bisection
            return _bisection_iv(C_mkt, S, K, r, T, option, tol)

        sigma -= err / vega
        sigma  = np.clip(sigma, 1e-4, 10.0)

    return _bisection_iv(C_mkt, S, K, r, T, option, tol)


def _bisection_iv(C_mkt, S, K, r, T, option, tol=1e-8):
    """Bisection fallback — guaranteed convergence."""
    lo, hi = 1e-4, 8.0
    f_lo = bs_price(S, K, r, T, lo, option) - C_mkt
    f_hi = bs_price(S, K, r, T, hi, option) - C_mkt
    if f_lo * f_hi > 0:
        return np.nan

    for _ in range(200):
        mid  = (lo + hi) / 2
        f_mid = bs_price(S, K, r, T, mid, option) - C_mkt
        if abs(f_mid) < tol or (hi - lo) / 2 < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────
# 3. YIELD CURVE BOOTSTRAPPING
# ─────────────────────────────────────────────

def bootstrap_discount_factors(maturities: np.ndarray,
                                par_rates: np.ndarray) -> pd.DataFrame:
    """
    Bootstrap zero-coupon discount factors from par rates (annual coupon bonds).
    For each maturity T_n:
        P(T_n) = (1 - c_n · Σ_{k<n} P(T_k) · Δt_k) / (1 + c_n · Δt_n)
    where c_n is the par rate (coupon = yield for par bond).
    Returns DataFrame with maturities, zero rates, and discount factors.
    """
    n  = len(maturities)
    df = np.zeros(n)
    dt = np.diff(np.concatenate([[0], maturities]))

    for i in range(n):
        c = par_rates[i]
        coupon_sum = sum(c * dt[j] * df[j] for j in range(i))
        df[i] = (1 - coupon_sum) / (1 + c * dt[i])

    zero_rates = -np.log(df) / maturities

    return pd.DataFrame({
        "Maturity":        maturities,
        "Par Rate":        par_rates,
        "Discount Factor": df,
        "Zero Rate":       zero_rates,
    })


def interpolate_discount_factor(bootstrap_df: pd.DataFrame, T: float) -> float:
    """Linear interpolation of bootstrapped discount factors."""
    mats = bootstrap_df["Maturity"].values
    dfs  = bootstrap_df["Discount Factor"].values
    if T <= mats[0]:
        return dfs[0]
    if T >= mats[-1]:
        return dfs[-1]
    return float(np.interp(T, mats, dfs))


# ─────────────────────────────────────────────
# 4. BUILD IV SURFACE
# ─────────────────────────────────────────────

def build_iv_surface(S: float, r: float,
                      strikes: np.ndarray, maturities: np.ndarray,
                      smile_func=None) -> np.ndarray:
    """
    Construct IV surface on a (T × K) grid.
    `smile_func(K, T, S, r)` generates synthetic market prices.
    Default: SVI-like skew with realistic vol smile.
    """
    if smile_func is None:
        def smile_func(K, T, S, r):
            """Synthetic smile: flat σ + skew + curvature."""
            moneyness = np.log(K / S)
            base_vol  = 0.20
            skew      = -0.05 * moneyness / np.sqrt(T)
            smile_    = 0.10 * moneyness ** 2 / T
            return np.clip(base_vol + skew + smile_, 0.05, 1.5)

    K_grid = strikes
    T_grid = maturities
    iv_surf = np.zeros((len(T_grid), len(K_grid)))

    for i, T in enumerate(T_grid):
        for j, K in enumerate(K_grid):
            true_sigma = smile_func(K, T, S, r)
            C_mkt      = bs_price(S, K, r, T, true_sigma, "call")
            iv_surf[i, j] = implied_vol(C_mkt, S, K, r, T, sigma0=true_sigma * 0.9)

    return iv_surf


# ─────────────────────────────────────────────
# 5. ARBITRAGE DETECTION
# ─────────────────────────────────────────────

def detect_calendar_arbitrage(iv_surface: np.ndarray,
                               maturities: np.ndarray,
                               strikes: np.ndarray) -> pd.DataFrame:
    """
    Calendar spread arbitrage:
        Total variance w(K,T) = σ²_iv(K,T) · T must be non-decreasing in T.
        Violation: w(K, T2) < w(K, T1) for T2 > T1.

    Returns DataFrame of violations.
    """
    total_var = iv_surface ** 2 * maturities[:, None]
    violations = []

    for j, K in enumerate(strikes):
        for i in range(len(maturities) - 1):
            T1, T2 = maturities[i], maturities[i + 1]
            w1, w2 = total_var[i, j], total_var[i + 1, j]
            if w2 < w1 - 1e-6:
                violations.append({
                    "Strike": K,
                    "T1": T1, "T2": T2,
                    "w(T1)": round(w1, 5),
                    "w(T2)": round(w2, 5),
                    "Violation": round(w1 - w2, 5),
                })
    df = pd.DataFrame(violations)
    return df


def detect_butterfly_arbitrage(iv_surface: np.ndarray,
                                strikes: np.ndarray,
                                maturities: np.ndarray,
                                S: float, r: float) -> pd.DataFrame:
    """
    Butterfly arbitrage:
        ∂²C/∂K² ≥ 0  (call price is convex in K → risk-neutral density ≥ 0).
        Approximation via second central difference:
            C(K-h) - 2C(K) + C(K+h) ≥ 0
    """
    violations = []
    for i, T in enumerate(maturities):
        for j in range(1, len(strikes) - 1):
            K_m, K, K_p = strikes[j - 1], strikes[j], strikes[j + 1]
            iv_m = iv_surface[i, j - 1]
            iv_0 = iv_surface[i, j]
            iv_p = iv_surface[i, j + 1]

            if any(np.isnan([iv_m, iv_0, iv_p])):
                continue

            C_m = bs_price(S, K_m, r, T, iv_m, "call")
            C_0 = bs_price(S, K,   r, T, iv_0, "call")
            C_p = bs_price(S, K_p, r, T, iv_p, "call")

            butterfly = C_m - 2 * C_0 + C_p
            if butterfly < -1e-5:
                violations.append({
                    "Maturity": T,
                    "K_left": K_m, "K_mid": K, "K_right": K_p,
                    "Butterfly Value": round(butterfly, 5),
                })

    return pd.DataFrame(violations)


def detect_put_call_parity_violations(S, r, T_arr, K_arr, iv_surface):
    """
    Put-call parity: C - P = S - K·e^{-rT}
    Check consistency of IV surface.
    """
    violations = []
    for i, T in enumerate(T_arr):
        for j, K in enumerate(K_arr):
            iv = iv_surface[i, j]
            if np.isnan(iv):
                continue
            C = bs_price(S, K, r, T, iv, "call")
            P = bs_price(S, K, r, T, iv, "put")
            pcp = C - P - (S - K * np.exp(-r * T))
            if abs(pcp) > 1e-4:
                violations.append({"T": T, "K": K, "PCP Error": round(pcp, 6)})
    return pd.DataFrame(violations)


# ─────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────

def plot_iv_surface(iv_surface, strikes, maturities):
    from mpl_toolkits.mplot3d import Axes3D
    K_mesh, T_mesh = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(K_mesh, T_mesh, iv_surface * 100,
                             cmap="RdYlGn_r", alpha=0.85)
    ax1.set_xlabel("Strike K")
    ax1.set_ylabel("Maturity T (yrs)")
    ax1.set_zlabel("Implied Vol (%)")
    ax1.set_title("Implied Volatility Surface")
    fig.colorbar(surf, ax=ax1, shrink=0.4, pad=0.1)

    ax2 = fig.add_subplot(122)
    for i, T in enumerate(maturities):
        ax2.plot(strikes, iv_surface[i] * 100, label=f"T={T:.2f}y", linewidth=1.8)
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Implied Vol (%)")
    ax2.set_title("Volatility Smile (per maturity)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("iv_surface.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_discount_curve(boot_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Yield Curve Bootstrapping", fontsize=13, fontweight="bold")

    axes[0].plot(boot_df["Maturity"], boot_df["Discount Factor"],
                  "o-", linewidth=2, color="steelblue")
    axes[0].set_xlabel("Maturity (yrs)")
    axes[0].set_ylabel("Discount Factor P(0,T)")
    axes[0].set_title("Bootstrapped Discount Factors")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(boot_df["Maturity"], boot_df["Zero Rate"] * 100,
                  "s-", linewidth=2, color="crimson", label="Zero Rate")
    axes[1].plot(boot_df["Maturity"], boot_df["Par Rate"] * 100,
                  "--", linewidth=1.5, color="navy", label="Par Rate")
    axes[1].set_xlabel("Maturity (yrs)")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_title("Zero vs Par Rates")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("yield_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    S = 100.0
    r = 0.04

    print("=" * 60)
    print("PROJECT 4 — IMPLIED VOLATILITY SURFACE & ARBITRAGE")
    print("=" * 60)

    # ── Yield curve bootstrapping
    print("\n--- Yield Curve Bootstrapping ---")
    mats  = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    rates = np.array([0.040, 0.042, 0.045, 0.048, 0.050, 0.052, 0.053, 0.055])
    boot  = bootstrap_discount_factors(mats, rates)
    print(boot.to_string(index=False))
    plot_discount_curve(boot)

    # ── IV surface construction
    print("\n--- Building IV Surface ---")
    strikes    = np.array([75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125], dtype=float)
    maturities = np.array([0.08, 0.17, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0])
    iv_surf    = build_iv_surface(S, r, strikes, maturities)
    print(f"  Surface shape: {iv_surf.shape}  (T × K)")
    print(f"  IV range: [{iv_surf[~np.isnan(iv_surf)].min()*100:.1f}%, "
          f"{iv_surf[~np.isnan(iv_surf)].max()*100:.1f}%]")
    plot_iv_surface(iv_surf, strikes, maturities)

    # ── Arbitrage detection
    print("\n--- Calendar Spread Arbitrage Check ---")
    cal_viol = detect_calendar_arbitrage(iv_surf, maturities, strikes)
    if cal_viol.empty:
        print("  ✓ No calendar spread violations found.")
    else:
        print(f"  ✗ {len(cal_viol)} violation(s) found:")
        print(cal_viol.to_string(index=False))

    print("\n--- Butterfly Arbitrage Check ---")
    but_viol = detect_butterfly_arbitrage(iv_surf, strikes, maturities, S, r)
    if but_viol.empty:
        print("  ✓ No butterfly violations found.")
    else:
        print(f"  ✗ {len(but_viol)} violation(s) found:")
        print(but_viol.to_string(index=False))

    # ── Newton-Raphson convergence demo
    print("\n--- Newton-Raphson IV Solver Demo ---")
    true_sigma = 0.25
    C_test     = bs_price(S, 100.0, r, 1.0, true_sigma, "call")
    iv_est     = implied_vol(C_test, S, 100.0, r, 1.0, sigma0=0.10)
    print(f"  True σ = {true_sigma:.4f}")
    print(f"  Solved σ = {iv_est:.6f}  |  Error: {abs(iv_est - true_sigma):.2e}")
