"""
Volatility Forecasting — Historical, EWMA & GARCH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────

TICKERS  = ["SPY", "QQQ", "GLD", "TLT", "XLE"]
START    = "2018-01-01"
END      = "2024-12-31"
ANNUALISE = 252


def fetch_prices(tickers=TICKERS, start=START, end=END) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna()
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


# ─────────────────────────────────────────────
# 2. HISTORICAL VOLATILITY (ROLLING)
# ─────────────────────────────────────────────

def historical_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Rolling historical volatility (annualised).
    σ̂_hist(t) = sqrt( 252 / (n-1)  ·  Σ (r_{t-i} - r̄)² )
    """
    return (
        returns
        .rolling(window=window, min_periods=window // 2)
        .std(ddof=1)
        .mul(np.sqrt(ANNUALISE))
        .rename(f"HV_{window}d")
    )


# ─────────────────────────────────────────────
# 3. EWMA VOLATILITY
# ─────────────────────────────────────────────

def ewma_variance(returns: np.ndarray, lam: float) -> np.ndarray:
    """
    Recursive EWMA variance:
        σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t
    """
    n = len(returns)
    var = np.empty(n)
    var[0] = returns[0] ** 2
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1 - lam) * returns[t] ** 2
    return var


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Annualised EWMA volatility series."""
    v = ewma_variance(returns.values, lam)
    return pd.Series(np.sqrt(v * ANNUALISE), index=returns.index, name=f"EWMA_λ={lam:.2f}")


def calibrate_lambda(returns: pd.Series, realised_window: int = 5) -> float:
    """
    Calibrate λ by minimising RMSE between EWMA forecast and
    realised variance over a rolling out-of-sample window.
    """
    realised = (
        returns.rolling(realised_window).var(ddof=1).shift(-realised_window) * ANNUALISE
    ).dropna()

    def rmse(params):
        lam = params[0]
        if not (0.80 < lam < 0.999):
            return 1e10
        vol = ewma_volatility(returns, lam)
        var_f = vol ** 2
        aligned = pd.concat([var_f, realised], axis=1).dropna()
        return np.sqrt(np.mean((aligned.iloc[:, 0] - aligned.iloc[:, 1]) ** 2))

    result = minimize(rmse, x0=[0.94], method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 500})
    return float(result.x[0])


# ─────────────────────────────────────────────
# 4. GARCH(1,1)
# ─────────────────────────────────────────────

def garch_variance(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """
    GARCH(1,1) variance path:
        σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
    """
    n = len(returns)
    var = np.empty(n)
    var[0] = np.var(returns)
    for t in range(1, n):
        var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
    return var


def fit_garch(returns: pd.Series):
    """
    Maximum-likelihood estimation for GARCH(1,1) parameters (ω, α, β).
    Minimise negative log-likelihood:
        L = Σ [ log(σ²_t) + r²_t / σ²_t ]
    """
    r = returns.values

    def neg_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        var = garch_variance(r, omega, alpha, beta)
        var = np.maximum(var, 1e-12)
        return float(np.sum(np.log(var) + r ** 2 / var))

    # Initial guess: low persistence
    s2 = np.var(r)
    x0 = [s2 * 0.05, 0.08, 0.88]
    bounds = [(1e-8, None), (0, 1), (0, 1)]
    res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)
    omega, alpha, beta = res.x
    return omega, alpha, beta


def garch_volatility(returns: pd.Series) -> pd.Series:
    """Fit GARCH(1,1) and return annualised volatility series."""
    omega, alpha, beta = fit_garch(returns)
    print(f"  GARCH(1,1) fitted: ω={omega:.6f}, α={alpha:.4f}, β={beta:.4f}, "
          f"persistence α+β={alpha+beta:.4f}")
    v = garch_variance(returns.values, omega, alpha, beta)
    return pd.Series(np.sqrt(v * ANNUALISE), index=returns.index, name="GARCH(1,1)")


# ─────────────────────────────────────────────
# 5. CROSS-VALIDATION COMPARISON
# ─────────────────────────────────────────────

def rolling_cv_rmse(returns: pd.Series, n_splits: int = 5, window: int = 21) -> dict:
    """
    Walk-forward cross-validation.
    Splits the series into n_splits folds; for each fold trains on the
    preceding data and evaluates RMSE against realised variance.
    """
    n = len(returns)
    fold_size = n // (n_splits + 1)
    results = {"HV": [], "EWMA": [], "GARCH": []}

    for fold in range(1, n_splits + 1):
        train_end = fold * fold_size
        test_end  = train_end + fold_size
        if test_end > n:
            break

        train = returns.iloc[:train_end]
        test  = returns.iloc[train_end:test_end]

        realised_var = test.rolling(window).var(ddof=1).dropna() * ANNUALISE

        # HV
        hv = historical_volatility(pd.concat([train, test]), window).iloc[train_end:]
        hv_var = (hv ** 2).reindex(realised_var.index).dropna()

        # EWMA
        lam = calibrate_lambda(train)
        ew = ewma_volatility(pd.concat([train, test]), lam).iloc[train_end:]
        ew_var = (ew ** 2).reindex(realised_var.index).dropna()

        # GARCH
        omega, alpha, beta = fit_garch(train)
        gv = garch_variance(pd.concat([train, test]).values, omega, alpha, beta)
        garch_var = pd.Series(gv * ANNUALISE,
                               index=pd.concat([train, test]).index).iloc[train_end:]
        garch_var = garch_var.reindex(realised_var.index).dropna()

        def rmse(f, r): return np.sqrt(np.mean((f.values - r.values) ** 2))

        rv = realised_var.reindex(hv_var.index).dropna()
        results["HV"].append(rmse(hv_var.reindex(rv.index).dropna(), rv))
        rv = realised_var.reindex(ew_var.index).dropna()
        results["EWMA"].append(rmse(ew_var.reindex(rv.index).dropna(), rv))
        rv = realised_var.reindex(garch_var.index).dropna()
        results["GARCH"].append(rmse(garch_var.reindex(rv.index).dropna(), rv))

    return {k: np.mean(v) for k, v in results.items() if v}


# ─────────────────────────────────────────────
# 6. PCA ON MULTI-ASSET COVARIANCE MATRIX
# ─────────────────────────────────────────────

def pca_risk_factors(returns: pd.DataFrame, n_components: int = 3):
    """
    Apply PCA to multi-asset return matrix.
    Returns:
        explained_var  — fraction of variance per component
        loadings       — eigenvectors (components × assets)
        scores         — principal component time series
    """
    scaler   = StandardScaler()
    scaled   = scaler.fit_transform(returns.dropna())
    pca      = PCA(n_components=n_components)
    scores   = pca.fit_transform(scaled)

    loadings = pd.DataFrame(
        pca.components_,
        columns=returns.columns,
        index=[f"PC{i+1}" for i in range(n_components)]
    )
    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(n_components)]
    )
    score_df = pd.DataFrame(
        scores,
        index=returns.dropna().index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return explained, loadings, score_df


# ─────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────

def plot_volatility_comparison(returns: pd.Series, ticker: str = "SPY", lam: float = 0.94):
    hv21  = historical_volatility(returns, 21)
    hv63  = historical_volatility(returns, 63)
    ewma  = ewma_volatility(returns, lam)
    garch = garch_volatility(returns)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Volatility Forecasting — {ticker}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(hv21,  label="HV 21d",        alpha=0.8, linewidth=1.2)
    ax.plot(hv63,  label="HV 63d",        alpha=0.8, linewidth=1.2, linestyle="--")
    ax.plot(ewma,  label=f"EWMA λ={lam:.2f}", linewidth=1.5, color="crimson")
    ax.plot(garch, label="GARCH(1,1)",    linewidth=1.5, color="navy", linestyle="-.")
    ax.set_ylabel("Annualised Volatility")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Volatility Model Comparison")

    ax2 = axes[1]
    ax2.plot(returns.rolling(21).std() * np.sqrt(252), color="grey", alpha=0.6, label="Realised 21d")
    ax2.set_ylabel("Realised Vol")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("volatility_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: volatility_comparison.png")


def plot_pca(explained, loadings, scores, tickers):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PCA Risk Factor Decomposition", fontsize=13, fontweight="bold")

    axes[0].bar(explained.index, explained.values * 100, color="steelblue")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Scree Plot")
    axes[0].grid(True, alpha=0.3)

    im = axes[1].imshow(loadings.values, cmap="RdBu_r", aspect="auto",
                         vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(tickers)))
    axes[1].set_xticklabels(tickers, rotation=45, ha="right")
    axes[1].set_yticks(range(len(loadings)))
    axes[1].set_yticklabels(loadings.index)
    axes[1].set_title("Factor Loadings")
    plt.colorbar(im, ax=axes[1])

    for i, pc in enumerate(scores.columns[:2]):
        axes[2].plot(scores[pc], label=pc, alpha=0.8, linewidth=0.9)
    axes[2].set_title("PC1 & PC2 Time Series")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pca_risk_factors.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: pca_risk_factors.png")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT 1 — VOLATILITY FORECASTING")
    print("=" * 60)

    prices  = fetch_prices()
    returns = log_returns(prices)
    spy_ret = returns["SPY"]

    # Calibrate λ
    lam_opt = calibrate_lambda(spy_ret)
    print(f"\nOptimal EWMA λ for SPY: {lam_opt:.4f}")

    # Cross-validation
    print("\nWalk-forward cross-validation RMSE (annualised variance):")
    cv = rolling_cv_rmse(spy_ret)
    for model, rmse_val in cv.items():
        print(f"  {model:<10}: {rmse_val:.6f}")

    # Plots
    plot_volatility_comparison(spy_ret, ticker="SPY", lam=lam_opt)

    # PCA
    explained, loadings, scores = pca_risk_factors(returns, n_components=3)
    print(f"\nPCA explained variance:\n{explained.to_string()}")
    print(f"\nFactor loadings:\n{loadings.to_string()}")
    plot_pca(explained, loadings, scores, returns.columns.tolist())
