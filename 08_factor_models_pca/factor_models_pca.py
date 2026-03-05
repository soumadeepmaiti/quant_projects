"""
Factor Models, Greeks & PCA-Based Alpha Signals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────

TICKERS = ["SPY", "QQQ", "GLD", "TLT", "XLE", "XLF", "XLV", "EEM", "IWM", "VNQ"]
START   = "2016-01-01"
END     = "2024-12-31"


def fetch_returns(tickers=TICKERS, start=START, end=END) -> pd.DataFrame:
    raw     = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
    prices  = raw["Close"].dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


# ─────────────────────────────────────────────
# 2. PCA FACTOR MODEL
# ─────────────────────────────────────────────

class PCAFactorModel:
    """
    Statistical factor model via PCA on the sample covariance matrix.

    Model:
        r_t = B · f_t + ε_t
        B        — (n_assets × k) factor loading matrix  (eigenvectors)
        f_t      — k-dimensional factor returns (principal component scores)
        ε_t      — idiosyncratic returns  (unexplained by factors)

    Covariance decomposition:
        Σ = B · Λ · B' + D
        Λ = diag(λ_1, ..., λ_k)    (eigenvalues)
        D = diag(σ²_ε,1, ..., σ²_ε,n)  (idiosyncratic variances)

    Alpha signal:
        α_i = mean(ε_{i,t}) / std(ε_{i,t})   (idiosyncratic Sharpe)
    """

    def __init__(self, n_factors: int = 3):
        self.k        = n_factors
        self.pca      = PCA(n_components=n_factors)
        self.scaler   = StandardScaler()
        self.loadings_  = None   # (k × n_assets) — pca.components_
        self.scores_    = None   # (T × k) — principal component time series
        self.resid_     = None   # (T × n_assets) — idiosyncratic returns
        self.explained_ = None
        self.assets_    = None

    def fit(self, returns: pd.DataFrame):
        """Fit PCA factor model."""
        self.assets_ = returns.columns.tolist()
        R = returns.values  # (T × n)

        # Standardise
        R_std = self.scaler.fit_transform(R)

        # PCA
        scores = self.pca.fit_transform(R_std)  # (T × k)

        # Reconstruct factor-explained returns
        R_explained = self.pca.inverse_transform(scores)   # (T × n) in std space
        R_explained_raw = self.scaler.inverse_transform(R_explained)

        # Idiosyncratic = actual - factor-explained
        self.scores_    = pd.DataFrame(scores, index=returns.index,
                                        columns=[f"PC{i+1}" for i in range(self.k)])
        self.loadings_  = pd.DataFrame(self.pca.components_,
                                        index=[f"PC{i+1}" for i in range(self.k)],
                                        columns=self.assets_)
        self.resid_     = pd.DataFrame(R - R_explained_raw,
                                        index=returns.index,
                                        columns=self.assets_)
        self.explained_ = pd.Series(self.pca.explained_variance_ratio_,
                                     index=[f"PC{i+1}" for i in range(self.k)])
        return self

    def factor_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Full covariance matrix decomposition:
            Σ = B'ΛB + D
        where B = loadings (scaled back to return space).
        """
        n      = len(self.assets_)
        scales = self.scaler.scale_
        # Loadings in return space: B_{n×k}
        B      = (self.pca.components_.T * scales).T    # (k × n) → scale back
        Lambda = np.diag(self.pca.explained_variance_)
        factor_cov = B.T @ Lambda @ B

        D      = np.diag(self.resid_.var().values)
        total  = factor_cov + D
        return pd.DataFrame(total, index=self.assets_, columns=self.assets_)

    def alpha_signals(self) -> pd.DataFrame:
        """
        Idiosyncratic Sharpe (annualised) per asset.
        α_i = (252 · mean(ε_i)) / (√252 · std(ε_i))
        """
        mu    = self.resid_.mean() * 252
        sigma = self.resid_.std()  * np.sqrt(252)
        sharpe = mu / sigma
        return pd.DataFrame({
            "Idiosyn Mean (ann)":   mu,
            "Idiosyn Vol (ann)":    sigma,
            "Idiosyn Sharpe":       sharpe,
        }).sort_values("Idiosyn Sharpe", ascending=False)

    def variance_attribution(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Per-asset variance explained by each PC.
        Attr_{i,k} = Cov(r_i, f_k)² / Var(r_i)
        """
        R     = returns.values
        attrs = {}
        for j, pc_name in enumerate(self.scores_.columns):
            f   = self.scores_[pc_name].values
            cov = np.array([np.cov(R[:, i], f)[0, 1] for i in range(R.shape[1])])
            var = R.var(axis=0)
            attrs[pc_name] = cov ** 2 / var
        return pd.DataFrame(attrs, index=self.assets_)


# ─────────────────────────────────────────────
# 3. PORTFOLIO OPTIMISATION ON PC SCORES
# ─────────────────────────────────────────────

class PCPortfolio:
    """
    Sharpe-Ratio-maximising portfolio constructed from PCA factor scores.

    Strategy:
        1. Extract PC factor time series f_t ∈ R^k
        2. Estimate mean μ_f and covariance Σ_f of factor returns
        3. Solve max-Sharpe in factor space:
               max  w'μ_f / √(w'Σ_f w)
               s.t. Σw_i = 1,  w_i ≥ 0 (long-only)
        4. Map factor weights → asset weights:  w_asset = B' · w_factor
           where B = loadings (k × n) in return space
    """

    def __init__(self, factor_model: PCAFactorModel, risk_free: float = 0.04 / 252):
        self.fm = factor_model
        self.rf = risk_free
        self.w_factor_ = None
        self.w_asset_  = None

    def fit(self, factor_scores: pd.DataFrame):
        """Estimate optimal factor weights via Sharpe maximisation."""
        mu    = factor_scores.mean().values
        Sigma = factor_scores.cov().values
        k     = len(mu)

        def neg_sharpe(w):
            ret  = np.dot(w, mu) - self.rf
            vol  = np.sqrt(w @ Sigma @ w)
            return -ret / (vol + 1e-12)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(-1.0, 1.0)] * k   # allow short factor positions
        x0 = np.ones(k) / k
        res = minimize(neg_sharpe, x0, method="SLSQP",
                       constraints=constraints, bounds=bounds)
        self.w_factor_ = res.x
        return self

    def asset_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Project factor weights back to asset weights using loadings.
        w_asset = loadings.T @ w_factor  (in standardised return space)
        Then normalise to sum to 1.
        """
        if self.w_factor_ is None:
            raise RuntimeError("Call fit() first.")
        # loadings: (k × n), w_factor: (k,) → w_raw: (n,)
        w_raw = self.fm.loadings_.values.T @ self.w_factor_
        # Re-scale to account for StandardScaler
        scales = self.fm.scaler.scale_
        w_raw /= scales
        # Normalise
        w_asset = w_raw / (np.abs(w_raw).sum() + 1e-10)
        self.w_asset_ = pd.Series(w_asset, index=self.fm.assets_)
        return self.w_asset_


# ─────────────────────────────────────────────
# 4. BACKTESTING ENGINE
# ─────────────────────────────────────────────

def backtest_pca_portfolio(returns: pd.DataFrame,
                            n_factors: int = 3,
                            n_splits: int = 5,
                            rebal_freq: int = 21) -> dict:
    """
    Walk-forward backtest of PCA Sharpe portfolio.
    Each fold:
        - Train: fit PCA + optimise weights
        - Test:  apply weights to out-of-sample returns

    Returns: cumulative return, Sharpe ratio, max drawdown, weights history.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_returns  = []
    all_dates    = []
    weights_hist = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(returns)):
        train = returns.iloc[train_idx]
        test  = returns.iloc[test_idx]

        # Fit model
        fm = PCAFactorModel(n_factors=n_factors).fit(train)
        pp = PCPortfolio(fm).fit(fm.scores_)
        w  = pp.asset_weights(train)

        # Rebalance every rebal_freq days within test
        port_ret = []
        for start in range(0, len(test), rebal_freq):
            window = test.iloc[start:start + rebal_freq]
            # Refit weights at each rebalance date
            if start > 0:
                fm_rb = PCAFactorModel(n_factors=n_factors).fit(
                    train.append(test.iloc[:start]) if hasattr(train, 'append')
                    else pd.concat([train, test.iloc[:start]])
                )
                pp_rb = PCPortfolio(fm_rb).fit(fm_rb.scores_)
                w     = pp_rb.asset_weights(train)

            daily_ret = window @ w
            port_ret.extend(daily_ret.values.tolist())

        all_returns.extend(port_ret)
        all_dates.extend(test.index.tolist())
        weights_hist.append({"fold": fold + 1, "weights": w})

    port_series = pd.Series(all_returns, index=all_dates[:len(all_returns)])

    # Metrics
    ann_ret = port_series.mean() * 252
    ann_vol = port_series.std() * np.sqrt(252)
    sharpe  = ann_ret / (ann_vol + 1e-10)
    cum_ret = (1 + port_series).cumprod()
    drawdown = cum_ret / cum_ret.cummax() - 1
    max_dd   = drawdown.min()

    return {
        "returns":         port_series,
        "cumulative":      cum_ret,
        "drawdown":        drawdown,
        "ann_return":      ann_ret,
        "ann_vol":         ann_vol,
        "sharpe":          sharpe,
        "max_drawdown":    max_dd,
        "weights_history": weights_hist,
    }


def performance_summary(bt: dict) -> pd.Series:
    return pd.Series({
        "Ann. Return":   f"{bt['ann_return']*100:.2f}%",
        "Ann. Vol":      f"{bt['ann_vol']*100:.2f}%",
        "Sharpe Ratio":  f"{bt['sharpe']:.3f}",
        "Max Drawdown":  f"{bt['max_drawdown']*100:.2f}%",
        "Calmar Ratio":  f"{bt['ann_return'] / max(abs(bt['max_drawdown']), 1e-10):.3f}",
    })


# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────

def plot_pca_factor_analysis(fm: PCAFactorModel, returns: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PCA Factor Model Analysis", fontsize=13, fontweight="bold")

    # Scree plot
    ax = axes[0, 0]
    cum_exp = fm.explained_.cumsum()
    ax.bar(fm.explained_.index, fm.explained_.values * 100,
            color="steelblue", alpha=0.8, label="Individual")
    ax.plot(fm.explained_.index, cum_exp.values * 100,
             "o--r", linewidth=2, label="Cumulative")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Scree Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Factor loadings heatmap
    ax2 = axes[0, 1]
    im = ax2.imshow(fm.loadings_.values, cmap="RdBu_r", aspect="auto",
                     vmin=-1, vmax=1)
    ax2.set_xticks(range(len(fm.assets_)))
    ax2.set_xticklabels(fm.assets_, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(fm.k))
    ax2.set_yticklabels(fm.loadings_.index)
    ax2.set_title("Factor Loadings  (B)")
    plt.colorbar(im, ax=ax2)

    # PC score time series
    ax3 = axes[1, 0]
    for pc in fm.scores_.columns[:3]:
        ax3.plot(fm.scores_.index, fm.scores_[pc].rolling(21).mean(),
                  linewidth=1.5, label=f"{pc} (21d MA)")
    ax3.set_title("PC Score Time Series (rolling 21d MA)")
    ax3.set_ylabel("PC Score")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Variance attribution
    ax4 = axes[1, 1]
    attr = fm.variance_attribution(returns)
    attr.plot(kind="bar", ax=ax4, alpha=0.8)
    ax4.set_title("Variance Attribution by Factor")
    ax4.set_ylabel("R² with each PC")
    ax4.set_xticklabels(attr.index, rotation=45, ha="right", fontsize=8)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pca_factor_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_backtest(bt: dict, benchmark_returns: pd.Series = None):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
    fig.suptitle("PCA Portfolio Backtest — Walk-Forward Cross-Validation",
                  fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(bt["cumulative"].index, bt["cumulative"].values,
             linewidth=2, color="steelblue", label="PCA Portfolio")
    if benchmark_returns is not None:
        bm_cum = (1 + benchmark_returns.reindex(bt["cumulative"].index).fillna(0)).cumprod()
        ax.plot(bm_cum.index, bm_cum.values,
                 linewidth=1.5, color="grey", linestyle="--", label="SPY Benchmark")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Performance (out-of-sample)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(bt["drawdown"].index, bt["drawdown"].values * 100,
                      0, alpha=0.5, color="crimson")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title(f"Drawdown  (Max: {bt['max_drawdown']*100:.1f}%)")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    rolling_sr = (bt["returns"].rolling(63).mean() * 252 /
                  (bt["returns"].rolling(63).std() * np.sqrt(252) + 1e-10))
    ax3.plot(rolling_sr.index, rolling_sr, linewidth=1.5, color="navy")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.axhline(bt["sharpe"], color="red", linestyle="--",
                 label=f"Full-period Sharpe: {bt['sharpe']:.2f}")
    ax3.set_ylabel("Rolling Sharpe (63d)")
    ax3.set_xlabel("Date")
    ax3.set_title("Rolling Sharpe Ratio")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pca_backtest.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_alpha_signals(alpha_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["crimson" if v < 0 else "steelblue"
              for v in alpha_df["Idiosyn Sharpe"]]
    ax.barh(alpha_df.index, alpha_df["Idiosyn Sharpe"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Idiosyncratic Sharpe Ratio (annualised)")
    ax.set_title("PCA Alpha Signals — Idiosyncratic Sharpe per Asset")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("alpha_signals.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT 8 — FACTOR MODELS, GREEKS & PCA ALPHA SIGNALS")
    print("=" * 60)

    # ── Fetch data
    print(f"\nFetching data for: {TICKERS}")
    returns = fetch_returns()
    print(f"  Shape: {returns.shape}  ({returns.index[0].date()} → {returns.index[-1].date()})")

    # ── PCA Factor Model
    print("\n--- PCA Factor Model (k=3) ---")
    fm = PCAFactorModel(n_factors=3).fit(returns)
    print(f"  Explained variance per PC:")
    for pc, ev in fm.explained_.items():
        print(f"    {pc}: {ev*100:.1f}%")
    print(f"  Total explained: {fm.explained_.sum()*100:.1f}%")

    print("\n  Factor Loadings:")
    print(fm.loadings_.round(3).to_string())

    # ── Alpha signals
    print("\n--- Idiosyncratic Alpha Signals ---")
    alpha_df = fm.alpha_signals()
    print(alpha_df.round(4).to_string())
    plot_alpha_signals(alpha_df)

    # ── Variance attribution
    print("\n--- Variance Attribution ---")
    attr = fm.variance_attribution(returns)
    print(attr.round(3).to_string())

    # ── Factor covariance
    print("\n--- Factor Decomposed Covariance (sample vs PCA) ---")
    pca_cov   = fm.factor_covariance(returns)
    true_cov  = returns.cov()
    cov_error = (pca_cov - true_cov).abs().mean().mean()
    print(f"  Mean absolute covariance error (3-factor vs true): {cov_error:.6f}")

    # ── PCA plots
    plot_pca_factor_analysis(fm, returns)

    # ── Portfolio optimisation + backtest
    print("\n--- Walk-Forward Backtest (5-fold, rebal every 21d) ---")
    bt = backtest_pca_portfolio(returns, n_factors=3, n_splits=5, rebal_freq=21)
    summary = performance_summary(bt)
    print(summary.to_string())

    # Benchmark SPY
    spy_ret = returns["SPY"]
    plot_backtest(bt, benchmark_returns=spy_ret)

    print("\nDone. All figures saved.")
