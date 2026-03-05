"""
Kalman Filter & State-Space Time-Series Modelling

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import solve
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


# ─────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────

def fetch_returns(ticker="SPY", start="2015-01-01", end="2024-12-31"):
    prices  = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)["Close"]
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


# ─────────────────────────────────────────────
# 2. KALMAN FILTER — LOCAL LEVEL MODEL FOR LATENT VOLATILITY
# ─────────────────────────────────────────────

class KalmanFilter:
    """
    Linear Gaussian state-space model for tracking latent volatility:

        State equation (transition):
            x_t = F · x_{t-1} + w_t,    w_t ~ N(0, Q)

        Observation equation:
            y_t = H · x_t + v_t,         v_t ~ N(0, R)

    Here we track the latent log-variance:
        x_t = log(σ²_t)  (latent)
        y_t = log(r²_t)  (observed proxy for realised variance)

    The Kalman filter computes optimal linear estimates:
        Predict:    x̂_{t|t-1} = F · x̂_{t-1|t-1}
                    P_{t|t-1}  = F · P_{t-1|t-1} · F' + Q
        Update:     K_t = P_{t|t-1} · H' / (H·P_{t|t-1}·H' + R)   (Kalman gain)
                    x̂_{t|t} = x̂_{t|t-1} + K_t·(y_t - H·x̂_{t|t-1})
                    P_{t|t}  = (I - K_t·H)·P_{t|t-1}
    """

    def __init__(self, F: float = 1.0, H: float = 1.0,
                 Q: float = 0.01, R: float = 1.0, x0: float = -2.0, P0: float = 1.0):
        self.F  = F    # state transition (random walk → F=1)
        self.H  = H    # observation matrix
        self.Q  = Q    # process noise (variance of state)
        self.R  = R    # observation noise (log(r²) noise ≈ π²/2)
        self.x0 = x0   # initial state (log-variance)
        self.P0 = P0   # initial state variance

    def filter(self, y: np.ndarray) -> dict:
        """Run forward Kalman filter. Returns filtered states and variances."""
        n  = len(y)
        x_pred = np.empty(n)
        x_filt = np.empty(n)
        P_pred = np.empty(n)
        P_filt = np.empty(n)
        K_arr  = np.empty(n)
        innov  = np.empty(n)

        x = self.x0
        P = self.P0

        for t in range(n):
            # Predict
            x_p = self.F * x
            P_p = self.F ** 2 * P + self.Q
            x_pred[t] = x_p
            P_pred[t] = P_p

            # Innovation
            y_hat    = self.H * x_p
            S        = self.H ** 2 * P_p + self.R
            K        = P_p * self.H / S
            innov[t] = y[t] - y_hat
            K_arr[t] = K

            # Update
            x = x_p + K * innov[t]
            P = (1 - K * self.H) * P_p
            x_filt[t] = x
            P_filt[t] = P

        return {
            "x_filtered":  x_filt,
            "x_predicted": x_pred,
            "P_filtered":  P_filt,
            "P_predicted": P_pred,
            "innovations": innov,
            "kalman_gain": K_arr,
        }

    def log_likelihood(self, y: np.ndarray) -> float:
        """Prediction error decomposition log-likelihood."""
        n    = len(y)
        x, P = self.x0, self.P0
        ll   = 0.0
        for t in range(n):
            x_p = self.F * x
            P_p = self.F ** 2 * P + self.Q
            S   = self.H ** 2 * P_p + self.R
            innov = y[t] - self.H * x_p
            ll -= 0.5 * (np.log(2 * np.pi * S) + innov ** 2 / S)
            K = P_p * self.H / S
            x = x_p + K * innov
            P = (1 - K * self.H) * P_p
        return ll

    def fit(self, y: np.ndarray):
        """MLE calibration of Q (process noise) and R (observation noise)."""
        def neg_ll(params):
            Q, R = np.exp(params)  # ensure positivity
            self.Q = Q
            self.R = R
            return -self.log_likelihood(y)

        init = [np.log(self.Q), np.log(self.R)]
        res  = minimize(neg_ll, init, method="Nelder-Mead",
                        options={"xatol": 1e-6, "maxiter": 1000})
        self.Q, self.R = np.exp(res.x)
        print(f"  KF calibrated: Q={self.Q:.4f}, R={self.R:.4f}")
        return self


# ─────────────────────────────────────────────
# 3. ARIMA MODEL
# ─────────────────────────────────────────────

def fit_arima(series: pd.Series, order=(1, 0, 1)):
    """
    Fit ARIMA(p,d,q) to absolute returns or squared returns.
    Tests stationarity with ADF before fitting.
    """
    # ADF test
    adf_stat, pvalue, *_ = adfuller(series.dropna())
    print(f"  ADF test: statistic={adf_stat:.3f}, p-value={pvalue:.4f}  "
          f"({'stationary' if pvalue < 0.05 else 'non-stationary'})")

    model  = ARIMA(series, order=order)
    result = model.fit()
    return result


def arima_forecast(arima_result, steps: int) -> np.ndarray:
    """One-step-ahead out-of-sample forecast."""
    fc = arima_result.forecast(steps=steps)
    return fc.values


# ─────────────────────────────────────────────
# 4. GARCH(1,1)
# ─────────────────────────────────────────────

def garch_variance_path(returns: np.ndarray, omega: float,
                         alpha: float, beta: float) -> np.ndarray:
    n   = len(returns)
    var = np.empty(n)
    var[0] = np.var(returns)
    for t in range(1, n):
        var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
    return var


def fit_garch(returns: pd.Series):
    r = returns.values
    def neg_ll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        var = garch_variance_path(r, omega, alpha, beta)
        var = np.maximum(var, 1e-12)
        return float(np.sum(np.log(var) + r ** 2 / var))

    s2  = np.var(r)
    res = minimize(neg_ll, [s2 * 0.05, 0.08, 0.88],
                   method="L-BFGS-B",
                   bounds=[(1e-8, None), (0, 1), (0, 1)])
    return res.x


# ─────────────────────────────────────────────
# 5. REGIME DETECTION
# ─────────────────────────────────────────────

def detect_vol_regimes(kf_result: dict, threshold_std: float = 1.0) -> np.ndarray:
    """
    Classify each date as HIGH/LOW volatility regime based on
    Kalman-filtered log-variance relative to rolling mean ± threshold·std.
    Returns 0 = low, 1 = high.
    """
    x     = kf_result["x_filtered"]
    roll  = pd.Series(x).rolling(63, min_periods=21)
    mu    = roll.mean().fillna(np.mean(x))
    sig   = roll.std().fillna(np.std(x))
    return (x > mu + threshold_std * sig).astype(int)


# ─────────────────────────────────────────────
# 6. FORECAST EVALUATION & HYPOTHESIS TESTING
# ─────────────────────────────────────────────

def evaluate_forecasts(returns: pd.Series,
                        kf_vol: np.ndarray,
                        garch_vol: np.ndarray,
                        horizon: int = 5) -> pd.DataFrame:
    """
    Compare forecast volatility against realised (rolling std) using:
      - RMSE
      - MAE
      - Diebold-Mariano statistic (informal)
    """
    realised = (returns.rolling(horizon).std() * np.sqrt(252)).shift(-horizon).dropna()
    n = len(realised)

    kf_f    = pd.Series(kf_vol[:n], index=returns.index[:n])
    garch_f = pd.Series(garch_vol[:n], index=returns.index[:n])

    aligned = pd.DataFrame({
        "Realised": realised,
        "KF":       kf_f,
        "GARCH":    garch_f,
    }).dropna()

    results = {}
    for name in ["KF", "GARCH"]:
        err   = aligned[name] - aligned["Realised"]
        rmse  = np.sqrt(np.mean(err ** 2))
        mae   = np.mean(np.abs(err))
        r2    = 1 - np.var(err) / np.var(aligned["Realised"])
        results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

    return pd.DataFrame(results).T


def cross_validate_garch(returns: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Walk-forward CV for GARCH(1,1)."""
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    r      = returns.values

    for train_idx, test_idx in tscv.split(r):
        train, test = r[train_idx], r[test_idx]
        omega, alpha, beta = fit_garch(pd.Series(train))
        var_test = garch_variance_path(
            np.concatenate([train, test]), omega, alpha, beta)[len(train):]
        vol_test  = np.sqrt(var_test * 252)
        realised  = pd.Series(test).rolling(5).std().dropna() * np.sqrt(252)
        min_len   = min(len(vol_test), len(realised))
        rmse = np.sqrt(mean_squared_error(realised[:min_len],
                                          vol_test[:min_len]))
        scores.append({"fold_rmse": rmse, "omega": omega,
                       "alpha": alpha, "beta": beta,
                       "persistence": alpha + beta})
    return pd.DataFrame(scores)


# ─────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────

def plot_kalman_vs_garch(returns, kf_result, garch_vol, regimes, ticker="SPY"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f"Kalman Filter vs GARCH — Volatility Regime Tracking ({ticker})",
                  fontsize=13, fontweight="bold")
    t = returns.index

    # Log variance
    ax = axes[0]
    y_obs = np.log(returns.values ** 2 + 1e-8)
    ax.plot(t, y_obs, alpha=0.3, linewidth=0.5, color="grey", label="log(r²) observed")
    ax.plot(t, kf_result["x_filtered"], linewidth=1.5, color="steelblue",
             label="KF filtered log(σ²)")
    ax.set_ylabel("log(σ²)")
    ax.set_title("Latent Log-Variance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annualised volatility
    kf_vol  = np.sqrt(np.exp(kf_result["x_filtered"]) * 252)
    garch_a = np.sqrt(garch_vol * 252)
    ax2 = axes[1]
    ax2.plot(t, kf_vol * 100,   linewidth=1.5, color="steelblue", label="KF volatility")
    ax2.plot(t, garch_a * 100,  linewidth=1.5, color="crimson", linestyle="--",
              label="GARCH(1,1)")
    ax2.set_ylabel("Ann. Vol (%)")
    ax2.set_title("Volatility Forecast Comparison")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Regimes
    ax3 = axes[2]
    ax3.fill_between(t, 0, regimes, step="post", alpha=0.4, color="orange",
                      label="High-vol regime")
    ax3.plot(t, (returns.abs() * np.sqrt(252)).rolling(21).mean() * 100,
              linewidth=1, color="navy", alpha=0.8, label="21d Realised Vol (%)")
    ax3.set_ylabel("Regime / Vol (%)")
    ax3.set_xlabel("Date")
    ax3.set_title("Detected Volatility Regimes")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kalman_vol_regimes.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT 6 — KALMAN FILTER & STATE-SPACE MODELLING")
    print("=" * 60)

    returns = fetch_returns("SPY")
    r       = returns.values

    # ── Kalman Filter
    print("\n--- Kalman Filter Calibration ---")
    y_kf = np.log(r ** 2 + 1e-8)  # observation: log squared returns
    kf   = KalmanFilter(F=1.0, H=1.0, Q=0.05, R=np.pi ** 2 / 2, x0=-5.0, P0=1.0)
    kf.fit(y_kf)
    kf_result = kf.filter(y_kf)
    kf_vol    = np.sqrt(np.exp(kf_result["x_filtered"]))   # daily vol
    print(f"  KF vol range: [{kf_vol.min()*100:.1f}%, {kf_vol.max()*100:.1f}%] (daily)")

    # ── GARCH(1,1)
    print("\n--- GARCH(1,1) Estimation ---")
    omega, alpha, beta = fit_garch(returns)
    print(f"  ω={omega:.6f}, α={alpha:.4f}, β={beta:.4f}, α+β={alpha+beta:.4f}")
    garch_var = garch_variance_path(r, omega, alpha, beta)

    # ── ARIMA on absolute returns
    print("\n--- ARIMA(1,0,1) on |returns| ---")
    abs_ret     = returns.abs()
    arima_result = fit_arima(abs_ret, order=(1, 0, 1))
    print(arima_result.summary().tables[1])

    # ── Regime detection
    regimes = detect_vol_regimes(kf_result, threshold_std=0.8)
    pct_high = regimes.mean() * 100
    print(f"\nRegime detection: {pct_high:.1f}% of days in high-vol regime")

    # ── Forecast evaluation
    print("\n--- Forecast Evaluation ---")
    eval_df = evaluate_forecasts(returns, kf_vol * np.sqrt(252), np.sqrt(garch_var * 252))
    print(eval_df)

    # ── GARCH Cross-validation
    print("\n--- GARCH Walk-Forward CV ---")
    cv_df = cross_validate_garch(returns)
    print(cv_df.to_string(index=False))
    print(f"  Mean RMSE: {cv_df['fold_rmse'].mean():.5f}")
    print(f"  Mean persistence: {cv_df['persistence'].mean():.4f}")

    # ── Plots
    plot_kalman_vs_garch(returns, kf_result, garch_var, regimes)
