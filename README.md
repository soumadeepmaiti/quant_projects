# Quantitative Finance Projects

**Soumadeep Maiti** — Physics MSc, LMU Munich  
`soumadeepmaiti21@gmail.com` 

---

A portfolio of eight production-quality quantitative finance projects spanning derivatives pricing, volatility modelling, stochastic simulation, and factor-based portfolio construction. Each project is self-contained with full mathematical derivations, clean Python implementations, and validated results. The work covers every project listed across both the **Quant Analyst** and **Quant Researcher** CVs.

---

## Repository Structure

```
quant_projects/
├── 01_volatility_forecasting/       # HV, EWMA, GARCH(1,1), PCA risk factors
├── 02_monte_carlo_options/          # GBM MC · European · Asian · Barrier · Exchange
├── 03_greeks_hedging/               # Δ Γ V Θ ρ · Delta-Gamma hedging · PnL attribution
├── 04_implied_vol_surface/          # Newton–Raphson IV · Vol surface · Arbitrage detection
├── 05_sde_euler_maruyama/           # EM · OU · GBM · CIR · Heston · Convergence analysis
├── 06_kalman_filter/                # Kalman filter · ARIMA · GARCH · Regime detection
├── 07_multi_asset_mc/               # Correlated GBM · Cholesky · Variance reduction · Margrabe
├── 08_factor_models_pca/            # PCA factor model · Alpha signals · Portfolio backtest
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/soumadeepmaiti/quant_projects.git
cd quant_projects
pip install -r requirements.txt
python 01_volatility_forecasting/volatility_forecasting.py
```

---

## Project 1 — Volatility Forecasting: HV, EWMA & GARCH

**CV:** Quant Analyst · `Python · NumPy · Pandas · Time-Series`

### What it does
Implements and compares three volatility estimators on live equity data (SPY, QQQ, GLD, TLT, XLE), then applies PCA to extract principal risk factors from a multi-asset covariance matrix.

### Mathematics

**Historical Volatility** (rolling window of $n$ days):

$$\hat{\sigma}_{\text{HV},t} = \sqrt{\frac{252}{n-1} \sum_{i=0}^{n-1} \left(r_{t-i} - \bar{r}\right)^2}$$

**EWMA Volatility** (RiskMetrics):

$$\hat{\sigma}^2_t = \lambda\,\hat{\sigma}^2_{t-1} + (1-\lambda)\,r_t^2, \qquad \lambda \in (0,1)$$

The optimal decay parameter $\lambda^*$ is found by minimising RMSE against realised variance:

$$\lambda^* = \arg\min_\lambda \sqrt{\frac{1}{T} \sum_t \left(\hat{\sigma}^2_{\text{EWMA},t}(\lambda) - \hat{\sigma}^2_{\text{realised},t}\right)^2}$$

**GARCH(1,1)** — MLE via negative log-likelihood:

$$\hat{\sigma}^2_t = \omega + \alpha\,r_{t-1}^2 + \beta\,\hat{\sigma}^2_{t-1}$$

$$\mathcal{L}(\omega,\alpha,\beta) = -\sum_t \left[\ln\hat{\sigma}^2_t + \frac{r_t^2}{\hat{\sigma}^2_t}\right]$$

Stationarity condition: $\alpha + \beta < 1$ (persistence < 1).

**PCA Covariance Decomposition:**

$$\Sigma = V \Lambda V^\top, \qquad \Lambda = \text{diag}(\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n)$$

PC scores: $F_t = R_t V_{[:,1:k]}$ — the dominant $k$ risk factors.

### Key results
- GARCH(1,1) outperforms HV and EWMA in walk-forward RMSE on SPY
- PC1 captures the broad market factor (~55% variance); PC2 separates equities from bonds; PC3 captures commodity exposure
- Optimal EWMA $\lambda \approx 0.94$ (consistent with RiskMetrics standard)

---

## Project 2 — Monte Carlo Option Pricing & Variance Reduction

**CV:** Quant Analyst & Quant Researcher · `Python · NumPy · GBM · Probability & Statistics`

### What it does
Risk-neutral MC pricing of European, Asian (arithmetic), barrier (down-and-out), and exchange options under GBM. Implements antithetic variates, control variates, and verifies $O(1/\sqrt{N})$ convergence.

### Mathematics

**GBM exact simulation** (via Itô's lemma):

$$S_T = S_0 \exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\,Z\right], \qquad Z \sim \mathcal{N}(0,1)$$

**European call price** (risk-neutral expectation):

$$C_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[\max(S_T - K,\,0)\right]$$

**MC standard error and convergence:**

$$\text{SE} = \frac{\hat{\sigma}_g}{\sqrt{N}}, \qquad \text{error} = O\!\left(N^{-1/2}\right)$$

**Antithetic Variates** — uses paired draws $Z$ and $-Z$:

$$\hat{C}_{\text{AV}} = \frac{1}{2}\left[f(Z) + f(-Z)\right], \qquad \text{Var}_{\text{AV}} = \frac{\text{Var}(f)\cdot(1+\rho)}{2}$$

Since $\rho = \text{Corr}(f(Z), f(-Z)) < 0$ for monotone payoffs, $\text{Var}_{\text{AV}} < \text{Var}_{\text{naive}}$.

**Control Variate** — use $\mathbb{E}[S_T] = S_0 e^{rT}$:

$$\hat{C}_{\text{CV}} = \hat{C}_{\text{MC}} - \hat{\beta}\!\left(S_T - S_0 e^{rT}\right), \qquad \hat{\beta} = \frac{\widehat{\text{Cov}}(C, S_T)}{\widehat{\text{Var}}(S_T)}$$

**Asian option** (arithmetic average, no closed form):

$$C_{\text{Asian}} = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[\max\!\left(\frac{1}{M}\sum_{k=1}^M S_{t_k} - K,\;0\right)\right]$$

**Down-and-out barrier with Brownian bridge correction:**

$$P\!\left(\text{cross } B \mid S_{t_k}, S_{t_{k+1}} > B\right) = \exp\!\left(-\frac{2\ln(S_{t_k}/B)\ln(S_{t_{k+1}}/B)}{\sigma^2\Delta t}\right)$$

### Key results
- Sub-0.1% error vs Black-Scholes at $N = 200{,}000$ paths
- ~12× variance reduction with combined antithetic + control variate
- Brownian bridge correction eliminates $O(\sqrt{\Delta t})$ discretisation bias in barrier prices
- Empirical convergence exponent: $\approx -0.50$ (consistent with theory)

---

## Project 3 — Greeks Computation & Delta-Gamma Hedging

**CV:** Quant Analyst & Quant Researcher · `Python · Finite Differences · PnL Attribution`

### What it does
Computes all five Black-Scholes Greeks numerically via central finite differences, validates against closed-form values (error $< 10^{-5}$), then simulates a discrete delta-hedging strategy and decomposes realised PnL into delta, gamma, and theta components.

### Mathematics

**Central difference Greeks** — $O(h^2)$ accuracy:

$$\Delta \approx \frac{V(S+h) - V(S-h)}{2h}, \qquad \Gamma \approx \frac{V(S+h) - 2V(S) + V(S-h)}{h^2}$$

Error balance between truncation $O(h^2)$ and rounding $O(\varepsilon_{\text{mach}}/h)$ gives optimal step:

$$h^* \approx \varepsilon_{\text{mach}}^{1/3} \approx 10^{-5} \quad (\text{double precision})$$

**Closed-form Black-Scholes Greeks:**

$$\Delta = \mathcal{N}(d_1), \quad \Gamma = \frac{\mathcal{N}'(d_1)}{S\sigma\sqrt{T}}, \quad \mathcal{V} = S\sqrt{T}\,\mathcal{N}'(d_1), \quad \Theta = -\frac{S\sigma\,\mathcal{N}'(d_1)}{2\sqrt{T}} - rKe^{-rT}\mathcal{N}(d_2)$$

**Delta-hedging PnL decomposition** (Taylor expansion of $V$):

$$\text{PnL}_t \approx \underbrace{\Delta\,\Delta S}_{\text{delta}} + \underbrace{\frac{1}{2}\Gamma(\Delta S)^2}_{\text{gamma}} + \underbrace{\Theta\,\Delta t}_{\text{theta}} + O\!\left((\Delta S)^3\right)$$

For a delta-neutral portfolio, residual PnL $\approx \frac{1}{2}\Gamma(\Delta S)^2 + \Theta\,\Delta t$ — the gamma-theta trade-off.

**Hedging error vs rebalancing frequency** — theory:

$$\text{Std}\!\left[\text{PnL}\right] \propto \sigma\sqrt{\Delta t} = \sigma\sqrt{T/M}$$

More frequent rebalancing ($M \uparrow$) reduces hedge error at cost $O(M)$ in transaction costs.

### Key results
- All five Greeks match Black-Scholes analytic within $10^{-6}$
- Optimal finite-difference step $h \approx 10^{-5}$ (confirmed empirically)
- Gamma P&L dominates residual for high-$\Gamma$ (ATM) options; theta offsets it in calm periods
- Hedge error decreases as $\sqrt{1/\text{rebalancing frequency}}$

---

## Project 4 — Implied Volatility Surface & Arbitrage Detection

**CV:** Quant Analyst · `Python · Newton–Raphson · Yield Curves · Bootstrapping`

### What it does
Inverts the Black-Scholes formula for implied volatility using Newton–Raphson (with bisection fallback), constructs a full IV surface across strikes and maturities, bootstraps the discount curve from par rates, and flags no-arbitrage violations.

### Mathematics

**Newton–Raphson IV solver:**

$$\sigma_{n+1} = \sigma_n - \frac{C_{\text{BS}}(\sigma_n) - C_{\text{mkt}}}{\mathcal{V}(\sigma_n)}, \qquad \mathcal{V} = \frac{\partial C_{\text{BS}}}{\partial \sigma} = S\sqrt{T}\,\mathcal{N}'(d_1)$$

Convergence is quadratic near the root (ATM where $\mathcal{V}$ is large). Falls back to bisection when $\mathcal{V} \to 0$ (deep ITM/OTM).

**Yield curve bootstrapping** (par bond method):

$$P(T_n) = \frac{1 - c_n \sum_{k<n} P(T_k)\,\Delta t_k}{1 + c_n\,\Delta t_n}, \qquad r_0(T_n) = -\frac{\ln P(T_n)}{T_n}$$

**No-arbitrage conditions on the IV surface:**

*Calendar spread* — total variance must be non-decreasing in maturity:

$$w(K, T) = \sigma_{\text{IV}}^2(K,T)\cdot T \quad \text{non-decreasing in } T \;\Leftrightarrow\; \frac{\partial w}{\partial T} \geq 0$$

*Butterfly arbitrage* — call prices must be convex in strike (risk-neutral density $\geq 0$):

$$\frac{\partial^2 C}{\partial K^2} \geq 0 \quad \Leftrightarrow \quad C(K-h) - 2C(K) + C(K+h) \geq 0$$

*Put-call parity*:

$$C - P = S - Ke^{-rT}$$

### Key results
- Newton–Raphson converges in 3–5 iterations for ATM options; bisection fallback handles extreme strikes robustly
- IV surface exhibits realistic skew (OTM puts more expensive) and term structure
- Arbitrage detector flags calendar violations when IV surface is imported from noisy market data

---

## Project 5 — SDE Simulation: Euler–Maruyama & Convergence Analysis

**CV:** Quant Researcher · `Python · NumPy · Stochastic Calculus · Itô's Lemma`

### What it does
Implements a general Euler–Maruyama solver for arbitrary SDEs, applies it to GBM, Ornstein–Uhlenbeck, CIR, and Heston variance processes, then rigorously verifies strong and weak convergence orders against exact analytical solutions derived from Itô's lemma.

### Mathematics

**General SDE** (Itô form):

$$dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dW_t$$

**Euler–Maruyama discretisation:**

$$X_{t_{n+1}} = X_{t_n} + \mu(X_{t_n}, t_n)\,\Delta t + \sigma(X_{t_n}, t_n)\,\sqrt{\Delta t}\,Z_n, \qquad Z_n \stackrel{\text{iid}}{\sim} \mathcal{N}(0,1)$$

**Convergence orders:**

| Type | Definition | Rate |
|------|-----------|------|
| Strong | $\mathbb{E}\!\left[\lvert X_T^{\text{EM}} - X_T^{\text{exact}}\rvert\right] \leq C\,\Delta t^{1/2}$ | $O(\sqrt{\Delta t})$ |
| Weak | $\left\lvert\mathbb{E}[g(X_T^{\text{EM}})] - \mathbb{E}[g(X_T^{\text{exact}})]\right\rvert \leq C\,\Delta t$ | $O(\Delta t)$ |

**GBM exact solution** (Itô's lemma applied to $\ln S_t$):

$$S_T = S_0\exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma W_T\right]$$

**Ornstein–Uhlenbeck** (mean-reverting):

$$dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t$$

Exact terminal distribution: $X_T \sim \mathcal{N}(\mu_T, \sigma^2_T)$ where

$$\mu_T = X_0 e^{-\kappa T} + \theta\!\left(1 - e^{-\kappa T}\right), \qquad \sigma^2_T = \frac{\sigma^2\!\left(1 - e^{-2\kappa T}\right)}{2\kappa}$$

**Cox–Ingersoll–Ross** (positive short-rate model):

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t, \qquad \text{Feller condition: } 2\kappa\theta \geq \sigma^2$$

**Brownian motion properties** verified empirically:
- Markov property: $W_t \sim \mathcal{N}(0, t)$
- Martingale: $\mathbb{E}[W_t \mid \mathcal{F}_s] = W_s$
- Quadratic variation: $[W]_T = T$ a.s.

### Key results
- Strong convergence exponent: $\approx 0.50$ (matches theory)
- Weak convergence exponent: $\approx 1.00$ (matches theory)
- OU terminal distribution matches exact $\mathcal{N}(\mu_T, \sigma^2_T)$ for all step sizes tested
- CIR Feller condition enforced via reflection; Heston variance kept non-negative

---

## Project 6 — Kalman Filter & State-Space Time-Series Modelling

**CV:** Quant Researcher · `Python · Kalman Filter · ARIMA · GARCH · Hypothesis Testing`

### What it does
Builds a Kalman Filter state-space model to track latent log-variance in equity returns, comparing the filtered estimates against ARIMA and GARCH(1,1) forecasts. Calibrates all models via MLE/RMSE and uses walk-forward CV to benchmark forecast accuracy. Detects high/low volatility regimes.

### Mathematics

**State-space model** for latent log-variance $x_t = \ln\sigma^2_t$:

$$\underbrace{x_t = F\,x_{t-1} + w_t}_{\text{state/transition}}, \quad w_t \sim \mathcal{N}(0, Q)$$

$$\underbrace{y_t = H\,x_t + v_t}_{\text{observation}}, \quad v_t \sim \mathcal{N}(0, R)$$

Observation proxy: $y_t = \ln r_t^2$, where $\mathbb{E}[\ln r_t^2] \approx \ln\sigma^2_t - 1.27$ and $R \approx \pi^2/2$ (log-chi-squared noise).

**Kalman filter recursion:**

*Predict:*

$$\hat{x}_{t|t-1} = F\,\hat{x}_{t-1|t-1}, \qquad P_{t|t-1} = F^2 P_{t-1|t-1} + Q$$

*Kalman gain:*

$$K_t = \frac{P_{t|t-1}\,H}{H^2 P_{t|t-1} + R}$$

*Update:*

$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t\!\left(y_t - H\,\hat{x}_{t|t-1}\right), \qquad P_{t|t} = (1 - K_t H)\,P_{t|t-1}$$

**MLE calibration** via prediction error decomposition:

$$\ln\mathcal{L}(Q, R) = -\frac{1}{2}\sum_t \left[\ln(2\pi S_t) + \frac{\nu_t^2}{S_t}\right], \qquad S_t = H^2 P_{t|t-1} + R, \quad \nu_t = y_t - H\hat{x}_{t|t-1}$$

**GARCH(1,1):**

$$\hat{\sigma}^2_t = \omega + \alpha r_{t-1}^2 + \beta\hat{\sigma}^2_{t-1}$$

Persistence $= \alpha + \beta$ (typically $\approx 0.97$ for equity indices); long-run variance $= \omega/(1-\alpha-\beta)$.

**Regime detection** — classify day $t$ as high-volatility if:

$$\hat{x}_t > \hat{\mu}_{63d}(x) + 0.8\,\hat{\sigma}_{63d}(x)$$

### Key results
- Kalman Filter adapts faster than GARCH to regime changes (lower Kalman gain lag)
- GARCH achieves lower RMSE on quiet periods; KF is superior around crisis episodes
- ADF tests confirm log-squared returns are stationary (suitable for ARIMA modelling)
- Walk-forward CV: 5-fold GARCH persistence consistently $0.96$–$0.98$ (near-IGARCH)

---

## Project 7 — Multi-Asset Monte Carlo: Variance Reduction & Correlated Pricing

**CV:** Quant Researcher · `Python · NumPy · Cholesky · Brownian Motion`

### What it does
Prices European, Asian, barrier, basket, and exchange options under correlated multi-asset GBM. Implements Cholesky decomposition for correlation structure, and benchmarks four variance reduction methods. Validates exchange option MC against Margrabe's closed-form.

### Mathematics

**Correlated GBM via Cholesky decomposition:**

Given correlation matrix $\rho$ and volatility vector $\boldsymbol{\sigma}$, the covariance matrix is $\Sigma_{ij} = \sigma_i \rho_{ij} \sigma_j$. Cholesky factorisation gives $\Sigma = L L^\top$, so:

$$\mathbf{Z}_{\text{corr}} = L\,\boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \stackrel{\text{iid}}{\sim} \mathcal{N}(\mathbf{0}, I)$$

For two assets ($\rho_{12} = \rho$):

$$Z_1 = \sigma_1\,\varepsilon_1, \qquad Z_2 = \rho\sigma_2\,\varepsilon_1 + \sigma_2\sqrt{1-\rho^2}\,\varepsilon_2$$

**Margrabe's formula** — exchange option $\max(S_1(T) - S_2(T), 0)$:

$$C = S_1\,\mathcal{N}(d_1) - S_2\,\mathcal{N}(d_2)$$

$$d_1 = \frac{\ln(S_1/S_2) + \frac{1}{2}\sigma_{\text{sp}}^2 T}{\sigma_{\text{sp}}\sqrt{T}}, \qquad \sigma_{\text{sp}} = \sqrt{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}$$

**Variance reduction summary:**

| Method | Variance Reduction | Mechanism |
|--------|-------------------|-----------|
| Antithetic Variates | $\approx 2\times$ | Negative correlation via $-Z$ |
| Control Variate | $\approx 5$–$15\times$ | Known $\mathbb{E}[S_T] = S_0 e^{rT}$ |
| Stratified Sampling | $\approx 3$–$8\times$ | Even coverage of $\mathcal{N}(0,1)$ quantiles |

**Stratified sampling** — partitions $[0,1]$ into $n$ strata and samples uniformly within each:

$$U_k \sim \text{Uniform}\!\left(\frac{k-1}{n}, \frac{k}{n}\right), \quad k=1,\ldots,n, \qquad Z_k = \Phi^{-1}(U_k)$$

Variance reduction is guaranteed: $\text{Var}_{\text{strat}} \leq \text{Var}_{\text{naive}}$ by ANOVA decomposition.

### Key results
- Exchange option MC vs Margrabe: absolute error $< 0.01$ at $N = 200{,}000$
- Combined antithetic + control variate: $\approx 12\times$ variance reduction
- Empirical convergence rate: $-0.50$ (consistent with $O(N^{-1/2})$ theory)
- Spread volatility $\sigma_{\text{sp}}$ minimised at $\rho = 1$ (perfectly correlated assets cannot be exchanged profitably)

---

## Project 8 — Factor Models, Greeks & PCA-Based Alpha Signals

**CV:** Quant Analyst & Quant Researcher · `Python · PCA · Finite Differences · Backtesting`

### What it does
Builds a statistical factor model via PCA on a 10-asset covariance matrix, extracts factor loadings and idiosyncratic returns, identifies alpha signals (idiosyncratic Sharpe), and back-tests a Sharpe-ratio-optimised portfolio constructed along dominant eigenvectors — with walk-forward cross-validation.

### Mathematics

**PCA factor model:**

$$r_t = B\,f_t + \varepsilon_t$$

where $B$ is the $(n \times k)$ loading matrix (scaled eigenvectors), $f_t$ are $k$ factor scores, and $\varepsilon_t$ are idiosyncratic residuals.

**Covariance decomposition:**

$$\Sigma = B\,\Lambda\,B^\top + D, \qquad \Lambda = \text{diag}(\lambda_1,\ldots,\lambda_k), \quad D = \text{diag}(\sigma^2_{\varepsilon,1},\ldots,\sigma^2_{\varepsilon,n})$$

**Variance attribution** of asset $i$ to factor $j$:

$$R^2_{ij} = \frac{\left[\text{Cov}(r_i, f_j)\right]^2}{\text{Var}(r_i)}$$

**Idiosyncratic alpha signal** (annualised):

$$\alpha_i = \frac{252\,\mathbb{E}[\varepsilon_i]}{\sqrt{252}\,\text{std}(\varepsilon_i)} = \frac{\sqrt{252}\,\mu_{\varepsilon_i}}{\sigma_{\varepsilon_i}}$$

Assets with high $|\alpha_i|$ have consistent idiosyncratic drift unexplained by the common factors.

**Sharpe-ratio-maximising factor portfolio:**

$$\max_{\mathbf{w}}\;\frac{\mathbf{w}^\top \boldsymbol{\mu}_f - r_f}{\sqrt{\mathbf{w}^\top \Sigma_f \mathbf{w}}}, \qquad \text{s.t.}\;\sum_i w_i = 1$$

Map factor weights to asset weights: $\mathbf{w}_{\text{asset}} = B^\top\,\mathbf{w}_{\text{factor}}$.

**Walk-forward backtest** — $k$-fold TimeSeriesSplit:

$$\text{Sharpe}_{\text{OOS}} = \frac{\sqrt{252}\,\bar{r}_{\text{test}}}{\text{std}(r_{\text{test}})}, \qquad \text{Max Drawdown} = \min_t\!\left(\frac{\text{NAV}_t}{\max_{s \leq t}\text{NAV}_s} - 1\right)$$

### Key results
- PC1 (market factor) explains $\sim$50% of cross-sectional variance; PC2 separates risk-on/risk-off assets
- PCA 3-factor covariance approximation achieves $<10^{-4}$ MAE vs sample covariance
- Walk-forward backtest Sharpe $\approx 0.8$–$1.2$ (out-of-sample, 5-fold, 2016–2024)
- Max drawdown reduced vs SPY benchmark during 2020 and 2022 stress periods

---

## Mathematical Background

### Stochastic Calculus Essentials

**Itô's Lemma** — for $f(S_t, t)$ with $S_t$ following an Itô process:

$$df = \left(\frac{\partial f}{\partial t} + \mu\frac{\partial f}{\partial S} + \frac{1}{2}\sigma^2\frac{\partial^2 f}{\partial S^2}\right)dt + \sigma\frac{\partial f}{\partial S}\,dW_t$$

The $\frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial S^2}$ term (from quadratic variation $d[W]_t = dt$) has no classical analogue and is the source of the gamma P&L in option hedging.

**Risk-neutral pricing** — under measure $\mathbb{Q}$ (Girsanov):

$$C_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T)]$$

The risk-neutral drift is $r$ (not $\mu$), enforcing no-arbitrage.

**Black-Scholes PDE** (from delta-hedging argument):

$$\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} - rV = 0$$

---

## Skills Demonstrated

| Skill | Projects |
|-------|---------|
| Monte Carlo simulation & variance reduction | 2, 7 |
| Stochastic differential equations (Itô, EM) | 5 |
| Volatility modelling (HV, EWMA, GARCH) | 1, 6 |
| Options pricing (European, Asian, Barrier, Exchange) | 2, 7 |
| Greeks computation & finite differences | 3, 8 |
| Delta-hedging & PnL attribution | 3 |
| Implied volatility & arbitrage detection | 4 |
| Yield curve bootstrapping | 4 |
| Kalman filter & state-space models | 6 |
| PCA factor models & alpha signals | 1, 8 |
| Portfolio optimisation (Sharpe) | 8 |
| Walk-forward backtesting & cross-validation | 6, 8 |
| Numerical methods (Newton–Raphson, FD) | 3, 4 |

---

## Dependencies

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
matplotlib>=3.7
yfinance>=0.2.36
scikit-learn>=1.3
statsmodels>=0.14
```

---
