# Portfolio Mangement System 
```

```

This is your portfolio

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://Portfolio_management.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

## Section Heading

This is filler text, please replace this with text for this section.

## Further Reading

# Professional Portfolio Manager

## **Institutional-Grade Portfolio Construction with Risk Management**

This repository provides an advanced wealth optimizer using Modern Portfolio Theory (MPT) with risk management techniques, partial factor investing, and Monte Carlo simulations for financial projections.

---
## **Features**

1. **Investor Risk Profiling**
   - Users answer questions to determine their risk profile (**Conservative, Moderate, Aggressive**).
   - Factors include **investment horizon, risk tolerance, financial knowledge, and age**.

2. **Portfolio Optimization**
   - Uses **Markowitz Modern Portfolio Theory (MPT)** to construct an optimal portfolio.
   - Incorporates **Ledoit-Wolf shrinkage** for stable covariance estimation.
   - **Moderate risk profile** uses **momentum/volatility screening** (partial factor investing).

3. **Risk Metrics Calculation**
   - **Expected Annual Return**
   - **Annual Volatility (Standard Deviation of Returns)**
   - **Sharpe Ratio**
   - **Maximum Drawdown** (Worst peak-to-trough loss)
   - **Value at Risk (VaR 95%)**
   - **Conditional Value at Risk (CVaR 95%)**

4. **Monte Carlo Simulations with Geometric Brownian Motion (GBM)**
   - Generates projections for **3, 5, 7, and 10 years**.
   - Provides **confidence bounds (80% range)** for each projection.
   - Uses **500 simulations per projection**.

---
## **Key Concepts and Formulas**

### **1. Factor Investing vs. Markowitz Theory**
#### **Markowitz Theory (Modern Portfolio Theory - MPT)**
- **Used in Code**: Yes.
- The code implements MPT through the `EfficientFrontier` class from `pypfopt`, which optimizes portfolios using mean-variance analysis.
- **Key components**:
  - **Efficient Frontier**: Portfolios that maximize returns for a given risk level.
  - **Sharpe Ratio Maximization**: `ef.max_sharpe()` optimizes risk-adjusted returns.
  - **Minimum Volatility**: `ef.min_volatility()` for conservative portfolios.

#### **Factor Investing**
- **Used in Code**: Partially (implied, not explicit).
- The code incorporates a **momentum/volatility factor** for stock selection in the "Moderate" risk profile:
  ```python
  momentum = data.pct_change(90).mean()
  volatility = data.pct_change().std()
  selected = (momentum / volatility).nlargest(8).index.tolist()
  ```
- This mimics a **quality factor** (high momentum, low volatility).
- **Not Explicitly Used**: Traditional factors like value, size, or quality are not directly modeled.

#### **Why Combine Both?**
- MPT provides the **portfolio optimization framework**, while factor-based selection improves the **asset universe** for optimization.
- Example: The "Moderate" profile uses momentum/volatility filtering to pre-select assets before applying MPT.

### **2. Ledoit-Wolf Shrinkage**
- **Purpose**:
  - Historical covariance matrices are noisy and unstable, especially with limited data.
  - Ledoit-Wolf shrinkage blends the sample covariance matrix with a structured estimator:
    
    \[\Sigma_{\text{shrunk}} = \alpha \cdot \Sigma_{\text{sample}} + (1-\alpha) \cdot \Sigma_{\text{target}}\]
  
- **Code Implementation**:
  ```python
  S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
  ```

### **3. CAPM (Capital Asset Pricing Model)**
- **Formula**:
  \[E(R_i) = R_f + \beta_i (E(R_m) - R_f)\]
  - \(R_f\): Risk-free rate
  - \(\beta_i\): Sensitivity to market movements
  - \(E(R_m)\): Expected market return
  
- **Code Implementation**:
  ```python
  mu = expected_returns.capm_return(data)
  ```

### **4. Risk Metrics**
#### **Sharpe Ratio**
- Measures risk-adjusted return:
  \[ S = \frac{E(R_p) - R_f}{\sigma_p} \]
- Higher values indicate better risk-adjusted performance.

#### **Maximum Drawdown (MDD)**
- Measures worst peak-to-trough loss:
  ```python
  cumulative_returns = portfolio_returns.cumsum()
  max_drawdown = (cumulative_returns.expanding().max() - cumulative_returns).max()
  ```

#### **Value at Risk (VaR 95%)**
- Measures worst-case loss with 95% confidence:
  ```python
  var_95 = np.percentile(portfolio_returns, 5) * 100
  ```

#### **Conditional Value at Risk (CVaR 95%)**
- Measures expected loss in worst 5% scenarios:
  ```python
  cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100
  ```

---
## **License**
This project is licensed under the MIT License.

---
## **Disclaimer**
This project is for **educational purposes only** and should not be considered as financial advice. Use it at your own discretion.

---
## **Author**
Developed by **Aditya Ravada**. Reach out on GitHub for collaboration!


