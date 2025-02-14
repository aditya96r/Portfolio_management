import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.ticker import FuncFormatter

# =====================
# Enhanced MPT Engine with CVaR
# =====================

class AdvancedPortfolioOptimizer:
    def __init__(self, risk_free_rate=0.015):
        self.assets = {
            'Equities': {'return': 0.07, 'volatility': 0.18, 'skew': -0.3},
            'Bonds': {'return': 0.03, 'volatility': 0.06, 'skew': 0.2},
            'REITs': {'return': 0.05, 'volatility': 0.12, 'skew': -0.1},
            'Commodities': {'return': 0.04, 'volatility': 0.15, 'skew': -0.4}
        }
        self.corr_matrix = np.array([
            [1.00, 0.15, 0.40, 0.55],
            [0.15, 1.00, 0.10, -0.05],
            [0.40, 0.10, 1.00, 0.25],
            [0.55, -0.05, 0.25, 1.00]
        ])
        self.rfr = risk_free_rate
        
    def calculate_portfolio_metrics(self, weights):
        ret = sum(w * self.assets[asset]['return'] for w, asset in zip(weights, self.assets))
        vol = np.sqrt(np.dot(weights.T, np.dot(self.corr_matrix * np.outer(
            [self.assets[asset]['volatility'] for asset in self.assets], 
            [self.assets[asset]['volatility'] for asset in self.assets]), 
            weights)))
        skew = sum(w * self.assets[asset]['skew'] for w, asset in zip(weights, self.assets))
        sharpe = (ret - self.rfr) / vol
        return ret, vol, sharpe, skew

    def optimize_portfolio(self, risk_profile):
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = self._get_bounds(risk_profile)
        
        def objective(weights):
            _, vol, sharpe, _ = self.calculate_portfolio_metrics(weights)
            return -sharpe + 0.1*vol  # Risk-adjusted optimization
        
        result = minimize(objective,
                         x0=np.ones(len(self.assets))/len(self.assets),
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        return result.x

    def _get_bounds(self, risk_profile):
        if risk_profile == 'Conservative':
            return [(0, 0.4), (0.2, 0.5), (0.1, 0.3), (0.05, 0.2)]
        elif risk_profile == 'Moderate':
            return [(0.3, 0.6), (0.1, 0.4), (0.05, 0.25), (0.05, 0.3)]
        return [(0.5, 0.8), (0, 0.3), (0, 0.2), (0, 0.2)]

# =====================
# Advanced GBM Simulation with Jump Diffusion
# =====================

def advanced_gbm_simulation(initial, weights, years, trading_days=250):
    optimizer = AdvancedPortfolioOptimizer()
    ret, vol, _, skew = optimizer.calculate_portfolio_metrics(weights)
    
    dt = 1/trading_days
    n_steps = years * trading_days
    shocks = np.random.normal(size=n_steps)
    jumps = np.random.poisson(0.05 * dt, n_steps) * np.random.normal(-0.1, 0.15, n_steps)
    
    # Adjusted drift with volatility drag and skewness adjustment
    drift = (ret - 0.5*vol**2 + skew/100) * dt
    diffusion = vol * np.sqrt(dt) * shocks
    compound_returns = drift + diffusion + jumps
    
    return initial * np.exp(np.cumsum(compound_returns))

# =====================
# Intelligent Annotation System
# =====================

class AnnotationManager:
    def __init__(self, ax):
        self.ax = ax
        self.annotations = []
        
    def add_annotation(self, x, y, text):
        annotation = self.ax.annotate(text, (x, y), xytext=(20, 10),
                                     textcoords='offset points',
                                     ha='left', va='bottom',
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", lw=1),
                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        self._avoid_overlap(annotation)
        self.annotations.append(annotation)
        
    def _avoid_overlap(self, new_ann):
        if not self.annotations:
            return
        fig = self.ax.figure
        fig.canvas.draw()
        new_bbox = new_ann.get_window_extent()
        for existing in self.annotations:
            existing_bbox = existing.get_window_extent()
            if new_bbox.intersects(existing_bbox):
                new_ann.xyann = (new_ann.xyann[0], new_ann.xyann[1] - 25)

# =====================
# Dynamic Visualization Engine
# =====================

def create_projection_plot(initial=2e5):
    fig, ax = plt.subplots(figsize=(16, 9))
    optimizer = AdvancedPortfolioOptimizer()
    ann_manager = AnnotationManager(ax)
    
    risk_profiles = {
        'Conservative': {'color': '#2ecc71', 'linestyle': '--'},
        'Moderate': {'color': '#3498db', 'linestyle': '-'},
        'Aggressive': {'color': '#e74c3c', 'linestyle': '-.'}
    }
    
    for profile, style in risk_profiles.items():
        weights = optimizer.optimize_portfolio(profile)
        ret, vol, sharpe, skew = optimizer.calculate_portfolio_metrics(weights)
        
        for years in [3, 5, 7, 10]:
            days = years * 250
            growth = advanced_gbm_simulation(initial, weights, years)
            
            ax.plot(np.linspace(0, days, days), growth, 
                   color=style['color'],
                   linestyle=style['linestyle'],
                   alpha=0.8,
                   lw=2)
            
            # Add smart annotation
            final_value = growth[-1]
            ann_manager.add_annotation(days, final_value,
                                      f"€{final_value/1e6:.2f}M\n"
                                      f"Sharpe: {sharpe:.2f}\n"
                                      f"Vol: {vol*100:.1f}%")

    # Dynamic formatting
    ax.set_title("Advanced Portfolio Projection System\nRisk-Adjusted Growth Scenarios", pad=25)
    ax.set_xlabel("Trading Days", labelpad=15)
    ax.set_ylabel("Portfolio Value (€)", labelpad=15)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x//250)}Y'))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f'€{x/1e3:.0f}K' if x < 1e6 else f'€{x/1e6:.2f}M'))
    
    # Adaptive scaling
    y_max = ax.get_ylim()[1]
    ax.set_yscale('log' if y_max/initial > 100 else 'linear')
    
    # Risk legend
    risk_text = "\n".join([
        f"{profile}: "
        f"Max Vol {optimizer._get_bounds(profile)[0][1]*100:.0f}%"
        for profile in risk_profiles])
    ax.text(0.98, 0.02, risk_text,
           transform=ax.transAxes,
           ha='right', va='bottom',
           bbox=dict(facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.show()

# Execute the visualization
create_projection_plot(initial=200000)
