# ... [All previous code remains identical until Monte Carlo section] ...

        # Enhanced Monte Carlo Projections
        if investment > 0 and metrics.get('annual_return'):
            with st.expander(f"Monte Carlo Projections - €{investment:,.0f}", expanded=False):
                fig, ax = plt.subplots(figsize=(12, 7))
                periods = [3, 5, 7, 10]
                colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
                band_alphas = [0.15, 0.12, 0.09, 0.06]
                
                for years, color, alpha in zip(periods, colors, band_alphas):
                    simulations = 500
                    daily_returns = np.random.normal(
                        metrics['annual_return']/252,
                        metrics['annual_volatility']/np.sqrt(252),
                        (252*years, simulations)
                    )
                    growth = investment * np.exp(np.cumsum(daily_returns, axis=0))
                    growth_df = pd.DataFrame(growth)
                    
                    # Calculate confidence bands
                    upper = growth_df.quantile(0.9, axis=1)
                    lower = growth_df.quantile(0.1, axis=1)
                    median_growth = growth_df.median(axis=1)
                    
                    # Plot confidence bands
                    ax.fill_between(
                        range(len(median_growth)),
                        lower,
                        upper,
                        color=color,
                        alpha=alpha,
                        label=f'{years}Y 80% Range'
                    )
                    
                    # Plot median line
                    ax.plot(
                        median_growth, 
                        color=color, 
                        linewidth=2.8,
                        alpha=0.95,
                        label=f'{years}Y Median'
                    )
                    
                    # Final value annotation with dynamic positioning
                    x_pos = len(median_growth) - 1
                    y_pos = median_growth.iloc[-1]
                    
                    # Unique offsets for each timeline
                    offsets = {
                        3: (40, 40),   # (horizontal, vertical)
                        5: (40, -40),
                        7: (40, 20),
                        10: (40, -20)
                    }[years]
                    
                    ax.annotate(
                        f"€{y_pos/1e6:.2f}M" if y_pos >= 1e6 else f"€{y_pos/1e3:.0f}K",
                        xy=(x_pos, y_pos),
                        xytext=offsets,
                        textcoords='offset points',
                        color=color,
                        fontsize=10,
                        weight='bold',
                        ha='left',
                        va='center',
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=color,
                            lw=1,
                            alpha=0.6
                        ),
                        bbox=dict(
                            boxstyle='round,pad=0.2',
                            fc='white',
                            ec=color,
                            lw=1,
                            alpha=0.9
                        )
                    )

                # Simulation details moved to bottom-left
                sim_text = (
                    f"Monte Carlo Parameters:\n"
                    f"- 500 simulations per projection\n"
                    f"- GBM model with μ={metrics['annual_return']:.1%}\n"
                    f"- Annual volatility σ={metrics['annual_volatility']:.1%}"
                )
                ax.text(
                    0.02, 0.02,
                    sim_text,
                    transform=ax.transAxes,
                    ha='left',
                    va='bottom',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8)
                
                # Adjust plot boundaries
                ax.set_ylim(0, investment * 10)
                ax.set_xlim(0, 252*10 + 100)
                
                ax.set_title("Monte Carlo Projections with Confidence Bounds", 
                            fontsize=14, pad=15)
                ax.set_xlabel("Trading Days", fontsize=12)
                ax.set_ylabel("Portfolio Value (€)", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.2)
                ax.legend(loc='upper left', frameon=True, facecolor='white')
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f'€{x/1e6:.1f}M' if x >= 1e6 else f'€{x/1e3:.0f}K'))
                
                st.pyplot(fig)

