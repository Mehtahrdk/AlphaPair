import streamlit as st
import backtester as bt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="AlphaPair Engine", layout="wide")

# Custom CSS for "Quant" look
st.markdown("""
<style>
    .metric-container {
        background-color: #0E1117;
        border: 1px solid #262730;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("⚡ AlphaPair: Statistical Arbitrage Engine")
st.markdown("Institutional-grade backtesting engine for **Mean Reversion** strategies using Cointegration (ADF Test).")

# --- Sidebar ---
st.sidebar.header("Strategy Configuration")
ticker_1 = st.sidebar.text_input("Stock A (Leg 1)", "GOOG")
ticker_2 = st.sidebar.text_input("Stock B (Leg 2)", "GOOGL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
window = st.sidebar.slider("Rolling Window (Lookback)", 10, 100, 30)

if st.sidebar.button(" Run Backtest"):
    try:
        with st.spinner("Fetching Market Data & Calculating Signals..."):
            # 1. Fetch Data
            tickers = [ticker_1, ticker_2]
            data = bt.get_data(tickers, start_date, end_date)
            
            if data.empty or len(data.columns) < 2:
                st.error("Error: Could not fetch data. Please check tickers.")
                st.stop()
            
            # Ensure correct column mapping (First ticker is A, second is B)
            # data download might reorder columns alphabetically, so we reorder manually
            data = data[tickers]

        # 2. Main Dashboard Layout
        col1, col2 = st.columns([2, 1])
        
        # Run Backtest
        cum_returns, daily_returns, p_value, z_score, spread = bt.run_backtest(data, window)
        sharpe, max_drawdown = bt.calculate_metrics(daily_returns)
        total_return = (cum_returns.iloc[-1] - 1) * 100

        with col1:
            st.subheader(" Cumulative Returns (Equity Curve)")
            fig_pnl, ax_pnl = plt.subplots(figsize=(10, 5))
            ax_pnl.plot(cum_returns, label="AlphaPair Strategy", color='#00FFAA', linewidth=2)
            ax_pnl.set_title("Growth of $1 Investment")
            ax_pnl.grid(True, alpha=0.2)
            ax_pnl.legend()
            ax_pnl.set_facecolor('#0E1117') # Dark theme match
            fig_pnl.patch.set_facecolor('#0E1117')
            
            # Fix axis colors for dark theme
            ax_pnl.tick_params(colors='white')
            ax_pnl.xaxis.label.set_color('white')
            ax_pnl.yaxis.label.set_color('white')
            ax_pnl.title.set_color('white')
            for spine in ax_pnl.spines.values():
                spine.set_edgecolor('white')
                
            st.pyplot(fig_pnl)

        with col2:
            st.subheader(" Risk Metrics")
            st.metric("Total Return", f"{total_return:.2f}%")
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
            
            st.markdown("---")
            st.write(f"**Cointegration (ADF) P-Value:**")
            
            if p_value < 0.05:
                st.success(f"{p_value:.4f} (Cointegrated)")
            else:
                st.error(f" {p_value:.4f} (Broken Relationship)")

        # 3. Deep Dive Tabs
        tab1, tab2 = st.tabs([" Signal Analysis", " Parameter Optimization"])
        
        with tab1:
            st.subheader("Z-Score Divergence Signals")
            fig_z, ax_z = plt.subplots(figsize=(12, 4))
            ax_z.plot(z_score, label="Z-Score", color='cyan', linewidth=1)
            ax_z.axhline(2, color='red', linestyle='--', label="Short Threshold (+2)")
            ax_z.axhline(-2, color='lime', linestyle='--', label="Long Threshold (-2)")
            ax_z.axhline(0, color='white', alpha=0.5)
            ax_z.legend()
            ax_z.set_facecolor('#0E1117')
            fig_z.patch.set_facecolor('#0E1117')
            ax_z.tick_params(colors='white')
            st.pyplot(fig_z)

        with tab2:
            st.subheader("Find the Optimal Window")
            if st.button("Run Grid Search"):
                progress_bar = st.progress(0)
                results = []
                window_range = range(10, 90, 5)
                
                for i, w in enumerate(window_range):
                    # Quick run for optimization
                    cr, _, _, _, _ = bt.run_backtest(data, window=w)
                    final_ret = cr.iloc[-1]
                    results.append(final_ret)
                    progress_bar.progress((i + 1) / len(window_range))
                
                # Plot Optimization Curve
                fig_opt, ax_opt = plt.subplots(figsize=(10, 4))
                ax_opt.plot(window_range, results, marker='o', color='yellow')
                ax_opt.set_title("Total Return vs Window Size")
                ax_opt.set_xlabel("Window Size")
                ax_opt.set_ylabel("Final Equity Multiplier")
                ax_opt.grid(True, alpha=0.2)
                ax_opt.set_facecolor('#0E1117')
                fig_opt.patch.set_facecolor('#0E1117')
                ax_opt.tick_params(colors='white')
                ax_opt.xaxis.label.set_color('white')
                ax_opt.yaxis.label.set_color('white')
                ax_opt.title.set_color('white')
                for spine in ax_opt.spines.values():
                    spine.set_edgecolor('white')
                
                st.pyplot(fig_opt)
                
                best_win = window_range[np.argmax(results)]
                st.success(f" Optimal Window: {best_win} Days")

    except Exception as e:
        st.error(f"Runtime Error: {e}")

else:
    st.info(" Select parameters and click 'Run Backtest' to start.")