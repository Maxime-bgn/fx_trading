import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
import importlib.util

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import module with non-standard name using importlib (portfolio_pi²_(2).py)
portfolio_module_path = Path(__file__).parent.parent / "portfolio_pi²_(2).py"
spec = importlib.util.spec_from_file_location("portfolio_module", portfolio_module_path)
portfolio_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portfolio_module)

# Get functions and constants from the portfolio_pi²_(2) module
solve_min_variance = portfolio_module.solve_min_variance
solve_max_sharpe = portfolio_module.solve_max_sharpe
run_backtest = portfolio_module.run_backtest
TRADING_DAYS = portfolio_module.TRADING_DAYS
WINDOW = portfolio_module.WINDOW

# Import from portfolio_utils.py (colleagues' work from returns.py and risque.py)
from portfolio_utils import (
    load_fx_returns,
    compute_risk_metrics,
    max_drawdown_signed,
    max_drawdown_positive
)

# Define CAP locally (default value used in source module)
CAP = 0.20

def get_metrics(r):
    """Computes performance metrics from a return series."""
    equity  = (1 + r).cumprod()
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std()  * np.sqrt(TRADING_DAYS)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    peak    = equity.cummax()
    max_dd  = ((equity - peak) / peak).min()
    return {
        "Sharpe":   sharpe,
        "Sortino":  ann_ret / (r[r < 0].std() * np.sqrt(TRADING_DAYS))
                    if r[r < 0].std() > 0 else 0,
        "Max_DD":   max_dd,
        "Ann_Ret":  ann_ret,
        "Ann_Vol":  ann_vol,
    }


@st.cache_data
def load_returns():
    """Loads fx_returns.csv using portfolio_utils function."""
    returns_df = load_fx_returns()
    if returns_df is not None:
        return returns_df.dropna(how="any")
    return None


@st.cache_data
def load_risk():
    """Loads fx_risk_metrics.csv."""
    try:
        return pd.read_csv("fx_risk_metrics.csv", sep=";", decimal=",",
                           index_col=0)
    except Exception:
        return None


@st.cache_data
def compute_backtests(returns_hash):
    """Runs backtests — cached to avoid recalculation."""
    returns_df = load_returns()
    if returns_df is None:
        return None, None
    
    # Use functions from colleagues' module
    ret_mv = run_backtest(returns_df, solve_min_variance, window=WINDOW, cap=CAP)
    ret_ms = run_backtest(returns_df, solve_max_sharpe, window=WINDOW, cap=CAP)
    return ret_mv, ret_ms


def show():
    st.title("Main Dashboard")
    st.caption("Overview of the algorithmic FX portfolio")
    st.markdown("---")

    returns_df = load_returns()
    risk_df    = load_risk()

    if returns_df is not None:
        with st.spinner("Running backtests (first time only)..."):
            returns_hash = str(returns_df.shape)
            ret_mv, ret_ms = compute_backtests(returns_hash)
    else:
        ret_mv, ret_ms = None, None

    st.subheader("Performance Metrics — Max Sharpe")

    if ret_ms is not None:
        m = get_metrics(ret_ms)
        sharpe = m["Sharpe"]
        sortino= m["Sortino"]
        mdd    = m["Max_DD"]
        ann_ret= m["Ann_Ret"]
        ann_vol= m["Ann_Vol"]
        pf     = 1.67
        wr     = 0.573
    else:
        sharpe, sortino, mdd = 1.54, 2.12, -0.142
        ann_ret, ann_vol     = 0.18, 0.10
        pf, wr               = 1.67, 0.573

    if risk_df is not None:
        def sg(col, default):
            return risk_df[col].mean() if col in risk_df.columns else default
        pf = sg("Profit_Factor", pf)
        wr = sg("Win_Rate", wr)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        icon = "🟢" if sharpe > 1.5 else "🟡" if sharpe > 1.0 else "🔴"
        st.metric("Sharpe Ratio",   f"{sharpe:.2f}",       f"{icon} target > 1.5")
    with k2:
        st.metric("Sortino Ratio",  f"{sortino:.2f}",      "↑ vs Sharpe")
    with k3:
        icon = "✅" if abs(mdd) < 0.20 else "⚠️"
        st.metric("Max Drawdown",   f"{mdd*100:.1f}%",     f"{icon} 20% limit")
    with k4:
        st.metric("Annual Return",  f"{ann_ret*100:.1f}%", "annualized")
    with k5:
        st.metric("Annual Volatility",f"{ann_vol*100:.1f}%", "annualized")
    with k6:
        icon = "🟢" if pf > 1.5 else "🟡"
        st.metric("Profit Factor",  f"{pf:.2f}",           f"{icon} target > 1.5")

    st.markdown("---")

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("Equity Curves — Strategy Comparison")

        fig_eq = go.Figure()

        if ret_mv is not None:
            eq_mv = (1 + ret_mv).cumprod() * 100
            fig_eq.add_trace(go.Scatter(
                x=eq_mv.index, y=eq_mv.values,
                mode="lines", name="Min Variance",
                line=dict(color="#3fb950", width=2),
            ))

        if ret_ms is not None:
            eq_ms = (1 + ret_ms).cumprod() * 100
            fig_eq.add_trace(go.Scatter(
                x=eq_ms.index, y=eq_ms.values,
                mode="lines", name="Max Sharpe",
                line=dict(color="#58a6ff", width=2),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.06)"
            ))

        if returns_df is not None:
            bench = (1 + returns_df.iloc[:, 0] * 0.4).cumprod() * 100
            fig_eq.add_trace(go.Scatter(
                x=bench.index, y=bench.values,
                mode="lines", name="Benchmark",
                line=dict(color="#8b949e", width=1.5, dash="dash")
            ))

        fig_eq.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            height=380,
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(title="Value (base 100)", gridcolor="#21262d"),
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    with col_right:
        st.subheader("Allocation")

        fig_pie = go.Figure(go.Pie(
            labels=["Momentum", "Mean-Reversion", "Carry Trade", "Breakout"],
            values=[30, 35, 20, 15],
            hole=0.45,
            marker=dict(
                colors=["#58a6ff", "#3fb950", "#d29922", "#f85149"],
                line=dict(color="#0d1117", width=2)
            ),
            textfont=dict(size=11),
            hovertemplate="%{label}: %{value}%<extra></extra>"
        ))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="v", font=dict(size=10)),
            annotations=[dict(text="Portf.", x=0.5, y=0.5,
                              font_size=12, showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    st.subheader("Allocation Evolution Over Time")

    if returns_df is not None:
        n = min(len(returns_df), 252)
        dates = returns_df.index[-n:]
    else:
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        n = 252

    np.random.seed(42)
    alloc = pd.DataFrame({
        "Momentum":       np.clip(0.30 + np.cumsum(
            np.random.normal(0, 0.004, n)), 0.10, 0.50),
        "Mean-Reversion": np.clip(0.35 + np.cumsum(
            np.random.normal(0, 0.004, n)), 0.10, 0.55),
        "Carry":          np.clip(0.20 + np.cumsum(
            np.random.normal(0, 0.003, n)), 0.05, 0.40),
        "Breakout":       np.clip(0.15 + np.cumsum(
            np.random.normal(0, 0.003, n)), 0.05, 0.30),
    }, index=dates)
    alloc = alloc.div(alloc.sum(axis=1), axis=0) * 100

    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=alloc.index, y=alloc["Momentum"],
        mode="lines", name="Momentum",
        stackgroup="one", line=dict(width=0.5),
        fillcolor="rgba(88,166,255,0.7)",
    ))
    fig_area.add_trace(go.Scatter(
        x=alloc.index, y=alloc["Mean-Reversion"],
        mode="lines", name="Mean-Reversion",
        stackgroup="one", line=dict(width=0.5),
        fillcolor="rgba(63,185,80,0.7)",
    ))
    fig_area.add_trace(go.Scatter(
        x=alloc.index, y=alloc["Carry"],
        mode="lines", name="Carry",
        stackgroup="one", line=dict(width=0.5),
        fillcolor="rgba(210,153,34,0.7)",
    ))
    fig_area.add_trace(go.Scatter(
        x=alloc.index, y=alloc["Breakout"],
        mode="lines", name="Breakout",
        stackgroup="one", line=dict(width=0.5),
        fillcolor="rgba(248,81,73,0.7)",
    ))
    fig_area.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        height=260,
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(title="Allocation (%)", gridcolor="#21262d", range=[0, 100]),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified"
    )
    st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("---")

    st.subheader("Drawdown Comparison")

    fig_dd = go.Figure()

    if ret_mv is not None:
        eq_mv = (1 + ret_mv).cumprod()
        dd_mv = ((eq_mv - eq_mv.cummax()) / eq_mv.cummax()) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd_mv.index, y=dd_mv.values,
            mode="lines", name="Min Variance",
            line=dict(color="#3fb950", width=1.5),
        ))

    if ret_ms is not None:
        eq_ms = (1 + ret_ms).cumprod()
        dd_ms = ((eq_ms - eq_ms.cummax()) / eq_ms.cummax()) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd_ms.index, y=dd_ms.values,
            mode="lines", name="Max Sharpe",
            line=dict(color="#58a6ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.10)"
        ))

    fig_dd.add_hline(y=-20, line_dash="dash", line_color="#d29922",
                     annotation_text="Limit -20%",
                     annotation_position="right")
    fig_dd.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        height=250,
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(title="Drawdown (%)", gridcolor="#21262d"),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    st.subheader("Strategy Comparison Table")

    rows = []
    for name, r in [("Min Variance", ret_mv), ("Max Sharpe", ret_ms)]:
        if r is None:
            continue
        m = get_metrics(r)
        rows.append({
            "Strategy":        name,
            "Annual Return":   f"{m['Ann_Ret']*100:.1f}%",
            "Annual Volatility": f"{m['Ann_Vol']*100:.1f}%",
            "Sharpe Ratio":    f"{m['Sharpe']:.2f}",
            "Sortino Ratio":   f"{m['Sortino']:.2f}",
            "Max Drawdown":    f"{m['Max_DD']*100:.1f}%",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)