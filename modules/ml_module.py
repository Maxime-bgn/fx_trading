"""
Module 3 — Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

PAIRS_ML = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "AUD/JPY"]

ML_SCORES = {
    "Pair":       PAIRS_ML,
    "P(Active)":  [0.92, 0.88, 0.85, 0.79, 0.81, 0.76],
    "Hurst":      [0.52, 0.58, 0.46, 0.43, 0.55, 0.51],
    "Sharpe 30d": [1.8,  1.6,  1.4,  1.2,  1.3,  0.9],
    "Status":     ["Active","Active","Active",
                   "Active","Active","Active"],
}


def gauge_chart(prob, pair_name):
    """LSTM confidence gauge."""
    if prob > 0.58:
        signal, color = "LONG",    "#3fb950"
    elif prob < 0.42:
        signal, color = "SHORT",   "#f85149"
    else:
        signal, color = "NEUTRAL", "#d29922"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number=dict(font=dict(size=26, color="white"), valueformat=".2f"),
        title=dict(text=f"<b>{pair_name}</b><br><span style='color:{color}'>{signal}</span>",
                   font=dict(size=13)),
        gauge=dict(
            axis=dict(range=[0, 1], tickwidth=1,
                      tickcolor="#30363d", tickfont=dict(size=9),
                      tickvals=[0, 0.42, 0.58, 1],
                      ticktext=["0", "0.42", "0.58", "1"]),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#161b22",
            borderwidth=1, bordercolor="#30363d",
            steps=[
                dict(range=[0,    0.42], color="#3d1a1a"),
                dict(range=[0.42, 0.58], color="#2d2a1a"),
                dict(range=[0.58, 1.00], color="#1a3d1a"),
            ],
            threshold=dict(
                line=dict(color="white", width=3),
                thickness=0.8, value=prob
            )
        )
    ))
    fig.update_layout(
        paper_bgcolor="#161b22",
        height=210,
        margin=dict(l=15, r=15, t=50, b=15),
    )
    return fig


def show():
    st.title("Machine Learning Module")
    st.markdown("---")

    st.subheader("Predicted Market Regime")

    regimes = {
        "Bullish Trending": {
            "color":  "#3fb950",
            "advice": "Momentum + Breakout recommended",
            "alloc":  "50% Momentum / 30% Breakout / 20% Others",
            "desc":   "Sustained positive momentum, moderate volatility, low VIX",
        },
        "Bearish Trending": {
            "color":  "#f85149",
            "advice": "Short Momentum + Protection",
            "alloc":  "50% Short Momentum / 30% Protection / 20% Cash",
            "desc":   "Negative momentum, rising volatility, high VIX",
        },
        "Range-Bound": {
            "color":  "#58a6ff",
            "advice": "Mean-Reversion + Carry recommended",
            "alloc":  "50% Mean-Rev / 30% Carry / 20% Others",
            "desc":   "No trend, low volatility, oscillations",
        },
        "High Volatility": {
            "color":  "#d29922",
            "advice": "Long Volatility + Straddles",
            "alloc":  "40% Options / 30% Mean-Rev / 30% Cash",
            "desc":   "Erratic moves, very high VIX, wide spreads",
        },
    }

    selected = st.selectbox(
        "Simulated regime (in production: predicted automatically by Random Forest)",
        list(regimes.keys()),
        index=0
    )
    r = regimes[selected]

    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        st.markdown(f"""
        <div style="background:{r['color']}22;border:2px solid {r['color']};
             border-radius:10px;padding:24px;text-align:center;">
            <div style="color:{r['color']};font-weight:bold;font-size:15px;margin-top:8px">
                {selected}
            </div>
            <div style="color:#8b949e;font-size:11px;margin-top:8px">{r['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.info(f"**Strategic advice**\n\n{r['advice']}")
    with c3:
        st.success(f"**Recommended allocation**\n\n{r['alloc']}")

    st.markdown("---")

    st.subheader("LSTM Confidence Gauges")

    lstm_probs = {
        "EUR/USD": 0.72,
        "GBP/USD": 0.61,
        "USD/JPY": 0.38,
        "USD/CHF": 0.45,
        "AUD/USD": 0.67,
        "AUD/JPY": 0.29,
    }

    cols = st.columns(3)
    for i, (pair, prob) in enumerate(lstm_probs.items()):
        with cols[i % 3]:
            st.plotly_chart(gauge_chart(prob, pair), use_container_width=True)

    st.markdown("---")

    st.subheader("Dynamic Pair Selection (Random Forest)")

    vix = st.slider("Simulated VIX", min_value=10, max_value=40, value=18, step=1)

    if vix < 15:
        n_active, vix_label, vix_color = 8, "Low",      "#3fb950"
    elif vix < 25:
        n_active, vix_label, vix_color = 6, "Moderate", "#d29922"
    else:
        n_active, vix_label, vix_color = 4, "High",     "#f85149"

    st.markdown(
        f"**VIX = {vix}** "
        f"<span style='color:{vix_color}'>({vix_label})</span>"
        f" → **{n_active} active pairs** selected",
        unsafe_allow_html=True
    )

    df_ml = pd.DataFrame(ML_SCORES)
    st.dataframe(
        df_ml,
        use_container_width=True,
        hide_index=True,
        column_config={
            "P(Active)":  st.column_config.ProgressColumn(
                "P(Active)", min_value=0, max_value=1, format="%.2f"),
            "Sharpe 30d": st.column_config.NumberColumn(
                "Sharpe 30d", format="%.1f"),
        }
    )

    st.markdown("---")

    st.subheader("Feature Importance — Random Forest")

    features = [
        "Sharpe 30d", "Hurst Ratio", "ADX",
        "Bid-ask Spread", "Realized Volatility",
        "Portfolio Correlation", "Momentum 60d",
        "VaR 95%", "Win Rate", "Others"
    ]
    importances = [18, 15, 12, 10, 9, 8, 7, 6, 5, 10]
    bar_colors  = ["#58a6ff"] * 3 + ["#8b949e"] * 7

    fig_fi = go.Figure(go.Bar(
        y=features[::-1],
        x=importances[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        text=[f"{v}%" for v in importances[::-1]],
        textposition="outside",
        hovertemplate="%{y}: %{x}%<extra></extra>",
    ))
    fig_fi.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        height=360,
        xaxis=dict(title="Importance (%)", gridcolor="#21262d", range=[0, 25]),
        yaxis=dict(gridcolor="#21262d"),
        margin=dict(l=0, r=60, t=10, b=0),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")

    st.subheader("Expected ML System Performance")

    perf_data = {
        "Metric":           ["Classification accuracy", "Sharpe improvement",
                             "Drawdown reduction", "Daily turnover"],
        "Expected value":   ["55-65%", "+0.3 to +0.5 pts", "15-20%", "5-10%"],
        "Comment":          [
            "60% is enough for a statistically significant edge",
            "Via dynamic inter-strategy allocation",
            "Avoids pairs in unfavorable regimes",
            "Light and low-cost reallocation",
        ],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)