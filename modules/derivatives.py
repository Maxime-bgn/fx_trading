"""
Module 4 — Options & Derivatives
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


def garman_kohlhagen(S, K, T, rd, rf, sigma, option_type="call"):
    """FX option price and Greeks."""
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return dict(price=intrinsic, delta=0, gamma=0, vega=0, theta=0)

    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-rf*T) * norm.cdf(d1) - K * np.exp(-rd*T) * norm.cdf(d2)
        delta = np.exp(-rf*T) * norm.cdf(d1)
    else:
        price = K * np.exp(-rd*T) * norm.cdf(-d2) - S * np.exp(-rf*T) * norm.cdf(-d1)
        delta = -np.exp(-rf*T) * norm.cdf(-d1)

    gamma = np.exp(-rf*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * np.exp(-rf*T) * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (
        -(S * norm.pdf(d1) * sigma * np.exp(-rf*T)) / (2 * np.sqrt(T))
        - rd * K * np.exp(-rd*T) * norm.cdf(d2 if option_type=="call" else -d2)
        + rf * S * np.exp(-rf*T) * norm.cdf(d1 if option_type=="call" else -d1)
    ) / 365

    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)


def show():
    st.title("Options & Derivatives")
    st.markdown("---")

    st.sidebar.subheader("Option Parameters")

    S     = st.sidebar.number_input("Spot EUR/USD (S)", 0.90, 1.50, 1.10,
                                     0.001, format="%.4f")
    K     = st.sidebar.number_input("Strike (K)",       0.90, 1.50, 1.10,
                                     0.001, format="%.4f")
    T_days= st.sidebar.slider("Maturity (days)", 1, 365, 30)
    sigma = st.sidebar.slider("Implied volatility σ (%)", 1, 50, 10) / 100
    rd    = st.sidebar.slider("Domestic rate USD (%)", 0.0, 10.0, 5.0) / 100
    rf    = st.sidebar.slider("Foreign rate EUR (%)",  0.0, 10.0, 3.0) / 100
    notional = st.sidebar.number_input(
        "Notional (€)", 10_000, 10_000_000, 1_000_000, 10_000
    )
    T = T_days / 365

    call = garman_kohlhagen(S, K, T, rd, rf, sigma, "call")
    put  = garman_kohlhagen(S, K, T, rd, rf, sigma, "put")

    st.subheader("Options Portfolio Greeks")

    straddle = {
        "price": call["price"] + put["price"],
        "delta": call["delta"] + put["delta"],
        "gamma": call["gamma"] + put["gamma"],
        "vega":  call["vega"]  + put["vega"],
        "theta": call["theta"] + put["theta"],
    }

    df_greeks = pd.DataFrame([
        {
            "Strategy":    "Long Call ATM",
            "Premium (pip)": f"{call['price']*10000:.1f}",
            "Δ Delta":     f"{call['delta']:+.4f}",
            "Γ Gamma":     f"{call['gamma']:.5f}",
            "ν Vega (1%)": f"{call['vega']:+.4f}",
            "Θ Theta/d":   f"{call['theta']:+.4f}",
        },
        {
            "Strategy":    "Long Put ATM",
            "Premium (pip)": f"{put['price']*10000:.1f}",
            "Δ Delta":     f"{put['delta']:+.4f}",
            "Γ Gamma":     f"{put['gamma']:.5f}",
            "ν Vega (1%)": f"{put['vega']:+.4f}",
            "Θ Theta/d":   f"{put['theta']:+.4f}",
        },
        {
            "Strategy":    "Straddle (Call + Put)",
            "Premium (pip)": f"{straddle['price']*10000:.1f}",
            "Δ Delta":     f"{straddle['delta']:+.4f}",
            "Γ Gamma":     f"{straddle['gamma']:.5f}",
            "ν Vega (1%)": f"{straddle['vega']:+.4f}",
            "Θ Theta/d":   f"{straddle['theta']:+.4f}",
        },
    ])
    st.dataframe(df_greeks, use_container_width=True, hide_index=True)

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Δ Net Delta",   f"{straddle['delta']:+.4f}")
    g2.metric("Γ Gamma",       f"{straddle['gamma']:.5f}")
    g3.metric("ν Vega (1%)",   f"{straddle['vega']:+.4f}")
    g4.metric("Θ Theta/day",   f"{straddle['theta']:+.4f}")
    g5.metric("Total premium", f"{straddle['price']*10000:.0f} pips")

    st.markdown("---")

    st.subheader("Payoff Profile at Expiration")

    strategy = st.selectbox(
        "Strategy to display",
        ["Long Call", "Long Put", "Straddle (Call + Put)",
         "Risk Reversal (Collar)", "Bull Spread"]
    )

    spots = np.linspace(S * 0.85, S * 1.15, 400)

    def compute_payoff(strat, spots):
        pc = call["price"]
        pp = put["price"]

        if strat == "Long Call":
            return (np.maximum(spots - K, 0) - pc) * notional
        elif strat == "Long Put":
            return (np.maximum(K - spots, 0) - pp) * notional
        elif strat == "Straddle (Call + Put)":
            return (
                np.maximum(spots - K, 0) +
                np.maximum(K - spots, 0) - pc - pp
            ) * notional
        elif strat == "Risk Reversal (Collar)":
            K_put  = K * 0.99
            K_call = K * 1.01
            p_put  = garman_kohlhagen(S, K_put,  T, rd, rf, sigma, "put")["price"]
            p_call = garman_kohlhagen(S, K_call, T, rd, rf, sigma, "call")["price"]
            pnl_spot       = spots - S
            pnl_long_put   = np.maximum(K_put  - spots, 0) - p_put
            pnl_short_call = -(np.maximum(spots - K_call, 0) - p_call)
            return (pnl_spot + pnl_long_put + pnl_short_call) * notional
        elif strat == "Bull Spread":
            K2   = K * 1.02
            p_c2 = garman_kohlhagen(S, K2, T, rd, rf, sigma, "call")["price"]
            return (
                np.maximum(spots - K,  0) -
                np.maximum(spots - K2, 0) - pc + p_c2
            ) * notional

    pnl = compute_payoff(strategy, spots)

    sign_changes = np.where(np.diff(np.sign(pnl)))[0]
    breakevens = [spots[i] for i in sign_changes]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spots, y=np.maximum(pnl, 0),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=spots, y=np.minimum(pnl, 0),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=spots, y=pnl,
        mode="lines", name=strategy,
        line=dict(color="#58a6ff", width=2.5),
        hovertemplate="Spot: %{x:.4f}<br>P&L: €%{y:,.0f}<extra></extra>"
    ))
    fig.add_hline(y=0, line_color="#8b949e", line_width=1)
    fig.add_vline(x=S, line_dash="dash", line_color="#d29922",
                  annotation_text=f"Spot: {S:.4f}",
                  annotation_position="top right")
    fig.add_vline(x=K, line_dash="dot", line_color="#8b949e",
                  annotation_text=f"Strike: {K:.4f}",
                  annotation_position="top left")
    for be in breakevens:
        fig.add_vline(x=be, line_dash="dash", line_color="#3fb950",
                      annotation_text=f"BE: {be:.4f}",
                      annotation_position="bottom right")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        height=420,
        xaxis=dict(title="EUR/USD price at expiration", gridcolor="#21262d"),
        yaxis=dict(title="P&L (€)", gridcolor="#21262d"),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Maximum gain",  f"€{np.max(pnl):,.0f}")
    with col_b:
        st.metric("Maximum loss",  f"€{np.min(pnl):,.0f}")
    with col_c:
        if breakevens:
            st.metric("Break-even(s)",
                      " / ".join([f"{be:.4f}" for be in breakevens]))
        else:
            st.metric("Break-even(s)", "N/A")

    st.markdown("---")

    st.subheader("Greeks Reference Table")

    df_ref = pd.DataFrame({
        "Greek":          ["Δ Delta", "Γ Gamma", "ν Vega", "Θ Theta", "ρ Rho"],
        "Measure":        ["∂Price/∂Spot", "∂²Price/∂Spot²", "∂Price/∂Vol",
                           "∂Price/∂Time", "∂Price/∂Rate"],
        "Long Call":      ["+0 to +1", "Positive", "Positive", "Negative", "Positive"],
        "Long Put":       ["-1 to 0",  "Positive", "Positive", "Negative", "Negative"],
        "Maximum":        ["ITM", "ATM", "ATM", "ATM", "Long term"],
        "Interpretation": [
            "Equivalent spot held",
            "Gamma scalping profit",
            "Gain if volatility rises",
            "Cost of time decay",
            "Interest rate sensitivity",
        ],
    })
    st.dataframe(df_ref, use_container_width=True, hide_index=True)