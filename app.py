"""
  FX TRADING SYSTEM - Graphical Interface
"""

import streamlit as st

st.set_page_config(
    page_title="FX Trading System | Eq487",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
    }
    h2, h3 { color: #c9d1d9; }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 FX Trading System")
    st.caption("Team 487")
    st.markdown("---")

    module = st.radio(
        "Navigation",
        options=[
            "Main Dashboard",
            "Market Analysis",
            "Machine Learning Module",
            "Options & Derivatives",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**System Status**")
    col1, col2 = st.columns(2)
    with col1:
        st.success("● Live")
    with col2:
        st.info("VIX: 18.3")
    st.markdown("---")

if "Dashboard" in module:
    from modules.dashboard import show
    show()
elif "Market" in module:
    from modules.market_analysis import show
    show()
elif "Machine Learning" in module:
    from modules.ml_module import show
    show()
elif "Options" in module:
    from modules.derivatives import show
    show()
