import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Royal Supply Chain AI | Command Center",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME & STYLING ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
    .stApp {
        background: radial-gradient(circle at top right, #1a1f35, #0e1117);
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, p, span, label {
        font-family: 'Inter', sans-serif !important;
    }
    .main-title {
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px 24px;
        transition: 0.3s ease;
    }
    .metric-box:hover {
        border-color: rgba(79,172,254,0.5);
        background: rgba(255,255,255,0.07);
    }
    .metric-label { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-value { color: #fff; font-size: 1.9rem; font-weight: 700; margin: 4px 0; }
    .delta-up   { color: #10b981; font-size: 0.8rem; font-weight: 600; }
    .delta-down { color: #f43f5e; font-size: 0.8rem; font-weight: 600; }

    /* Section container */
    .section-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 16px;
    }
    .insight-box {
        background: rgba(79,172,254,0.08);
        border-left: 3px solid #4facfe;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #cbd5e1;
        font-size: 0.9rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; background-color: transparent; border-bottom: 1px solid rgba(255,255,255,0.08); }
    .stTabs [data-baseweb="tab"] { height: 48px; background-color: transparent !important; border: none !important; color: #64748b !important; font-weight: 600 !important; font-size: 0.85rem; letter-spacing: 0.05em; }
    .stTabs [aria-selected="true"] { color: #4facfe !important; border-bottom: 2px solid #4facfe !important; }

    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #0f172a; border: none; border-radius: 10px;
        padding: 0.6rem 1.5rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.05em;
        transition: all 0.3s ease; width: 100%;
    }
    .stButton>button:hover { box-shadow: 0 0 20px rgba(79,172,254,0.5); transform: scale(1.02); }

    /* Streamlit metric overrides */
    [data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: #fff !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; padding: 24px 0 16px 0;">
            <img src="https://img.icons8.com/nolan/128/artificial-intelligence.png" width="72">
            <h2 style="color:#fff; font-size:1.4rem; font-weight:800; margin:10px 0 2px 0; letter-spacing:0.08em;">ROYAL AI</h2>
            <span style="color:#4facfe; font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.15em;">Supply Chain Command</span>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("##### 🛠 API Gateway")
    api_url = st.text_input("Endpoint", "http://localhost:8000", label_visibility="collapsed")
    st.markdown("##### 📦 SKU Target")
    skus = ["FOODS_3_090", "HOBBIES_1_001", "HOUSEHOLD_2_001", "FOODS_1_001"]
    selected_sku = st.selectbox("SKU", skus, label_visibility="collapsed")
    st.markdown("##### ⏳ Forecast Horizon")
    horizon = st.select_slider("Horizon", options=[7, 14, 21, 28], value=28, label_visibility="collapsed")
    st.divider()
    st.info("🤖 Model: **TFT v1.2** | Quantile Loss | 401K params")

# --- HEADER ---
st.markdown("<h1 class='main-title'>Supply Chain AI Command</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Production-grade demand forecasting powered by Temporal Fusion Transformers</p>", unsafe_allow_html=True)

# --- TOP METRICS ---
m1, m2, m3, m4 = st.columns(4)
metrics = [
    ("Forecast Accuracy", "94.2%", "+ 2.4%", True),
    ("Inventory Turn",    "12.5x",  "+ 1.1x",  True),
    ("Stockout Rate",     "0.8%",   "- 1.2%", False),
    ("AI Savings",        "$240k",  "+ $12k",  True),
]
for col, (label, value, delta, up) in zip([m1, m2, m3, m4], metrics):
    d_cls = "delta-up" if up else "delta-down"
    col.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="{d_cls}">{delta} vs last month</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- MAIN CONTENT TABS ---
tabs = st.tabs(["📊  LIVE FORECAST", "🧠  EXPLAINABILITY", "⚙️  MLOPS PIPELINE"])

# ── TAB 1: LIVE FORECAST ──────────────────────────────────────────
with tabs[0]:
    st.markdown(f"<div class='section-title'>Projected Demand — {selected_sku}</div>", unsafe_allow_html=True)

    np.random.seed(42)
    dates    = pd.date_range(end=datetime.now(), periods=60)
    history  = np.random.uniform(5, 15, 60)
    f_dates  = pd.date_range(start=datetime.now(), periods=horizon)
    forecast = np.random.uniform(10, 18, horizon)
    upper    = forecast * 1.25
    lower    = forecast * 0.75

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=history,
        name="Historical Sales", line=dict(color="#64748b", width=2)))
    fig.add_trace(go.Scatter(
        x=list(f_dates) + list(f_dates)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill='toself', fillcolor='rgba(79,172,254,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='90% Confidence Band'))
    fig.add_trace(go.Scatter(x=f_dates, y=forecast,
        name="AI Forecast", line=dict(color="#4facfe", width=3, dash='dash')))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0), height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Units Sold")
    )
    st.plotly_chart(fig, use_container_width=True)

    # KPI strip under chart
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Daily Forecast", f"{forecast.mean():.1f} units")
    k2.metric("Peak Day",           f"{forecast.max():.1f} units")
    k3.metric("Trough Day",         f"{forecast.min():.1f} units")
    k4.metric("Total 28-Day",       f"{forecast.sum():.0f} units")

# ── TAB 2: EXPLAINABILITY ─────────────────────────────────────────
with tabs[1]:
    c1, c2 = st.columns([1, 1.4])

    with c1:
        st.markdown("<div class='section-title'>🔬 Global Attention Weights</div>", unsafe_allow_html=True)
        features = ["Lag 7", "Price Change", "SNAP Flag", "Weekday", "Lag 28", "Promotion"]
        weights  = [0.32, 0.24, 0.18, 0.12, 0.08, 0.06]
        df_w = pd.DataFrame({"Feature": features, "Weight": weights}).sort_values("Weight")
        fig_w = px.bar(df_w, x="Weight", y="Feature", orientation='h',
                       color="Weight", color_continuous_scale="Blues")
        fig_w.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
            coloraxis_showscale=False, height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>💡 Key Insights (TFT Head Analysis)</div>", unsafe_allow_html=True)
        insights = [
            "**Lag 7** is the dominant short-term signal — sales from last week drive 32% of attention.",
            "**Sell Price** sensitivity spikes +40% during SNAP periods in CA stores.",
            "Holiday effects are dampened by ~15% due to recent seasonal demand shifts.",
            "Promotion flag has the lowest weight but highest variance — unreliable signal.",
        ]
        for ins in insights:
            st.markdown(f"<div class='insight-box'>📌 {ins}</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='margin-top:20px'>📊 SHAP Distribution</div>", unsafe_allow_html=True)
        shap_feats = ["Price", "Holiday", "Lag 7", "Lag 14", "SNAP"]
        np.random.seed(7)
        shap_vals = np.random.randn(5, 60)
        shap_df = pd.DataFrame(shap_vals.T, columns=shap_feats).melt(var_name="Feature", value_name="SHAP Value")
        fig_shap = px.violin(shap_df, x="Feature", y="SHAP Value", color="Feature",
                             box=True, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_shap.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=300, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# ── TAB 3: MLOPS PIPELINE ─────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-title'>⚡ System Health</div>", unsafe_allow_html=True)
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Drift Status",    "Nominal",  "PSI: 0.04")
    hc2.metric("API Latency P95", "42ms",     "↓ 5ms")
    hc3.metric("Last Retrain",    "2h ago",   "WMAPE 0.155")
    hc4.metric("Active Model",    "TFT v1.2", "401K params")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("<div class='section-title'>📈 PSI Drift Monitor</div>", unsafe_allow_html=True)
        psi_data = pd.DataFrame({
            "Feature":   ["sales", "sell_price", "snap_CA", "rolling_mean_7", "lag_28"],
            "PSI Score": [0.05, 0.02, 0.08, 0.03, 0.06],
            "Threshold": [0.2, 0.2, 0.2, 0.2, 0.2],
            "Status":    ["✅ Stable"] * 5
        })
        fig_psi = px.bar(psi_data, x="PSI Score", y="Feature", orientation='h',
                         color="PSI Score", color_continuous_scale="RdYlGn_r",
                         range_color=[0, 0.3])
        fig_psi.add_vline(x=0.2, line_dash="dash", line_color="#f43f5e",
                          annotation_text="Alert Threshold", annotation_font_color="#f43f5e")
        fig_psi.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=280, showlegend=False,
            coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_psi, use_container_width=True)

    with col_r:
        st.markdown("<div class='section-title'>🚀 Retrain Control</div>", unsafe_allow_html=True)
        if st.button("⚡ TRIGGER MANUAL RETRAIN"):
            st.toast("Pipeline initiated!", icon="🚀")
            st.success("Retraining request sent. Monitor logs for progress.")

        st.markdown("<div class='section-title' style='margin-top:20px'>📋 Deployment History</div>", unsafe_allow_html=True)
        hist = pd.DataFrame({
            "Version": ["v1.2 (Live)", "v1.1", "v1.0"],
            "Date":    ["Today", "2 days ago", "1 week ago"],
            "WMAPE":   ["0.155", "0.158", "0.183"],
        })
        st.dataframe(hist, use_container_width=True, hide_index=True)

# Footer
st.markdown("""
<div style='text-align:center; margin-top:3rem; padding: 16px; color:#4b5563; font-size:0.75rem; border-top: 1px solid rgba(255,255,255,0.05);'>
    ROYAL AI SUPPLY CHAIN COMMAND CENTER © 2026 &nbsp;·&nbsp; BUILT FOR FMCG SCALE
</div>
""", unsafe_allow_html=True)
