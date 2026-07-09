import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go

from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="SalesIQ — Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════
# COLOR PALETTE (matches reference design)
# ══════════════════════════════════════════════════════════════════════════
BG        = "#0d1117"
PANEL     = "#12161c"
PANEL2    = "#161b22"
BORDER    = "#262c36"
BORDER_LT = "#1d222a"
TEXT      = "#e6edf3"
TEXT_DIM  = "#8b949e"
BLUE      = "#3b82f6"
PURPLE    = "#7c5cff"
GREEN     = "#22c55e"
RED       = "#ef4444"
ORANGE    = "#f59e0b"
CYAN      = "#38bdf8"

GRID      = "#1d222a"

# ══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — recreates the SalesIQ SaaS dashboard look
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

* {{ font-family: 'Inter', sans-serif !important; box-sizing: border-box; }}

.stApp {{ background: {BG}; color: {TEXT}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem !important; padding-bottom: 2rem !important; max-width: 1500px; }}

::-webkit-scrollbar {{ width: 6px; height:6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {PANEL2} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 0 !important; }}
[data-testid="stSidebar"] * {{ color: {TEXT_DIM} !important; }}

.brand-wrap {{
    display:flex; align-items:center; gap:10px;
    padding: 18px 18px 16px 18px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 6px;
}}
.brand-icon {{
    width:34px; height:34px; border-radius:9px;
    background: linear-gradient(135deg, {PURPLE}, {BLUE});
    display:flex; align-items:center; justify-content:center;
    font-size:16px; flex-shrink:0;
}}
.brand-name {{ font-size:16px; font-weight:800; color:{TEXT} !important; letter-spacing:-0.3px; line-height:1.1; }}
.brand-sub {{ font-size:10.5px; color:{TEXT_DIM} !important; margin-top:1px; letter-spacing:0.3px; }}

.nav-label {{
    font-size:10px; font-weight:700; color:{TEXT_DIM} !important;
    text-transform:uppercase; letter-spacing:1.2px;
    padding: 10px 18px 6px 18px; display:block;
}}

section[data-testid="stSidebar"] .stRadio > div {{ gap: 2px !important; padding: 0 10px; }}
section[data-testid="stSidebar"] .stRadio > div > label {{
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 9px 12px !important;
    font-size: 13.5px !important;
    font-weight: 500;
    color: {TEXT_DIM} !important;
    width: 100%;
    transition: all .15s;
}}
section[data-testid="stSidebar"] .stRadio > div > label:hover {{
    background: {BORDER_LT} !important;
    color: {TEXT} !important;
}}
section[data-testid="stSidebar"] .stRadio input:checked + div {{
    color: {BLUE} !important;
    font-weight: 600 !important;
}}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {{ display:none; }}

.quick-stats {{ padding: 6px 18px 0 18px; }}
.qs-row {{
    display:flex; justify-content:space-between; padding:7px 0;
    border-bottom: 1px solid {BORDER_LT};
}}
.qs-row:last-of-type {{ border-bottom:none; }}
.qs-label {{ font-size:12px; color:{TEXT_DIM}; }}
.qs-val {{ font-size:12px; color:{TEXT}; font-weight:700; font-family:'JetBrains Mono',monospace; }}
.qs-val.green {{ color:{GREEN}; }}

.live-badge {{
    margin: 14px 14px 10px 14px;
    background: {GREEN}14;
    border: 1px solid {GREEN}40;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 11px;
    color: {GREEN} !important;
    font-weight: 600;
    display:flex; align-items:center; gap:7px;
}}
.live-dot {{
    width:7px; height:7px; border-radius:50%; background:{GREEN};
    box-shadow: 0 0 0 0 {GREEN}80;
    animation: pulse 1.8s infinite;
}}
@keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 {GREEN}66; }}
    70% {{ box-shadow: 0 0 0 6px {GREEN}00; }}
    100% {{ box-shadow: 0 0 0 0 {GREEN}00; }}
}}

/* Page header */
h1.page-title {{ color:{TEXT} !important; font-weight:800 !important; font-size:1.85rem !important; letter-spacing:-0.6px; margin:0 0 4px 0 !important; }}
p.page-sub {{ color:{TEXT_DIM} !important; font-size:13.5px !important; margin:0 0 22px 0 !important; }}

/* KPI stat cards */
.stat-card {{
    background: {PANEL2};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    transition: border-color .18s, transform .18s;
    height: 100%;
}}
.stat-card:hover {{ border-color: #3b4250; transform: translateY(-2px); }}
.stat-icon {{
    position:absolute; top:16px; right:16px;
    width:28px; height:28px; border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-size:13px;
}}
.stat-label {{ font-size:10.5px; color:{TEXT_DIM}; text-transform:uppercase; letter-spacing:0.9px; font-weight:600; }}
.stat-value {{ font-size:25px; font-weight:800; color:{TEXT}; margin:8px 0 5px 0; letter-spacing:-0.8px; font-family:'JetBrains Mono',monospace;}}
.stat-sub {{ font-size:11.5px; font-weight:600; }}
.stat-up {{ color:{GREEN}; }}
.stat-down {{ color:{RED}; }}
.stat-neutral {{ color:{TEXT_DIM}; }}

/* Panels */
.panel {{
    background: {PANEL2};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 20px;
}}
.panel-title {{ font-size:14.5px; font-weight:700; color:{TEXT}; margin-bottom:2px; }}
.panel-sub {{ font-size:11.5px; color:{TEXT_DIM}; margin-bottom:14px; font-family:'JetBrains Mono',monospace; }}

/* Pill toggle buttons (segmented control) */
div[data-testid="column"] .stButton > button {{
    background: {BORDER_LT} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_DIM} !important;
    border-radius: 7px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
    width: 100%;
    transition: all .15s;
}}
div[data-testid="column"] .stButton > button:hover {{
    border-color: {BLUE} !important; color:{TEXT} !important;
}}
.stButton > button:focus {{ box-shadow:none !important; }}

.stSelectbox > div > div, .stMultiSelect > div > div {{
    background: {BORDER_LT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
}}
.stSlider > div > div > div {{ background: {BLUE} !important; }}
.stSlider label, .stSelectbox label {{ color:{TEXT_DIM} !important; font-size:11px !important; text-transform:uppercase; letter-spacing:0.8px; font-weight:600;}}

div.main .stRadio > div {{ gap: 8px !important; }}
div.main .stRadio > div > label {{
    background: {BORDER_LT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 7px 16px !important;
    font-size: 12px !important;
    color: {TEXT_DIM} !important;
    cursor: pointer;
    transition: all 0.15s;
}}
div.main .stRadio > div > label:hover {{ border-color:{ORANGE}; color:{TEXT} !important; }}

/* Tables */
.stDataFrame {{ border-radius: 10px; overflow: hidden; border: 1px solid {BORDER} !important; }}
[data-testid="stDataFrame"] * {{ color: {TEXT} !important; font-size: 12.5px !important; }}
[data-testid="stDataFrame"] th {{ background: {BORDER_LT} !important; color: {TEXT_DIM} !important; text-transform:uppercase; font-size:10.5px !important; letter-spacing:0.6px;}}
[data-testid="stDataFrame"] tr:hover td {{ background: {BORDER_LT} !important; }}

/* Cluster / segment cards */
.cluster-card {{
    background: {PANEL2};
    border: 1px solid {BORDER};
    border-left: 3px solid;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 14px;
}}
.cluster-head {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; }}
.cluster-dot {{ width:9px; height:9px; border-radius:50%; }}
.cluster-name {{ font-size:14.5px; font-weight:700; color:{TEXT}; }}
.cluster-items {{ font-size:12px; color:{TEXT_DIM}; margin-bottom:12px; line-height:1.6; }}
.cluster-strategy {{ font-size:12px; font-weight:600; padding:8px 12px; border-radius:8px; }}

/* Badges */
.badge {{ display:inline-block; padding:3px 11px; border-radius:12px; font-size:11px; font-weight:700; }}
.badge-up {{ background:{GREEN}1f; color:{GREEN}; }}
.badge-down {{ background:{RED}1f; color:{RED}; }}

hr {{ border-color:{BORDER_LT} !important; margin: 10px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Ship_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    return df

df = load_data()
monthly = df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
monthly.columns = ['Date', 'Sales']
weekly = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
weekly.columns = ['Date', 'Sales']

# ══════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════
def base_layout(height=380, legend=True):
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_DIM, family='Inter', size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        hovermode='x unified',
        showlegend=legend,
        legend=dict(orientation='h', yanchor='bottom', y=-0.28, xanchor='left', x=0,
                    font=dict(size=11, color=TEXT_DIM), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(showgrid=False, zeroline=False, color=TEXT_DIM, tickfont=dict(size=10.5)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, color=TEXT_DIM, tickfont=dict(size=10.5)),
    )

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
        <div class="brand-icon">📈</div>
        <div>
            <div class="brand-name">SalesIQ</div>
            <div class="brand-sub">Analytics Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="nav-label">Navigation</span>', unsafe_allow_html=True)
    page = st.radio("", [
        "📊  Sales Overview",
        "📈  Forecasting",
        "⚠️  Anomaly Report",
        "🗂️  Demand Segments"
    ], label_visibility='collapsed')

    st.markdown("<hr/>", unsafe_allow_html=True)

    yearly_rev = df.groupby('Year')['Sales'].sum()
    growth = (yearly_rev.iloc[-1] - yearly_rev.iloc[-2]) / yearly_rev.iloc[-2] * 100
    yr_min, yr_max = int(df['Year'].min()), int(df['Year'].max())

    st.markdown('<span class="nav-label">Quick Stats</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="quick-stats">
        <div class="qs-row"><span class="qs-label">Revenue</span><span class="qs-val">${df['Sales'].sum()/1e6:.2f}M</span></div>
        <div class="qs-row"><span class="qs-label">Orders</span><span class="qs-val">{len(df):,}</span></div>
        <div class="qs-row"><span class="qs-label">YoY Growth</span><span class="qs-val green">+{growth:.1f}%</span></div>
        <div class="qs-row"><span class="qs-label">Avg Order</span><span class="qs-val">${df['Sales'].mean():,.0f}</span></div>
        <div class="qs-row"><span class="qs-label">Period</span><span class="qs-val">{yr_min}–{yr_max}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="live-badge"><span class="live-dot"></span> LIVE · Updated just now</div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STAT CARD RENDERER
# ══════════════════════════════════════════════════════════════════════════
def stat_card(col, icon, icon_bg, label, value, sub, kind="neutral"):
    cls = {"up": "stat-up", "down": "stat-down", "neutral": "stat-neutral"}[kind]
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon" style="background:{icon_bg}22;color:{icon_bg};">{icon}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-sub {cls}">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — SALES OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "📊  Sales Overview":

    st.markdown("""
    <h1 class="page-title">Sales Overview</h1>
    <p class="page-sub">Full-year performance across all channels and regions</p>
    """, unsafe_allow_html=True)

    total_sales = df['Sales'].sum()
    total_orders = len(df)
    avg_order = df['Sales'].mean()
    avg_ship = df['Ship_Days'].mean()
    yearly_sales = df.groupby('Year')['Sales'].sum()
    yoy_growth = (yearly_sales.iloc[-1] - yearly_sales.iloc[-2]) / yearly_sales.iloc[-2] * 100
    yr_a, yr_b = yearly_sales.index[-1], yearly_sales.index[-2]
    orders_a = df[df.Year == yr_a].shape[0]
    orders_b = df[df.Year == yr_b].shape[0]
    orders_yoy = (orders_a - orders_b) / orders_b * 100
    n_cats = df['Category'].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    stat_card(c1, "$", BLUE, "Total Revenue", f"${total_sales/1e6:.2f}M", f"↗ +{yoy_growth:.1f}% YoY", "up")
    stat_card(c2, "🛒", GREEN, "Total Orders", f"{total_orders:,}", f"↗ +{orders_yoy:.1f}% YoY", "up")
    stat_card(c3, "📈", PURPLE, "Avg Order Value", f"${avg_order:,.0f}", "Per Transaction", "neutral")
    stat_card(c4, "🕐", ORANGE, "Avg Ship Time", f"{avg_ship:.1f} days", "Order to Delivery", "neutral")
    stat_card(c5, "◈", RED, "Categories", f"{n_cats}", " · ".join(df['Category'].unique()[:3]), "neutral")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Monthly Revenue Trend
    monthly['MA3'] = monthly['Sales'].rolling(3, min_periods=1).mean()
    peak_idx = monthly['Sales'].idxmax()

    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">Monthly Revenue Trend</div>
        <div class="panel-sub">{yr_min}-{yr_max} · 3-month moving average overlay · $K</div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['Date'], y=monthly['Sales']/1e3, name='Monthly Revenue',
                              line=dict(color=BLUE, width=2.2), fill='tozeroy',
                              fillcolor='rgba(59,130,246,0.10)'))
    fig.add_trace(go.Scatter(x=monthly['Date'], y=monthly['MA3']/1e3, name='3-Month Moving Avg',
                              line=dict(color=ORANGE, width=1.6, dash='dot')))
    fig.add_trace(go.Scatter(x=[monthly.loc[peak_idx, 'Date']]*2, y=[0, monthly['Sales'].max()/1e3],
                              name='Peak Month', mode='lines', line=dict(color=GREEN, width=1.2, dash='dash'),
                              hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[monthly.loc[peak_idx, 'Date']], y=[monthly.loc[peak_idx, 'Sales']/1e3],
                              mode='markers+text', text=[f"Peak: ${monthly.loc[peak_idx,'Sales']/1e3:.0f}K"],
                              textposition='top center', textfont=dict(color=GREEN, size=11),
                              marker=dict(color=GREEN, size=9, line=dict(color=BG, width=1.5)),
                              showlegend=False, hoverinfo='skip'))
    lay = base_layout(height=380)
    lay['yaxis']['tickprefix'] = '$'
    lay['yaxis']['ticksuffix'] = 'K'
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # By Year / Category / Region row
    col1, col2, col3 = st.columns([1, 1.3, 1.3])

    with col1:
        st.markdown('<div class="panel"><div class="panel-title">By Year</div><div class="panel-sub">Total revenue</div>', unsafe_allow_html=True)
        yearly = df.groupby('Year')['Sales'].sum()
        colors = [BLUE, GREEN, PURPLE, ORANGE, CYAN]
        fig2 = go.Figure(go.Bar(x=yearly.index.astype(str), y=yearly.values/1e6,
                                 marker_color=[colors[i % len(colors)] for i in range(len(yearly))],
                                 width=0.5,
                                 text=[f"${v/1e6:.1f}M" for v in yearly.values],
                                 textposition='outside', textfont=dict(color=TEXT, size=11)))
        lay2 = base_layout(height=320, legend=False)
        lay2['yaxis']['tickprefix'] = '$'
        lay2['yaxis']['ticksuffix'] = 'M'
        fig2.update_layout(**lay2)
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><div class="panel-title">Revenue by Category</div>', unsafe_allow_html=True)
        categories = df['Category'].unique().tolist()
        if 'cat_sel' not in st.session_state:
            st.session_state.cat_sel = categories[0]
        btn_cols = st.columns(len(categories))
        for i, cname in enumerate(categories):
            active = st.session_state.cat_sel == cname
            if btn_cols[i].button(cname, key=f"cat_{cname}", type=("primary" if active else "secondary")):
                st.session_state.cat_sel = cname
        cat = df.groupby('Category')['Sales'].sum().sort_values()
        bar_colors = [BLUE if idx == st.session_state.cat_sel else "#2a3140" for idx in cat.index]
        fig3 = go.Figure(go.Bar(y=cat.index, x=cat.values/1e3, orientation='h',
                                 marker_color=bar_colors,
                                 text=[f"${v/1e3:.0f}K" for v in cat.values],
                                 textposition='outside', textfont=dict(color=TEXT, size=11)))
        lay3 = base_layout(height=280, legend=False)
        lay3['xaxis']['tickprefix'] = '$'
        lay3['xaxis']['ticksuffix'] = 'K'
        lay3['xaxis']['showgrid'] = True
        lay3['xaxis']['gridcolor'] = GRID
        fig3.update_layout(**lay3)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="panel"><div class="panel-title">Revenue by Region</div>', unsafe_allow_html=True)
        regions = df['Region'].unique().tolist()
        if 'reg_sel' not in st.session_state:
            st.session_state.reg_sel = regions[0]
        btn_cols2 = st.columns(len(regions))
        for i, rname in enumerate(regions):
            active = st.session_state.reg_sel == rname
            if btn_cols2[i].button(rname, key=f"reg_{rname}", type=("primary" if active else "secondary")):
                st.session_state.reg_sel = rname
        reg = df.groupby('Region')['Sales'].sum().sort_values()
        bar_colors2 = [PURPLE if idx == st.session_state.reg_sel else "#2a3140" for idx in reg.index]
        fig4 = go.Figure(go.Bar(y=reg.index, x=reg.values/1e3, orientation='h',
                                 marker_color=bar_colors2,
                                 text=[f"${v/1e3:.0f}K" for v in reg.values],
                                 textposition='outside', textfont=dict(color=TEXT, size=11)))
        lay4 = base_layout(height=280, legend=False)
        lay4['xaxis']['tickprefix'] = '$'
        lay4['xaxis']['ticksuffix'] = 'K'
        lay4['xaxis']['showgrid'] = True
        lay4['xaxis']['gridcolor'] = GRID
        fig4.update_layout(**lay4)
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈  Forecasting":

    st.markdown("""
    <h1 class="page-title">Forecast Explorer</h1>
    <p class="page-sub">Prophet model projections with confidence intervals</p>
    """, unsafe_allow_html=True)

    segments = df['Category'].unique().tolist() + df['Region'].unique().tolist()
    if 'fc_sel' not in st.session_state:
        st.session_state.fc_sel = segments[0]

    st.markdown('<div class="panel"><div class="panel-title" style="margin-bottom:10px;">Segment Type</div>', unsafe_allow_html=True)
    seg_cols = st.columns(len(segments) + 2)
    for i, s in enumerate(segments):
        active = st.session_state.fc_sel == s
        if seg_cols[i].button(s, key=f"seg_{s}", type=("primary" if active else "secondary")):
            st.session_state.fc_sel = s
    with seg_cols[-1]:
        horizon = st.slider("FORECAST HORIZON — MONTHS", 1, 6, 3)
    st.markdown("</div>", unsafe_allow_html=True)

    seg_val = st.session_state.fc_sel
    if seg_val in df['Category'].unique():
        seg_df = df[df['Category'] == seg_val]
    else:
        seg_df = df[df['Region'] == seg_val]

    seg_monthly = seg_df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    seg_monthly.columns = ['ds', 'y']

    with st.spinner("Running Prophet model..."):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='additive')
        m.fit(seg_monthly)
        future = m.make_future_dataframe(periods=horizon, freq='ME')
        fc = m.predict(future)

    fo = fc.tail(horizon)

    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">Revenue Forecast — {seg_val}</div>
        <div class="panel-sub">Historical actuals + Prophet model projection · 95% confidence interval</div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seg_monthly['ds'], y=seg_monthly['y']/1e3, name='Actual Revenue',
                              line=dict(color=BLUE, width=2.2)))
    fig.add_trace(go.Scatter(x=fo['ds'], y=fo['yhat_upper']/1e3, line=dict(width=0),
                              showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fo['ds'], y=fo['yhat_lower']/1e3, fill='tonexty',
                              fillcolor='rgba(59,130,246,0.15)', line=dict(width=0),
                              name='95% Confidence Band'))
    fig.add_trace(go.Scatter(x=fo['ds'], y=fo['yhat']/1e3, name='Forecast',
                              line=dict(color=BLUE, width=2.2, dash='dash'),
                              mode='lines+markers', marker=dict(size=8, color=BLUE)))
    lay = base_layout(height=400)
    lay['yaxis']['tickprefix'] = '$'
    lay['yaxis']['ticksuffix'] = 'K'
    fig.update_layout(**lay)
    fig.add_annotation(x=fo['ds'].iloc[0], y=1.0, yref='paper', text="Forecast →",
                        showarrow=False, font=dict(color=TEXT_DIM, size=11), xanchor='left')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    train_pred = fc['yhat'].iloc[:len(seg_monthly)].values
    mae = np.mean(np.abs(seg_monthly['y'].values - train_pred))
    rmse = np.sqrt(np.mean((seg_monthly['y'].values - train_pred) ** 2))
    m1_pct = (fo['yhat'].iloc[0] - seg_monthly['y'].iloc[-1]) / seg_monthly['y'].iloc[-1] * 100
    m3_pct = (fo['yhat'].iloc[-1] - fo['yhat'].iloc[0]) / fo['yhat'].iloc[0] * 100

    c1, c2, c3, c4 = st.columns(4)
    stat_card(c1, "≈", CYAN, "MAE", f"${mae/1e3:.1f}K", "Mean Absolute Error", "neutral")
    stat_card(c2, "📈", ORANGE, "RMSE", f"${rmse/1e3:.1f}K", "Root Mean Sq. Error", "neutral")
    stat_card(c3, "↗", GREEN, "Month +1", f"${fo['yhat'].iloc[0]/1e3:.0f}K", f"↗ {m1_pct:+.1f}% vs prior", "up" if m1_pct >= 0 else "down")
    stat_card(c4, "↗", PURPLE, f"Month +{horizon}", f"${fo['yhat'].iloc[-1]/1e3:.0f}K", f"↗ {m3_pct:+.1f}% vs current", "up" if m3_pct >= 0 else "down")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">Forecast Breakdown</div>', unsafe_allow_html=True)
    tbl = fo[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    tbl.columns = ['Month', 'Forecast', 'Lower Bound', 'Upper Bound']
    tbl['Month'] = tbl['Month'].dt.strftime('%b %y')
    for c in ['Forecast', 'Lower Bound', 'Upper Bound']:
        tbl[c] = tbl[c].apply(lambda x: f"${x/1e3:,.0f}K")
    st.dataframe(tbl.set_index('Month'), use_container_width=True, hide_index=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY REPORT
# ══════════════════════════════════════════════════════════════════════════
elif page == "⚠️  Anomaly Report":

    st.markdown("""
    <h1 class="page-title">Anomaly Report</h1>
    <p class="page-sub">Isolation Forest + Z-Score detection on weekly sales data</p>
    """, unsafe_allow_html=True)

    ws = weekly.set_index('Date')['Sales']
    iso = IsolationForest(contamination=0.07, random_state=42)
    iso_labels = iso.fit_predict(ws.values.reshape(-1, 1))
    iso_anom = ws[iso_labels == -1]

    rm = ws.rolling(4, center=True).mean()
    rs = ws.rolling(4, center=True).std()
    z = (ws - rm) / rs
    z_anom = ws[z.abs() > 2]

    both = len(set(iso_anom.index.date) & set(z_anom.index.date))

    c1, c2, c3, c4 = st.columns(4)
    stat_card(c1, "⚠", ORANGE, "Isolation Forest", f"{len(iso_anom)} flags", "Detected anomalies", "neutral")
    stat_card(c2, "≈", RED, "Z-Score Anomalies", f"{len(z_anom)} flags", "|z| > 2", "neutral")
    stat_card(c3, "⚡", PURPLE, "Consensus Flags", f"{both} flags", "Both methods agree", "neutral")
    stat_card(c4, "📊", BLUE, "Weeks Analyzed", f"{len(ws)}", "Full dataset", "neutral")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if 'anom_method' not in st.session_state:
        st.session_state.anom_method = "Isolation Forest"

    st.markdown('<div class="panel"><div class="panel-title">Detection Method</div>', unsafe_allow_html=True)
    mcol1, mcol2, _ = st.columns([1, 1, 4])
    if mcol1.button("Isolation Forest", type=("primary" if st.session_state.anom_method == "Isolation Forest" else "secondary")):
        st.session_state.anom_method = "Isolation Forest"
    if mcol2.button("Z-Score", type=("primary" if st.session_state.anom_method == "Z-Score" else "secondary")):
        st.session_state.anom_method = "Z-Score"
    st.markdown("</div>", unsafe_allow_html=True)

    anom = iso_anom if st.session_state.anom_method == "Isolation Forest" else z_anom
    rolling_avg = ws.rolling(4, center=True).mean()

    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">Weekly Sales — Anomaly Detection</div>
        <div class="panel-sub">{st.session_state.anom_method} model · 4-week rolling average overlay</div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ws.index, y=ws.values/1e3, name='Weekly Revenue',
                              line=dict(color="#e6edf3", width=1.6)))
    fig.add_trace(go.Scatter(x=rolling_avg.index, y=rolling_avg.values/1e3, name='4-wk Rolling Avg',
                              line=dict(color=ORANGE, width=1.4, dash='dot')))
    spikes = anom[anom > ws.mean()]
    drops = anom[anom <= ws.mean()]
    if len(spikes):
        fig.add_trace(go.Scatter(x=spikes.index, y=spikes.values/1e3, mode='markers', name='Spike anomaly',
                                  marker=dict(symbol='triangle-up', size=13, color=GREEN,
                                              line=dict(color=BG, width=1))))
    if len(drops):
        fig.add_trace(go.Scatter(x=drops.index, y=drops.values/1e3, mode='markers', name='Drop anomaly',
                                  marker=dict(symbol='triangle-down', size=13, color=RED,
                                              line=dict(color=BG, width=1))))
    lay = base_layout(height=400)
    lay['yaxis']['tickprefix'] = '$'
    lay['yaxis']['ticksuffix'] = 'K'
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Anomaly Log</div>', unsafe_allow_html=True)
    anom_df = anom.reset_index()
    anom_df.columns = ['Date', 'Sales']
    anom_df['MonthNum'] = pd.to_datetime(anom_df['Date']).dt.month
    anom_df['Signal'] = anom_df['Sales'].apply(lambda x: 'Spike' if x > ws.mean() else 'Drop')
    cause_map = {
        1: 'Post-holiday slowdown', 2: 'Post-holiday dip', 3: 'Promotional campaign activation',
        4: 'Spring restocking', 5: 'Regional supply disruption', 6: 'Regional supply disruption',
        7: 'Summer promotions', 8: 'Back-to-school surge', 9: 'End-of-quarter procurement surge',
        10: 'Pre-holiday build-up', 11: 'Holiday season surge', 12: 'Christmas / year-end'
    }
    anom_df['Likely Cause'] = anom_df['MonthNum'].map(cause_map).fillna('Unusual fluctuation')
    anom_df['Revenue'] = anom_df['Sales'].apply(lambda x: f"${x:,.0f}")
    anom_df['Date'] = pd.to_datetime(anom_df['Date']).dt.strftime('%b %d, %Y')
    anom_df = anom_df[['Date', 'Revenue', 'Signal', 'Likely Cause']].iloc[::-1]
    st.dataframe(anom_df.set_index('Date'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — DEMAND SEGMENTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "🗂️  Demand Segments":

    st.markdown("""
    <h1 class="page-title">Demand Segments</h1>
    <p class="page-sub">K-Means clustering · 4 behavioral segments identified</p>
    """, unsafe_allow_html=True)

    features = df.groupby('Sub-Category').agg(
        Total_Sales=('Sales', 'sum'), Avg_Order=('Sales', 'mean'),
        Volatility=('Sales', 'std'), Count=('Sales', 'count')
    ).reset_index()

    yoy = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack()
    features['Growth'] = yoy.pct_change(axis=1).mean(axis=1).values
    features = features.dropna()

    X_s = StandardScaler().fit_transform(features[['Total_Sales', 'Avg_Order', 'Volatility', 'Growth']])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    features['Cluster'] = km.fit_predict(X_s)

    sc = features.groupby('Cluster')['Total_Sales'].mean().sort_values(ascending=False).index
    lmap = {sc[0]: 'High Volume Stable', sc[1]: 'Growing Demand',
            sc[2]: 'High Volatility', sc[3]: 'Declining Demand'}
    features['Segment'] = features['Cluster'].map(lmap)

    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_s)
    features['x'] = (Xp[:, 0] - Xp[:, 0].min()) / (Xp[:, 0].max() - Xp[:, 0].min()) * 100
    features['y'] = (Xp[:, 1] - Xp[:, 1].min()) / (Xp[:, 1].max() - Xp[:, 1].min()) * 100

    palette = {'High Volume Stable': BLUE, 'Growing Demand': GREEN,
               'High Volatility': ORANGE, 'Declining Demand': RED}

    strategies = {
        'High Volume Stable': ('Automate replenishment · maintain safety stock', BLUE),
        'Growing Demand': ('Increase forecast buffer 15-20% · expand SKUs', GREEN),
        'High Volatility': ('Safety stock ×2 · weekly demand sensing', ORANGE),
        'Declining Demand': ('Reduce orders · markdown aging inventory', RED),
    }

    col_l, col_r = st.columns([1.6, 1])

    with col_l:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">PCA Cluster Projection</div>
            <div class="panel-sub">K-Means k=4 · X-axis: sales volume · Y-axis: demand stability</div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        for seg, grp in features.groupby('Segment'):
            fig.add_trace(go.Scatter(
                x=grp['x'], y=grp['y'], mode='markers+text', name=seg,
                text=grp['Sub-Category'], textposition='top center',
                textfont=dict(size=9.5, color=TEXT_DIM),
                marker=dict(size=13, color=palette[seg], line=dict(color=PANEL2, width=1.5))
            ))
        lay = base_layout(height=520)
        lay['xaxis'].update(title='Sales Volume →', range=[-10, 110], showgrid=True, gridcolor=GRID)
        lay['yaxis'].update(title='Stability →', range=[-10, 110])
        lay['legend']['y'] = -0.14
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        for seg in ['High Volume Stable', 'Growing Demand', 'High Volatility', 'Declining Demand']:
            grp = features[features['Segment'] == seg]
            items = ' · '.join(sorted(grp['Sub-Category'].tolist()))
            color = palette[seg]
            strat_text, strat_color = strategies[seg]
            st.markdown(f"""
            <div class="cluster-card" style="border-left-color:{color};">
                <div class="cluster-head">
                    <span class="cluster-dot" style="background:{color};"></span>
                    <span class="cluster-name">{seg}</span>
                </div>
                <div class="cluster-items">{items}</div>
                <div class="cluster-strategy" style="background:{strat_color}18;color:{strat_color};">{strat_text}</div>
            </div>
            """, unsafe_allow_html=True)
