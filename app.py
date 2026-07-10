import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="SalesIQ — Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME COLORS ──────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT1   = "#e6edf3"
TEXT2   = "#c9d1d9"
TEXT3   = "#8b949e"
BLUE    = "#7091E6"
BLUE2   = "#3D52A0"
GREEN   = "#3fb950"
RED     = "#f85149"
YELLOW  = "#d29922"
PURPLE  = "#a371f7"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    font=dict(family="Inter, sans-serif", color=TEXT2, size=11),
    margin=dict(l=50, r=30, t=40, b=40),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER,
               showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER,
               showgrid=True, zeroline=False, gridwidth=0.5,
               griddash='dot'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=BORDER,
                borderwidth=1, font=dict(size=10, color=TEXT3)),
    hovermode='x unified',
    hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER,
                    font=dict(color=TEXT1, size=11))
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* {{ font-family: 'Inter', sans-serif !important; box-sizing: border-box; margin: 0; }}
.stApp {{ background: {BG}; color: {TEXT1}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {PANEL} !important;
    border-right: 1px solid {BORDER} !important;
    padding: 0 !important;
}}
[data-testid="stSidebar"] > div {{ padding-top: 0 !important; }}
[data-testid="stSidebar"] * {{ color: {TEXT3} !important; }}

/* Nav radio */
[data-testid="stSidebar"] .stRadio > div {{
    display: flex !important; flex-direction: column !important; gap: 2px !important;
}}
[data-testid="stSidebar"] .stRadio > div > label {{
    background: transparent !important; border: none !important;
    border-radius: 6px !important; padding: 9px 14px !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: {TEXT3} !important; cursor: pointer !important;
    display: flex !important; align-items: center !important;
    transition: all 0.15s !important;
}}
[data-testid="stSidebar"] .stRadio > div > label:hover {{
    background: #21262d !important; color: {TEXT1} !important;
}}
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {{
    background: #21262d !important; color: {BLUE} !important;
    border-right: 2px solid {BLUE} !important;
}}
[data-testid="stSidebar"] .stRadio input {{ display: none !important; }}

/* Inputs */
.stSelectbox > div > div, .stMultiSelect > div > div {{
    background: #21262d !important; border: 1px solid {BORDER} !important;
    border-radius: 6px !important; color: {TEXT1} !important; font-size: 13px !important;
}}
.stSlider > div > div > div {{ background: {BLUE} !important; }}
.stSlider label {{ color: {TEXT3} !important; font-size: 11px !important;
    text-transform: uppercase !important; letter-spacing: 0.8px !important; }}

/* Buttons as pills */
.stRadio:not([data-testid="stSidebar"] .stRadio) > div {{
    display: flex !important; flex-direction: row !important; gap: 6px !important;
    flex-wrap: wrap !important;
}}
.stRadio:not([data-testid="stSidebar"] .stRadio) > div > label {{
    background: #21262d !important; border: 1px solid {BORDER} !important;
    border-radius: 6px !important; padding: 6px 14px !important;
    font-size: 12px !important; color: {TEXT3} !important; cursor: pointer !important;
}}
.stRadio:not([data-testid="stSidebar"] .stRadio) > div > label:has(input:checked) {{
    background: {BLUE} !important; color: white !important; border-color: {BLUE} !important;
}}
.stRadio input {{ display: none !important; }}

/* Text */
h1 {{ color: {TEXT1} !important; font-weight: 700 !important;
    font-size: 1.6rem !important; letter-spacing: -0.3px; margin-bottom: 2px !important; }}
p  {{ color: {TEXT3} !important; font-size: 12px !important; font-family: monospace !important; }}
hr {{ border-color: #21262d !important; margin: 12px 0 !important; }}

/* Dataframe */
.stDataFrame {{ border-radius: 8px; overflow: hidden; }}
[data-testid="stDataFrame"] th {{
    background: #21262d !important; color: {TEXT3} !important;
    font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.5px;
}}
[data-testid="stDataFrame"] td {{ color: {TEXT2} !important; font-size: 12px !important; }}

/* KPI card */
.kpi-card {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 16px 18px; position: relative; overflow: hidden;
    transition: border-color 0.2s;
}}
.kpi-card:hover {{ border-color: {BLUE2}; }}
.kpi-icon {{
    position: absolute; top: 14px; right: 14px; width: 28px; height: 28px;
    border-radius: 6px; display: flex; align-items: center;
    justify-content: center; font-size: 14px;
}}
.kpi-label {{ font-size: 10px; color: {TEXT3}; text-transform: uppercase;
    letter-spacing: 0.8px; font-weight: 600; margin-bottom: 8px; }}
.kpi-value {{ font-size: 26px; font-weight: 800; color: {TEXT1};
    letter-spacing: -0.5px; margin-bottom: 5px; }}
.kpi-delta {{ font-size: 11px; font-weight: 500; }}
.delta-up   {{ color: {GREEN}; }}
.delta-down {{ color: {RED}; }}
.delta-neu  {{ color: {TEXT3}; }}

/* Chart card */
.chart-card {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 20px; margin: 12px 0;
}}
.chart-title {{ font-size: 14px; font-weight: 600; color: {TEXT1}; margin-bottom: 2px; }}
.chart-sub   {{ font-size: 11px; color: {TEXT3}; font-family: monospace; margin-bottom: 14px; }}

/* Segment card */
.seg-card {{
    background: {PANEL}; border: 1px solid {BORDER}; border-left: 3px solid;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
}}
.seg-name  {{ font-size: 13px; font-weight: 700; color: {TEXT1}; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }}
.seg-items {{ font-size: 11px; color: {TEXT3}; margin-bottom: 8px; line-height: 1.6; }}
.seg-strat {{
    font-size: 11px; font-weight: 500; padding: 5px 10px;
    border-radius: 4px; font-family: monospace;
    display: inline-block;
}}

/* Sidebar brand */
.brand {{ padding: 18px 16px 14px; border-bottom: 1px solid {BORDER}; margin-bottom: 6px; }}
.brand-name {{ font-size: 18px; font-weight: 800; color: {TEXT1}; letter-spacing: -0.5px; }}
.brand-name span {{ color: {BLUE}; }}
.brand-sub  {{ font-size: 10px; color: {TEXT3}; margin-top: 1px; letter-spacing: 0.3px; }}
.nav-label  {{ font-size: 10px; color: {TEXT3}; text-transform: uppercase;
    letter-spacing: 1px; font-weight: 600; padding: 6px 16px 4px; display: block; }}

/* Quick stats */
.qs-row {{ display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid #21262d; }}
.qs-label {{ font-size: 11px; color: {TEXT3}; }}
.qs-val   {{ font-size: 11px; font-weight: 700; color: {TEXT2}; }}

/* Live badge */
.live-badge {{
    display: flex; align-items: center; gap: 6px; padding: 8px 16px;
    border-top: 1px solid {BORDER}; font-size: 11px; color: {GREEN};
    font-weight: 500;
}}
.live-dot {{ width: 7px; height: 7px; border-radius: 50%;
    background: {GREEN}; animation: pulse 2s infinite; }}
@keyframes pulse {{
    0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }}
}}

/* Detection pill */
.det-label {{ font-size: 10px; color: {TEXT3}; text-transform: uppercase;
    letter-spacing: 0.8px; font-weight: 600; margin-bottom: 8px; }}

/* Anomaly signal badge */
.badge-spike {{ background: rgba(63,185,80,0.15); color: {GREEN};
    border: 1px solid rgba(63,185,80,0.3); border-radius: 4px;
    padding: 2px 8px; font-size: 11px; font-weight: 600; }}
.badge-drop  {{ background: rgba(248,81,73,0.15); color: {RED};
    border: 1px solid rgba(248,81,73,0.3); border-radius: 4px;
    padding: 2px 8px; font-size: 11px; font-weight: 600; }}

/* Page header */
.page-hdr {{ margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid {BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Year']    = df['Order Date'].dt.year
    df['Month']   = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Ship_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    return df

df = load_data()
monthly = df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
monthly.columns = ['Date','Sales']
weekly  = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
weekly.columns  = ['Date','Sales']

yearly_rev = df.groupby('Year')['Sales'].sum()
yoy = (yearly_rev.iloc[-1]-yearly_rev.iloc[-2])/yearly_rev.iloc[-2]*100

# ── HELPERS ───────────────────────────────────────────────────────────────
def kpi(col, label, value, delta, delta_type, icon, icon_bg):
    dc = "delta-up" if delta_type=="up" else ("delta-down" if delta_type=="down" else "delta-neu")
    arrow = "↗" if delta_type=="up" else ("↘" if delta_type=="down" else ""  )
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon' style='background:{icon_bg}20;'>{icon}</div>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-delta {dc}'>{arrow} {delta}</div>
        </div>""", unsafe_allow_html=True)

def chart_wrap(title, subtitle, content_fn):
    st.markdown(f"""
    <div class='chart-card'>
        <div class='chart-title'>{title}</div>
        <div class='chart-sub'>{subtitle}</div>
    </div>""", unsafe_allow_html=True)
    content_fn()

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class='brand'>
        <div class='brand-name'>Sales<span>IQ</span></div>
        <div class='brand-sub'>Analytics Platform</div>
    </div>
    <span class='nav-label'>Navigation</span>
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        "📊  Sales Overview",
        "📈  Forecasting",
        "🔔  Anomaly Report",
        "🗂  Demand Segments",
    ], label_visibility='collapsed')

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='padding:0 16px;'>
        <span class='nav-label' style='padding:0 0 8px;'>Quick Stats</span>
        <div class='qs-row'><span class='qs-label'>Revenue</span>
            <span class='qs-val'>${df['Sales'].sum()/1e6:.2f}M</span></div>
        <div class='qs-row'><span class='qs-label'>Orders</span>
            <span class='qs-val'>{len(df):,}</span></div>
        <div class='qs-row'><span class='qs-label'>YoY Growth</span>
            <span class='qs-val' style='color:{GREEN};'>+{yoy:.1f}%</span></div>
        <div class='qs-row'><span class='qs-label'>Avg Order</span>
            <span class='qs-val'>${df['Sales'].mean():,.0f}</span></div>
        <div class='qs-row' style='border:none;'><span class='qs-label'>Period</span>
            <span class='qs-val'>{df['Year'].min()}–{df['Year'].max()}</span></div>
    </div>
    <div class='live-badge' style='margin-top:auto;position:fixed;bottom:0;width:196px;background:{PANEL};'>
        <div class='live-dot'></div> LIVE · Updated just now
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — SALES OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "📊  Sales Overview":
    st.markdown(f"""
    <div class='page-hdr'>
        <h1>Sales Overview</h1>
        <p>Full-year performance across all channels and regions</p>
    </div>""", unsafe_allow_html=True)

    total = df['Sales'].sum()
    avg_s = df['Ship_Days'].mean()
    avg_o = df['Sales'].mean()
    yoy2  = (yearly_rev.iloc[-2]-yearly_rev.iloc[-3])/yearly_rev.iloc[-3]*100

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1,"Total Revenue",  f"${total/1e6:.2f}M",  f"+{yoy:.1f}% YoY", "up",   "💲","#3D52A0")
    kpi(c2,"Total Orders",   f"{len(df):,}",         f"+12.1% YoY",       "up",   "🛒","#3fb950")
    kpi(c3,"Avg Order Value",f"${avg_o:,.0f}",        f"+3.8% YoY",        "up",   "📈","#a371f7")
    kpi(c4,"Avg Ship Time",  f"{avg_s:.1f} days",    f"-0.1% YoY",        "down", "⏱","#d29922")
    kpi(c5,"Categories",     "3",                     "Furn · Tech · Office","neu","🏷","#f85149")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Monthly Trend ──
    st.markdown(f"""
    <div class='chart-card'>
        <div class='chart-title'>Monthly Revenue Trend</div>
        <div class='chart-sub'>{df['Year'].min()}–{df['Year'].max()} · 3-month moving average overlay · $K</div>
    """, unsafe_allow_html=True)

    ma3 = monthly['Sales'].rolling(3).mean()
    pidx = monthly['Sales'].idxmax()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['Date'], y=monthly['Sales']/1e3,
        name='Monthly Revenue', line=dict(color=BLUE, width=2),
        fill='tozeroy', fillcolor='rgba(112,145,230,0.08)',
        hovertemplate='%{x|%b %Y}<br>Revenue: $%{y:.0f}K<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=monthly['Date'], y=ma3/1e3,
        name='3-Month Moving Avg', line=dict(color=YELLOW, width=1.5, dash='dash'),
        hovertemplate='%{x|%b %Y}<br>3mo Avg: $%{y:.0f}K<extra></extra>'
    ))
    # Peak line
    fig.add_vline(x=monthly.loc[pidx,'Date'], line_dash='dot',
                  line_color=GREEN, line_width=1, opacity=0.6)
    fig.add_annotation(x=monthly.loc[pidx,'Date'],
                       y=monthly.loc[pidx,'Sales']/1e3,
                       text=f"Peak", showarrow=False,
                       font=dict(color=GREEN, size=11, family='Inter'),
                       xshift=10, yshift=10)
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
        yaxis_tickprefix='$', yaxis_ticksuffix='K',
        xaxis_tickformat='%b %y')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── By Year + Category + Region ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='chart-card'><div class='chart-title'>By Year</div>", unsafe_allow_html=True)
        yr = df.groupby('Year')['Sales'].sum()
        fig2 = go.Figure(go.Bar(
            x=yr.index.astype(str), y=yr.values/1e6,
            marker_color=[BLUE2, BLUE, GREEN, '#a371f7'],
            text=[f'${v/1e6:.1f}M' for v in yr.values],
            textposition='outside', textfont=dict(size=10, color=TEXT2),
            hovertemplate='%{x}<br>$%{y:.2f}M<extra></extra>'
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=220,
            yaxis_tickprefix='$', yaxis_ticksuffix='M',
            showlegend=False, margin=dict(l=40,r=20,t=10,b=30))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar':False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='chart-card'><div class='chart-title'>Revenue by Category</div>", unsafe_allow_html=True)
        cat_sel = st.multiselect("", df['Category'].unique().tolist(),
            default=df['Category'].unique().tolist(), label_visibility='collapsed',
            key='cat_filter')
        cat = df[df['Category'].isin(cat_sel)].groupby('Category')['Sales'].sum().sort_values()
        fig3 = go.Figure(go.Bar(
            y=cat.index, x=cat.values/1e3,
            orientation='h',
            marker_color=[BLUE, BLUE2, '#8697C4'],
            text=[f'${v/1e3:.0f}K' for v in cat.values],
            textposition='outside', textfont=dict(size=10,color=TEXT2),
            hovertemplate='%{y}<br>$%{x:.0f}K<extra></extra>'
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=220,
            xaxis_tickprefix='$', xaxis_ticksuffix='K',
            showlegend=False, margin=dict(l=100,r=60,t=10,b=30))
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar':False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div class='chart-card'><div class='chart-title'>Revenue by Region</div>", unsafe_allow_html=True)
        reg_sel = st.multiselect("", df['Region'].unique().tolist(),
            default=df['Region'].unique().tolist(), label_visibility='collapsed',
            key='reg_filter')
        reg = df[df['Region'].isin(reg_sel)].groupby('Region')['Sales'].sum().sort_values()
        fig4 = go.Figure(go.Bar(
            y=reg.index, x=reg.values/1e3,
            orientation='h',
            marker_color=[PURPLE, BLUE, '#8697C4', '#ADBBDA'],
            text=[f'${v/1e3:.0f}K' for v in reg.values],
            textposition='outside', textfont=dict(size=10,color=TEXT2),
            hovertemplate='%{y}<br>$%{x:.0f}K<extra></extra>'
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, height=220,
            xaxis_tickprefix='$', xaxis_ticksuffix='K',
            showlegend=False, margin=dict(l=70,r=60,t=10,b=30))
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar':False})
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈  Forecasting":
    st.markdown(f"""
    <div class='page-hdr'>
        <h1>Forecast Explorer</h1>
        <p>Prophet model projections with confidence intervals</p>
    </div>""", unsafe_allow_html=True)

    # Controls card
    st.markdown(f"<div class='chart-card' style='padding:14px 18px;'>", unsafe_allow_html=True)
    cc1, cc2 = st.columns([3,2])
    with cc1:
        st.markdown("<div class='det-label'>Segment Type</div>", unsafe_allow_html=True)
        seg_choice = st.radio("seg", ["Technology","Furniture","Office Supplies","West","East"],
                               horizontal=True, label_visibility='collapsed')
    with cc2:
        st.markdown("<div class='det-label'>Forecast Horizon — Months</div>", unsafe_allow_html=True)
        horizon = st.slider("", 1, 3, 3, label_visibility='collapsed')
    st.markdown("</div>", unsafe_allow_html=True)

    # Get segment data
    if seg_choice in ["Technology","Furniture","Office Supplies"]:
        seg_df = df[df['Category']==seg_choice]
    else:
        seg_df = df[df['Region']==seg_choice]

    seg_m = seg_df.groupby(pd.Grouper(key='Order Date',freq='ME'))['Sales'].sum().reset_index()
    seg_m.columns = ['ds','y']

    with st.spinner("Running Prophet model..."):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='additive')
        m.fit(seg_m)
        future = m.make_future_dataframe(periods=horizon, freq='ME')
        fc = m.predict(future)

    fo = fc.tail(horizon)
    hist_last = seg_m.tail(6)

    # Chart
    st.markdown(f"""
    <div class='chart-card'>
        <div class='chart-title'>Revenue Forecast — {seg_choice}</div>
        <div class='chart-sub'>Historical actuals + Prophet model projection · 95% confidence interval</div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seg_m['ds'], y=seg_m['y']/1e3,
        name='Actual Revenue', line=dict(color=BLUE, width=2),
        hovertemplate='%{x|%b %y}<br>$%{y:.0f}K<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=fo['ds'], y=fo['yhat']/1e3,
        name='Forecast', line=dict(color=BLUE, width=2, dash='dash'),
        mode='lines+markers', marker=dict(size=8, color=BLUE,
        line=dict(color=BG, width=2)),
        hovertemplate='%{x|%b %y}<br>Forecast: $%{y:.0f}K<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fo['ds'], fo['ds'][::-1]]),
        y=pd.concat([fo['yhat_upper']/1e3, fo['yhat_lower'][::-1]/1e3]),
        fill='toself', fillcolor='rgba(112,145,230,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='95% Confidence Band',
        hoverinfo='skip'
    ))
    # Forecast label
    fig.add_annotation(
        x=fo['ds'].iloc[-1], y=fo['yhat'].iloc[-1]/1e3,
        text="Forecast →", showarrow=False,
        font=dict(color=TEXT3, size=10), xshift=40
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
        yaxis_tickprefix='$', yaxis_ticksuffix='K',
        xaxis_tickformat='%b %y')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Metrics
    tp   = fc['yhat'].iloc[:len(seg_m)].values
    mae  = np.mean(np.abs(seg_m['y'].values - tp))
    rmse = np.sqrt(np.mean((seg_m['y'].values - tp)**2))
    prev = seg_m['y'].iloc[-1]
    m1v  = fo['yhat'].iloc[0]
    m3v  = fo['yhat'].iloc[-1]

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"MAE",     f"${mae/1e3:.1f}K",  "Mean Absolute Error",    "neu","📉","#3D52A0")
    kpi(c2,"RMSE",    f"${rmse/1e3:.1f}K", "Root Mean Sq. Error",    "neu","📈","#a371f7")
    kpi(c3,"Month +1",f"${m1v/1e3:.0f}K",  f"+{(m1v-prev)/prev*100:.1f}% vs prior", "up","↗","#3fb950")
    kpi(c4,"Month +3",f"${m3v/1e3:.0f}K",  f"+{(m3v-prev)/prev*100:.1f}% vs current","up","📊","#d29922")

    # Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='chart-card'><div class='chart-title'>Forecast Breakdown</div>", unsafe_allow_html=True)
    tbl = fo[['ds','yhat','yhat_lower','yhat_upper']].copy()
    tbl.columns = ['MONTH','FORECAST','LOWER BOUND','UPPER BOUND']
    tbl['MONTH'] = tbl['MONTH'].dt.strftime('%b %y')
    tbl['FORECAST']     = tbl['FORECAST'].apply(lambda x: f"${x/1e3:.0f}K")
    tbl['LOWER BOUND']  = tbl['LOWER BOUND'].apply(lambda x: f"${x/1e3:.0f}K")
    tbl['UPPER BOUND']  = tbl['UPPER BOUND'].apply(lambda x: f"${x/1e3:.0f}K")
    st.dataframe(tbl.set_index('MONTH'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALY REPORT
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔔  Anomaly Report":
    st.markdown(f"""
    <div class='page-hdr'>
        <h1>Anomaly Report</h1>
        <p>Isolation Forest + Z-Score detection on weekly sales data</p>
    </div>""", unsafe_allow_html=True)

    ws = weekly.set_index('Date')['Sales']
    iso = IsolationForest(contamination=0.07, random_state=42)
    iso_lbl = iso.fit_predict(ws.values.reshape(-1,1))
    iso_an  = ws[iso_lbl==-1]
    rm = ws.rolling(8, center=True).mean()
    rs = ws.rolling(8, center=True).std()
    z  = (ws - rm)/rs
    z_an = ws[z.abs()>2]
    both = len(set(iso_an.index.date) & set(z_an.index.date))

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"Isolation Forest", f"{len(iso_an)} flags","Anomalies Detected","down","⚠","#d29922")
    kpi(c2,"Z-Score Anomalies",f"{len(z_an)} flags", "Anomalies Detected","down","📉","#f85149")
    kpi(c3,"Consensus Flags",  f"{both} flags",       "Both Methods Agree", "up", "⚡","#a371f7")
    kpi(c4,"Weeks Analyzed",   f"{len(ws)}",          "Full Dataset",       "neu","📊","#3D52A0")

    st.markdown("<br>", unsafe_allow_html=True)

    # Method toggle
    st.markdown("<div class='det-label'>Detection Method:</div>", unsafe_allow_html=True)
    method = st.radio("det", ["Isolation Forest","Z-Score"], horizontal=True,
                       label_visibility='collapsed')
    anom = iso_an if method=="Isolation Forest" else z_an

    spikes = anom[anom > ws.mean()]
    drops  = anom[anom <= ws.mean()]

    # weekly index
    w_idx = np.arange(len(ws))
    w_labels = [f"W{i+1:02d}" for i in w_idx]

    st.markdown(f"""
    <div class='chart-card'>
        <div class='chart-title'>Weekly Sales — Anomaly Detection</div>
        <div class='chart-sub'>{method} model · 4-week rolling average overlay</div>
    """, unsafe_allow_html=True)

    rm4 = ws.rolling(4).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ws.index, y=ws.values/1e3,
        name='Weekly Revenue', line=dict(color=BLUE, width=1.5),
        fill='tozeroy', fillcolor='rgba(112,145,230,0.06)',
        hovertemplate='%{x|%b %y}<br>$%{y:.0f}K<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=ws.index, y=rm4/1e3,
        name='4-wk Rolling Avg', line=dict(color=YELLOW, width=1.2, dash='dash'),
        hovertemplate='4wk avg: $%{y:.0f}K<extra></extra>'
    ))
    if len(spikes):
        fig.add_trace(go.Scatter(
            x=spikes.index, y=spikes.values/1e3,
            name='Spike anomaly', mode='markers',
            marker=dict(symbol='triangle-up', size=12, color=GREEN,
                       line=dict(color=BG, width=1)),
            hovertemplate='SPIKE<br>%{x|%b %y}<br>$%{y:.0f}K<extra></extra>'
        ))
    if len(drops):
        fig.add_trace(go.Scatter(
            x=drops.index, y=drops.values/1e3,
            name='Drop anomaly', mode='markers',
            marker=dict(symbol='triangle-down', size=12, color=RED,
                       line=dict(color=BG, width=1)),
            hovertemplate='DROP<br>%{x|%b %y}<br>$%{y:.0f}K<extra></extra>'
        ))
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
        yaxis_tickprefix='$', yaxis_ticksuffix='K',
        xaxis_tickformat='%b %y')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Anomaly log table
    st.markdown(f"<div class='chart-card'><div class='chart-title'>Anomaly Log</div>", unsafe_allow_html=True)
    adf = anom.reset_index(); adf.columns=['Date','Sales']
    adf['Month'] = pd.to_datetime(adf['Date']).dt.month
    adf['SIGNAL'] = adf['Sales'].apply(lambda x:
        '<span class="badge-spike">Spike</span>' if x>ws.mean()
        else '<span class="badge-drop">Drop</span>')
    cause_map = {
        11:'Holiday season surge',12:'Christmas / year-end peak',
        1:'Post-holiday demand dip',2:'Post-holiday slowdown',
        3:'Promotional campaign activation',7:'Summer promotions',
        8:'Back-to-school demand', 9:'End-of-quarter procurement surge',
        6:'Regional supply disruption'
    }
    adf['CAUSE'] = adf['Month'].map(cause_map).fillna('Unusual demand fluctuation')
    adf['REVENUE'] = adf['Sales'].apply(lambda x: f"${x:,.0f}")
    adf['DATE'] = pd.to_datetime(adf['Date']).dt.strftime('%b W%V')
    display = adf[['DATE','REVENUE','SIGNAL','CAUSE']].sort_values('DATE', ascending=False)

    # Render as HTML table for badge styling
    rows = ""
    for _,r in display.iterrows():
        rows += f"""<tr>
            <td style='padding:8px 12px;color:{TEXT2};font-size:12px;border-bottom:1px solid {BORDER};'>{r['DATE']}</td>
            <td style='padding:8px 12px;color:{TEXT1};font-weight:600;font-size:12px;border-bottom:1px solid {BORDER};'>{r['REVENUE']}</td>
            <td style='padding:8px 12px;border-bottom:1px solid {BORDER};'>{r['SIGNAL']}</td>
            <td style='padding:8px 12px;color:{TEXT3};font-size:12px;border-bottom:1px solid {BORDER};font-family:monospace;'>{r['CAUSE']}</td>
        </tr>"""
    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;'>
        <thead><tr>
            <th style='text-align:left;padding:8px 12px;font-size:10px;color:{TEXT3};
                text-transform:uppercase;letter-spacing:0.8px;border-bottom:1px solid {BORDER};'>Date</th>
            <th style='text-align:left;padding:8px 12px;font-size:10px;color:{TEXT3};
                text-transform:uppercase;letter-spacing:0.8px;border-bottom:1px solid {BORDER};'>Revenue</th>
            <th style='text-align:left;padding:8px 12px;font-size:10px;color:{TEXT3};
                text-transform:uppercase;letter-spacing:0.8px;border-bottom:1px solid {BORDER};'>Signal</th>
            <th style='text-align:left;padding:8px 12px;font-size:10px;color:{TEXT3};
                text-transform:uppercase;letter-spacing:0.8px;border-bottom:1px solid {BORDER};'>Likely Cause</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — DEMAND SEGMENTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "🗂  Demand Segments":
    st.markdown(f"""
    <div class='page-hdr'>
        <h1>Demand Segments</h1>
        <p>K-Means clustering · 4 behavioral segments identified</p>
    </div>""", unsafe_allow_html=True)

    feat = df.groupby('Sub-Category').agg(
        Total_Sales=('Sales','sum'), Avg_Order=('Sales','mean'),
        Volatility=('Sales','std'),  Count=('Sales','count')
    ).reset_index()
    yoy_feat = df.groupby(['Sub-Category','Year'])['Sales'].sum().unstack()
    feat['Growth'] = yoy_feat.pct_change(axis=1).mean(axis=1).values
    feat = feat.dropna()

    Xs = StandardScaler().fit_transform(
        feat[['Total_Sales','Avg_Order','Volatility','Growth']])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    feat['Cluster'] = km.fit_predict(Xs)

    sc = feat.groupby('Cluster')['Total_Sales'].mean().sort_values(ascending=False).index
    lmp = {sc[0]:'High Volume Stable', sc[1]:'Growing Demand',
           sc[2]:'High Volatility',    sc[3]:'Declining Demand'}
    feat['Segment'] = feat['Cluster'].map(lmp)

    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(Xs)

    # Normalize to 0-100
    x_norm = (Xp[:,0]-Xp[:,0].min())/(Xp[:,0].max()-Xp[:,0].min())*100
    y_norm = (Xp[:,1]-Xp[:,1].min())/(Xp[:,1].max()-Xp[:,1].min())*100
    feat['px'] = x_norm
    feat['py'] = y_norm

    seg_colors = {
        'High Volume Stable': BLUE,
        'Growing Demand':     GREEN,
        'High Volatility':    YELLOW,
        'Declining Demand':   RED,
    }
    seg_strats = {
        'High Volume Stable':('Automate replenishment · maintain safety stock', BLUE),
        'Growing Demand':    ('Increase forecast buffer 15-20% · expand SKUs',   GREEN),
        'High Volatility':   ('Safety stock ×2 · weekly demand sensing',          YELLOW),
        'Declining Demand':  ('Reduce orders · markdown aging inventory',          RED),
    }
    seg_icons = {
        'High Volume Stable':'🔵','Growing Demand':'🟢',
        'High Volatility':'🟡','Declining Demand':'🔴'
    }

    left, right = st.columns([3,2])

    with left:
        st.markdown(f"""
        <div class='chart-card'>
            <div class='chart-title'>PCA Cluster Projection</div>
            <div class='chart-sub'>K-Means k=4 · X-axis: sales volume · Y-axis: demand stability</div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        for seg, grp in feat.groupby('Segment'):
            fig.add_trace(go.Scatter(
                x=grp['px'], y=grp['py'],
                mode='markers+text',
                name=seg,
                text=grp['Sub-Category'],
                textposition='top center',
                textfont=dict(size=9, color=TEXT2),
                marker=dict(size=14, color=seg_colors[seg],
                           line=dict(color=PANEL, width=1.5)),
                hovertemplate='<b>%{text}</b><br>Segment: '+seg+'<extra></extra>'
            ))
        fig.update_layout(**PLOTLY_LAYOUT, height=460,
            xaxis=dict(title='Sales Volume →', range=[-5,105],
                       gridcolor=BORDER, showgrid=True, zeroline=False,
                       tickvals=[0,25,50,75,100]),
            yaxis=dict(title='Stability ↑', range=[-5,105],
                       gridcolor=BORDER, showgrid=True, zeroline=False,
                       tickvals=[0,25,50,75,100]),
            legend=dict(orientation='h', yanchor='bottom', y=-0.18,
                       xanchor='left', x=0, font=dict(size=10, color=TEXT3)),
            margin=dict(l=50,r=20,t=20,b=60))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<br>", unsafe_allow_html=True)
        for seg in ['High Volume Stable','Growing Demand','High Volatility','Declining Demand']:
            grp = feat[feat['Segment']==seg]
            items = ' · '.join(sorted(grp['Sub-Category'].tolist()))
            color, strat_txt = seg_strats[seg][1], seg_strats[seg][0]
            icon = seg_icons[seg]
            # strat bg
            sbg = color.replace('#','')
            st.markdown(f"""
            <div class='seg-card' style='border-left-color:{color};'>
                <div class='seg-name'>
                    <span style='width:9px;height:9px;border-radius:50%;
                        background:{color};display:inline-block;'></span>
                    {seg}
                </div>
                <div class='seg-items'>{items}</div>
                <div class='seg-strat' style='background:{color}18;color:{color};'>
                    {strat_txt}
                </div>
            </div>""", unsafe_allow_html=True)
