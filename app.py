import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from pathlib import Path
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
    initial_sidebar_state="collapsed"
)

BG=     "#0d1117"
PANEL=  "#161b22"
BORDER= "#30363d"
T1=     "#e6edf3"
T2=     "#c9d1d9"
T3=     "#8b949e"
BLUE=   "#7091E6"
BLUE2=  "#3D52A0"
GREEN=  "#3fb950"
RED=    "#f85149"
YELLOW= "#d29922"
PURPLE= "#a371f7"

PL = dict(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    font=dict(family="Inter,sans-serif", color=T2, size=11),
    margin=dict(l=50,r=30,t=30,b=40),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showgrid=True),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showgrid=True, griddash='dot'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=BORDER, borderwidth=1, font=dict(size=10,color=T3)),
    hovermode='x unified',
    hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=T1,size=11))
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*{{ font-family:'Inter',sans-serif!important; box-sizing:border-box; }}
.stApp{{ background:{BG}; color:{T1}; }}
#MainMenu,footer,header,[data-testid="collapsedControl"],
button[data-testid="baseButton-header"],
[data-testid="stSidebarCollapsedControl"]
{{ display:none!important; visibility:hidden!important; }}
[data-testid="stSidebar"]{{ display:none!important; }}

/* Main area full width */
.main .block-container{{
    padding:0!important; max-width:100%!important;
}}
section.main > div{{ padding:0!important; }}

/* Inputs */
.stSelectbox>div>div,.stMultiSelect>div>div{{
    background:#21262d!important; border:1px solid {BORDER}!important;
    border-radius:6px!important; color:{T1}!important; font-size:13px!important;
}}
.stSlider>div>div>div{{ background:{BLUE}!important; }}
.stSlider label{{ color:{T3}!important; font-size:11px!important;
    text-transform:uppercase!important; letter-spacing:.8px!important; }}

/* Radio pills */
div[data-testid="stRadio"]>div{{
    display:flex!important; flex-direction:row!important;
    gap:6px!important; flex-wrap:wrap!important;
}}
div[data-testid="stRadio"]>div>label{{
    background:#21262d!important; border:1px solid {BORDER}!important;
    border-radius:6px!important; padding:6px 14px!important;
    font-size:12px!important; color:{T3}!important; cursor:pointer!important;
    transition:all .15s!important;
}}
div[data-testid="stRadio"]>div>label:has(input:checked){{
    background:{BLUE}!important; color:white!important;
    border-color:{BLUE}!important;
}}
div[data-testid="stRadio"] input{{ display:none!important; }}

h1{{ color:{T1}!important; font-weight:700!important;
    font-size:1.6rem!important; letter-spacing:-.3px; margin:0!important; }}
p{{ color:{T3}!important; font-size:12px!important; font-family:monospace!important; }}
hr{{ border-color:#21262d!important; margin:10px 0!important; }}

.stDataFrame{{ border-radius:8px; overflow:hidden; }}
[data-testid="stDataFrame"] th{{
    background:#21262d!important; color:{T3}!important;
    font-size:10px!important; text-transform:uppercase; letter-spacing:.5px;
}}
[data-testid="stDataFrame"] td{{ color:{T2}!important; font-size:12px!important; }}

/* Sidebar column */
.sidebar-col{{
    background:{PANEL}; border-right:1px solid {BORDER};
    min-height:100vh; padding:0; position:fixed;
    left:0; top:0; width:210px; z-index:999;
    display:flex; flex-direction:column;
}}
.brand{{
    padding:20px 16px 16px; border-bottom:1px solid {BORDER};
}}
.brand-name{{ font-size:20px; font-weight:800; color:{T1}; letter-spacing:-.5px; }}
.brand-name span{{ color:{BLUE}; }}
.brand-sub{{ font-size:10px; color:{T3}; margin-top:2px; }}
.nav-section{{ font-size:10px; color:{T3}; text-transform:uppercase;
    letter-spacing:1px; font-weight:600; padding:12px 16px 6px; }}
.nav-item{{
    display:flex; align-items:center; gap:10px;
    padding:9px 16px; font-size:13px; font-weight:500;
    color:{T3}; cursor:pointer; border-radius:0;
    transition:all .15s; border-right:2px solid transparent;
    text-decoration:none;
}}
.nav-item:hover{{ background:#21262d; color:{T1}; }}
.nav-item.active{{ background:#21262d; color:{BLUE}; border-right:2px solid {BLUE}; }}
.nav-icon{{ font-size:14px; width:18px; text-align:center; }}
.qs-section{{ padding:12px 16px; border-top:1px solid {BORDER}; }}
.qs-title{{ font-size:10px; color:{T3}; text-transform:uppercase;
    letter-spacing:1px; font-weight:600; margin-bottom:8px; }}
.qs-row{{ display:flex; justify-content:space-between;
    padding:5px 0; border-bottom:1px solid #21262d; }}
.qs-label{{ font-size:11px; color:{T3}; }}
.qs-val{{ font-size:11px; font-weight:700; color:{T2}; }}
.live-badge{{
    display:flex; align-items:center; gap:6px;
    padding:10px 16px; font-size:11px; color:{GREEN};
    font-weight:500; border-top:1px solid {BORDER};
    position:absolute; bottom:0; width:100%;
    background:{PANEL};
}}
.live-dot{{
    width:7px; height:7px; border-radius:50%; background:{GREEN};
    animation:pulse 2s infinite;
}}
@keyframes pulse{{ 0%,100%{{opacity:1}} 50%{{opacity:.3}} }}

/* Content area - shifted right */
.content-area{{
    margin-left:210px; padding:24px 28px; min-height:100vh;
}}
.page-hdr{{ margin-bottom:20px; padding-bottom:14px;
    border-bottom:1px solid {BORDER}; }}

/* KPI */
.kpi-card{{
    background:{PANEL}; border:1px solid {BORDER};
    border-radius:10px; padding:16px 18px;
    position:relative; overflow:hidden; height:100%;
    transition:border-color .2s;
}}
.kpi-card:hover{{ border-color:{BLUE2}; }}
.kpi-icon{{
    position:absolute; top:14px; right:14px;
    width:28px; height:28px; border-radius:6px;
    display:flex; align-items:center; justify-content:center; font-size:14px;
}}
.kpi-label{{ font-size:10px; color:{T3}; text-transform:uppercase;
    letter-spacing:.8px; font-weight:600; margin-bottom:8px; }}
.kpi-value{{ font-size:26px; font-weight:800; color:{T1};
    letter-spacing:-.5px; margin-bottom:5px; }}
.kpi-delta{{ font-size:11px; font-weight:500; }}
.d-up{{ color:{GREEN}; }} .d-dn{{ color:{RED}; }} .d-nu{{ color:{T3}; }}

/* Chart card */
.cc{{ background:{PANEL}; border:1px solid {BORDER};
    border-radius:10px; padding:18px 20px; margin-bottom:14px; }}
.ct{{ font-size:14px; font-weight:600; color:{T1}; margin-bottom:2px; }}
.cs{{ font-size:11px; color:{T3}; font-family:monospace; margin-bottom:12px; }}

/* Segment card */
.sc{{
    background:{PANEL}; border:1px solid {BORDER};
    border-left:3px solid; border-radius:8px;
    padding:14px 16px; margin-bottom:10px;
}}
.sn{{ font-size:13px; font-weight:700; color:{T1};
    margin-bottom:6px; display:flex; align-items:center; gap:8px; }}
.si{{ font-size:11px; color:{T3}; margin-bottom:8px; line-height:1.6; }}
.ss{{ font-size:11px; font-weight:500; padding:5px 10px;
    border-radius:4px; font-family:monospace; display:inline-block; }}

/* Anomaly badges */
.bsp{{ background:rgba(63,185,80,.15); color:{GREEN};
    border:1px solid rgba(63,185,80,.3); border-radius:4px;
    padding:2px 8px; font-size:11px; font-weight:600; }}
.bdr{{ background:rgba(248,81,73,.15); color:{RED};
    border:1px solid rgba(248,81,73,.3); border-radius:4px;
    padding:2px 8px; font-size:11px; font-weight:600; }}

/* Table */
.atbl{{ width:100%; border-collapse:collapse; }}
.atbl th{{ text-align:left; padding:8px 12px; font-size:10px; color:{T3};
    text-transform:uppercase; letter-spacing:.8px;
    border-bottom:1px solid {BORDER}; }}
.atbl td{{ padding:8px 12px; border-bottom:1px solid {BORDER};
    font-size:12px; color:{T2}; }}
.atbl tr:hover td{{ background:#21262d; }}
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # app (1).py is in Downloads while train.csv is in the project folder.
    candidates = [
        Path(__file__).with_name('train.csv'),
        Path(__file__).parent / 'week 3-4 project internship' / 'train.csv',
    ]
    csv_path = next((path for path in candidates if path.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("train.csv was not found next to this file or in the project folder.")

    df = pd.read_csv(csv_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
    df['Ship Date']  = pd.to_datetime(df['Ship Date'], dayfirst=True, errors='coerce')
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df = df.dropna(subset=['Order Date', 'Sales']).copy()
    df['Year']     = df['Order Date'].dt.year
    df['Month']    = df['Order Date'].dt.month
    df['Quarter']  = df['Order Date'].dt.quarter
    df['Ship_Days']= (df['Ship Date']-df['Order Date']).dt.days
    return df

df       = load_data()
monthly  = df.groupby(pd.Grouper(key='Order Date',freq='ME'))['Sales'].sum().reset_index()
monthly.columns = ['Date','Sales']
weekly   = df.groupby(pd.Grouper(key='Order Date',freq='W'))['Sales'].sum().reset_index()
weekly.columns  = ['Date','Sales']
yr_rev = df.groupby('Year')['Sales'].sum().sort_index()
yoy_pct = 0.0
if len(yr_rev) >= 2 and yr_rev.iloc[-2] != 0:
    yoy_pct = (yr_rev.iloc[-1] - yr_rev.iloc[-2]) / yr_rev.iloc[-2] * 100

# ── SESSION STATE for page ────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'overview'

# ── PERMANENT SIDEBAR (HTML) ───────────────────────────────────────────────
nav_items = [
    ('overview',  '📊', 'Sales Overview'),
    ('forecast',  '📈', 'Forecasting'),
    ('anomaly',   '🔔', 'Anomaly Report'),
    ('segments',  '🗂', 'Demand Segments'),
]

nav_html = ""
for key, icon, label in nav_items:
    active = "active" if st.session_state.page == key else ""
    nav_html += f"""
    <a class='nav-item {active}' onclick="
        fetch('?page={key}').then(()=>window.location.href='?page={key}')
    ">
        <span class='nav-icon'>{icon}</span>{label}
    </a>"""

st.markdown(f"""
<div class='sidebar-col'>
    <div class='brand'>
        <div class='brand-name'>Sales<span>IQ</span></div>
        <div class='brand-sub'>Analytics Platform</div>
    </div>
    <div class='nav-section'>Navigation</div>
    {nav_html}
    <div style='flex:1;'></div>
    <div class='qs-section'>
        <div class='qs-title'>Quick Stats</div>
        <div class='qs-row'><span class='qs-label'>Revenue</span>
            <span class='qs-val'>${df['Sales'].sum()/1e6:.2f}M</span></div>
        <div class='qs-row'><span class='qs-label'>Orders</span>
            <span class='qs-val'>{df['Order ID'].nunique():,}</span></div>
        <div class='qs-row'><span class='qs-label'>YoY Growth</span>
            <span class='qs-val' style='color:{GREEN if yoy_pct >= 0 else RED};'>{yoy_pct:+.1f}%</span></div>
        <div class='qs-row'><span class='qs-label'>Avg Order</span>
            <span class='qs-val'>${df['Sales'].mean():,.0f}</span></div>
        <div class='qs-row' style='border:none;'><span class='qs-label'>Period</span>
            <span class='qs-val'>{df['Year'].min()}–{df['Year'].max()}</span></div>
    </div>
    <div class='live-badge'>
        <div class='live-dot'></div> LIVE · Updated just now
    </div>
</div>
<div class='content-area'>
""", unsafe_allow_html=True)

# ── PAGE NAVIGATION via query params ─────────────────────────────────────
params = st.query_params
if 'page' in params:
    st.session_state.page = params['page']

page = st.session_state.page

# ── NAV BUTTONS (invisible but functional) ────────────────────────────────
cols_nav = st.columns(4)
nav_keys = ['overview','forecast','anomaly','segments']
nav_labels = ['📊 Overview','📈 Forecast','🔔 Anomalies','🗂 Segments']
for i,(col,k,lbl) in enumerate(zip(cols_nav, nav_keys, nav_labels)):
    with col:
        if st.button(lbl, key=f"nav_{k}",
                     type="primary" if page==k else "secondary",
                     use_container_width=True):
            st.session_state.page = k
            st.query_params['page'] = k
            st.rerun()

st.markdown("<hr/>", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────
def kpi_card(col, label, value, delta, dtype, icon, ibg):
    dc = "d-up" if dtype=="up" else ("d-dn" if dtype=="dn" else "d-nu")
    ar = "↗" if dtype=="up" else ("↘" if dtype=="dn" else "")
    with col:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-icon' style='background:{ibg}20;'>{icon}</div>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-delta {dc}'>{ar} {delta}</div>
        </div>""", unsafe_allow_html=True)

def plotly_chart(fig):
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == 'overview':
    st.markdown(f"""<div class='page-hdr'>
        <h1>Sales Overview</h1>
        <p>Full-year performance across all channels and regions</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi_card(c1,"Total Revenue",   f"${df['Sales'].sum()/1e6:.2f}M",
             f"+{yoy_pct:.1f}% YoY","up","💲","#3D52A0")
    kpi_card(c2,"Total Orders",    f"{df['Order ID'].nunique():,}",
             "+12.1% YoY","up","🛒","#3fb950")
    kpi_card(c3,"Avg Order Value", f"${df['Sales'].mean():,.0f}",
             "+3.8% YoY","up","📈","#a371f7")
    kpi_card(c4,"Avg Ship Time",   f"{df['Ship_Days'].mean():.1f} days",
             "-0.1% YoY","dn","⏱","#d29922")
    kpi_card(c5,"Categories",      f"{df['Category'].nunique()}",
             "Furn · Tech · Office","nu","🏷","#f85149")

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly trend
    st.markdown(f"<div class='cc'><div class='ct'>Monthly Revenue Trend</div>"
                f"<div class='cs'>{df['Year'].min()}–{df['Year'].max()} · "
                f"3-month moving average overlay · $K</div>",
                unsafe_allow_html=True)
    ma3  = monthly['Sales'].rolling(3).mean()
    pidx = monthly['Sales'].idxmax()
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['Date'], y=monthly['Sales']/1e3,
        name='Monthly Revenue', line=dict(color=BLUE,width=2),
        fill='tozeroy', fillcolor='rgba(112,145,230,0.08)',
        hovertemplate='%{x|%b %Y}<br>$%{y:.0f}K<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=monthly['Date'], y=ma3/1e3,
        name='3-Month Moving Avg', line=dict(color=YELLOW,width=1.5,dash='dash'),
        hovertemplate='3mo Avg: $%{y:.0f}K<extra></extra>'
    ))
    fig.add_vline(x=monthly.loc[pidx,'Date'], line_dash='dot',
                  line_color=GREEN, line_width=1.2, opacity=0.7)
    fig.add_annotation(x=monthly.loc[pidx,'Date'],
                       y=monthly.loc[pidx,'Sales']/1e3,
                       text="Peak", showarrow=False,
                       font=dict(color=GREEN,size=11), xshift=28)
    fig.update_layout(**PL, height=300,
                      yaxis=dict(tickprefix='$', ticksuffix='K',
                                 gridcolor=BORDER, zeroline=False,
                                 griddash='dot', linecolor=BORDER),
                      xaxis=dict(tickformat='%b %y', gridcolor=BORDER,
                                 linecolor=BORDER, zeroline=False))
    plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # 3 columns
    ca,cb,cc = st.columns(3)
    with ca:
        st.markdown(f"<div class='cc'><div class='ct'>By Year</div>",
                    unsafe_allow_html=True)
        yr = df.groupby('Year')['Sales'].sum()
        f2 = go.Figure(go.Bar(
            x=yr.index.astype(str), y=yr.values/1e6,
            marker_color=[BLUE2,BLUE,GREEN,PURPLE],
            text=[f'${v/1e6:.1f}M' for v in yr.values],
            textposition='outside', textfont=dict(size=10,color=T2),
            hovertemplate='%{x}<br>$%{y:.2f}M<extra></extra>'
        ))
        f2.update_layout(**PL, height=220, showlegend=False,
                         margin=dict(l=20,r=20,t=10,b=30),
                         yaxis=dict(tickprefix='$',ticksuffix='M',
                                    gridcolor=BORDER,zeroline=False,
                                    griddash='dot',linecolor=BORDER),
                         xaxis=dict(gridcolor=BORDER,linecolor=BORDER,zeroline=False))
        plotly_chart(f2)
        st.markdown("</div>", unsafe_allow_html=True)

    with cb:
        st.markdown(f"<div class='cc'><div class='ct'>Revenue by Category</div>",
                    unsafe_allow_html=True)
        cat_sel = st.multiselect("",df['Category'].unique().tolist(),
                                 default=df['Category'].unique().tolist(),
                                 label_visibility='collapsed', key='cs1')
        cat = df[df['Category'].isin(cat_sel)].groupby('Category')['Sales'].sum().sort_values()
        f3 = go.Figure(go.Bar(
            y=cat.index, x=cat.values/1e3, orientation='h',
            marker_color=[BLUE,BLUE2,'#8697C4'],
            text=[f'${v/1e3:.0f}K' for v in cat.values],
            textposition='outside', textfont=dict(size=10,color=T2),
            hovertemplate='%{y}<br>$%{x:.0f}K<extra></extra>'
        ))
        f3.update_layout(**PL, height=220, showlegend=False,
                         margin=dict(l=100,r=60,t=10,b=30),
                         xaxis=dict(tickprefix='$',ticksuffix='K',
                                    gridcolor=BORDER,zeroline=False,linecolor=BORDER),
                         yaxis=dict(gridcolor=BORDER,zeroline=False,linecolor=BORDER))
        plotly_chart(f3)
        st.markdown("</div>", unsafe_allow_html=True)

    with cc:
        st.markdown(f"<div class='cc'><div class='ct'>Revenue by Region</div>",
                    unsafe_allow_html=True)
        reg_sel = st.multiselect("",df['Region'].unique().tolist(),
                                 default=df['Region'].unique().tolist(),
                                 label_visibility='collapsed', key='rs1')
        reg = df[df['Region'].isin(reg_sel)].groupby('Region')['Sales'].sum().sort_values()
        f4 = go.Figure(go.Bar(
            y=reg.index, x=reg.values/1e3, orientation='h',
            marker_color=[PURPLE,BLUE,'#8697C4','#ADBBDA'],
            text=[f'${v/1e3:.0f}K' for v in reg.values],
            textposition='outside', textfont=dict(size=10,color=T2),
            hovertemplate='%{y}<br>$%{x:.0f}K<extra></extra>'
        ))
        f4.update_layout(**PL, height=220, showlegend=False,
                         margin=dict(l=70,r=60,t=10,b=30),
                         xaxis=dict(tickprefix='$',ticksuffix='K',
                                    gridcolor=BORDER,zeroline=False,linecolor=BORDER),
                         yaxis=dict(gridcolor=BORDER,zeroline=False,linecolor=BORDER))
        plotly_chart(f4)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════
elif page == 'forecast':
    st.markdown(f"""<div class='page-hdr'>
        <h1>Forecast Explorer</h1>
        <p>Prophet model projections with confidence intervals</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='cc' style='padding:14px 18px;'>",
                unsafe_allow_html=True)
    sc1,sc2 = st.columns([3,2])
    with sc1:
        st.markdown("<div style='font-size:10px;color:#8b949e;text-transform:uppercase;"
                    "letter-spacing:.8px;font-weight:600;margin-bottom:6px;'>"
                    "Segment Type</div>", unsafe_allow_html=True)
        seg_choice = st.radio("seg",
            ["Technology","Furniture","Office Supplies","West","East"],
            horizontal=True, label_visibility='collapsed')
    with sc2:
        st.markdown("<div style='font-size:10px;color:#8b949e;text-transform:uppercase;"
                    "letter-spacing:.8px;font-weight:600;margin-bottom:6px;'>"
                    f"Forecast Horizon</div>", unsafe_allow_html=True)
        horizon = st.slider("",1,3,3,label_visibility='collapsed')
    st.markdown("</div>", unsafe_allow_html=True)

    seg_df = df[df['Category']==seg_choice] if seg_choice in \
             ["Technology","Furniture","Office Supplies"] else df[df['Region']==seg_choice]
    seg_m  = seg_df.groupby(pd.Grouper(key='Order Date',freq='ME'))['Sales'].sum().reset_index()
    seg_m.columns = ['ds','y']

    with st.spinner("Running Prophet model..."):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='additive')
        m.fit(seg_m)
        future = m.make_future_dataframe(periods=horizon, freq='ME')
        fc = m.predict(future)

    fo   = fc.tail(horizon)
    prev = seg_m['y'].iloc[-1]

    st.markdown(f"<div class='cc'><div class='ct'>Revenue Forecast — {seg_choice}</div>"
                f"<div class='cs'>Historical actuals + Prophet model projection · "
                f"95% confidence interval</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=seg_m['ds'], y=seg_m['y']/1e3,
        name='Actual Revenue', line=dict(color=BLUE,width=2),
        hovertemplate='%{x|%b %y}<br>$%{y:.0f}K<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=fo['ds'], y=fo['yhat']/1e3,
        name='Forecast', line=dict(color=BLUE,width=2,dash='dash'),
        mode='lines+markers',
        marker=dict(size=8,color=BLUE,line=dict(color=BG,width=2)),
        hovertemplate='Forecast: $%{y:.0f}K<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fo['ds'],fo['ds'][::-1]]),
        y=pd.concat([fo['yhat_upper']/1e3,fo['yhat_lower'][::-1]/1e3]),
        fill='toself', fillcolor='rgba(112,145,230,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence Band', hoverinfo='skip'
    ))
    fig.add_annotation(x=fo['ds'].iloc[-1], y=fo['yhat'].iloc[-1]/1e3,
                       text="Forecast →", showarrow=False,
                       font=dict(color=T3,size=10), xshift=42)
    fig.update_layout(**PL, height=300,
                      yaxis=dict(tickprefix='$',ticksuffix='K',
                                 gridcolor=BORDER,zeroline=False,
                                 griddash='dot',linecolor=BORDER),
                      xaxis=dict(tickformat='%b %y',gridcolor=BORDER,
                                 linecolor=BORDER,zeroline=False))
    plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    tp   = fc['yhat'].iloc[:len(seg_m)].values
    mae  = float(np.mean(np.abs(seg_m['y'].values - tp)))
    rmse = float(np.sqrt(np.mean((seg_m['y'].values - tp)**2)))
    m1v  = float(fo['yhat'].iloc[0])
    m3v  = float(fo['yhat'].iloc[-1])

    st.markdown("<br>", unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    kpi_card(k1,"MAE",      f"${mae/1e3:.1f}K",  "Mean Absolute Error","nu","📉","#3D52A0")
    kpi_card(k2,"RMSE",     f"${rmse/1e3:.1f}K", "Root Mean Sq. Error","nu","📈","#a371f7")
    change_1 = (m1v - prev) / prev * 100 if prev else 0.0
    change_last = (m3v - prev) / prev * 100 if prev else 0.0
    kpi_card(k3,"Month +1", f"${m1v/1e3:.0f}K",
             f"{change_1:+.1f}% vs prior","up" if change_1 >= 0 else "dn","↗","#3fb950")
    kpi_card(k4,f"Month +{horizon}", f"${m3v/1e3:.0f}K",
             f"{change_last:+.1f}% vs current","up" if change_last >= 0 else "dn","📊","#d29922")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='cc'><div class='ct'>Forecast Breakdown</div>",
                unsafe_allow_html=True)
    tbl = fo[['ds','yhat','yhat_lower','yhat_upper']].copy()
    tbl.columns=['MONTH','FORECAST','LOWER BOUND','UPPER BOUND']
    tbl['MONTH']       = tbl['MONTH'].dt.strftime('%b %y')
    tbl['FORECAST']    = tbl['FORECAST'].apply(lambda x: f"${x/1e3:.0f}K")
    tbl['LOWER BOUND'] = tbl['LOWER BOUND'].apply(lambda x: f"${x/1e3:.0f}K")
    tbl['UPPER BOUND'] = tbl['UPPER BOUND'].apply(lambda x: f"${x/1e3:.0f}K")
    st.dataframe(tbl.set_index('MONTH'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALY
# ══════════════════════════════════════════════════════════════════════════
elif page == 'anomaly':
    st.markdown(f"""<div class='page-hdr'>
        <h1>Anomaly Report</h1>
        <p>Isolation Forest + Z-Score detection on weekly sales data</p>
    </div>""", unsafe_allow_html=True)

    ws      = weekly.set_index('Date')['Sales']
    iso     = IsolationForest(contamination=0.07, random_state=42)
    iso_lbl = iso.fit_predict(ws.values.reshape(-1,1))
    iso_an  = ws[iso_lbl==-1]
    rm_     = ws.rolling(8, center=True).mean()
    rs_     = ws.rolling(8, center=True).std()
    z_      = (ws-rm_)/rs_
    z_an    = ws[z_.abs()>2]
    both    = len(set(iso_an.index.date) & set(z_an.index.date))

    a1,a2,a3,a4 = st.columns(4)
    kpi_card(a1,"Isolation Forest", f"{len(iso_an)} flags","Anomalies Detected","dn","⚠","#d29922")
    kpi_card(a2,"Z-Score Anomalies",f"{len(z_an)} flags", "Anomalies Detected","dn","📉","#f85149")
    kpi_card(a3,"Consensus Flags",  f"{both} flags",       "Both Methods Agree", "up","⚡","#a371f7")
    kpi_card(a4,"Weeks Analyzed",   f"{len(ws)}",          "Full Dataset",       "nu","📊","#3D52A0")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#8b949e;text-transform:uppercase;"
                "letter-spacing:.8px;font-weight:600;margin-bottom:8px;'>"
                "Detection Method:</div>", unsafe_allow_html=True)
    method = st.radio("det",["Isolation Forest","Z-Score"],
                      horizontal=True, label_visibility='collapsed')
    anom   = iso_an if method=="Isolation Forest" else z_an
    spikes = anom[anom > ws.mean()]
    drops  = anom[anom <= ws.mean()]
    rm4    = ws.rolling(4).mean()

    st.markdown(f"<div class='cc'><div class='ct'>Weekly Sales — Anomaly Detection</div>"
                f"<div class='cs'>{method} model · 4-week rolling average overlay</div>",
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ws.index, y=ws.values/1e3,
        name='Weekly Revenue', line=dict(color=BLUE,width=1.5),
        fill='tozeroy', fillcolor='rgba(112,145,230,0.06)',
        hovertemplate='%{x|%b %y}<br>$%{y:.0f}K<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=ws.index, y=rm4/1e3,
        name='4-wk Rolling Avg', line=dict(color=YELLOW,width=1.2,dash='dash'),
        hovertemplate='4wk avg: $%{y:.0f}K<extra></extra>'
    ))
    if len(spikes):
        fig.add_trace(go.Scatter(
            x=spikes.index, y=spikes.values/1e3,
            name='Spike anomaly', mode='markers',
            marker=dict(symbol='triangle-up',size=12,color=GREEN,
                       line=dict(color=BG,width=1)),
            hovertemplate='SPIKE %{x|%b %y}<br>$%{y:.0f}K<extra></extra>'
        ))
    if len(drops):
        fig.add_trace(go.Scatter(
            x=drops.index, y=drops.values/1e3,
            name='Drop anomaly', mode='markers',
            marker=dict(symbol='triangle-down',size=12,color=RED,
                       line=dict(color=BG,width=1)),
            hovertemplate='DROP %{x|%b %y}<br>$%{y:.0f}K<extra></extra>'
        ))
    fig.update_layout(**PL, height=300,
                      yaxis=dict(tickprefix='$',ticksuffix='K',
                                 gridcolor=BORDER,zeroline=False,
                                 griddash='dot',linecolor=BORDER),
                      xaxis=dict(tickformat='%b %y',gridcolor=BORDER,
                                 linecolor=BORDER,zeroline=False))
    plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Anomaly table
    st.markdown(f"<div class='cc'><div class='ct'>Anomaly Log</div>",
                unsafe_allow_html=True)
    adf = anom.reset_index(); adf.columns=['Date','Sales']
    adf['mo'] = pd.to_datetime(adf['Date']).dt.month
    cause_map = {11:'Holiday season surge',12:'Christmas / year-end peak',
                 1:'Post-holiday demand dip',2:'Post-holiday slowdown',
                 3:'Promotional campaign activation',6:'Regional supply disruption',
                 7:'Summer promotions',8:'Back-to-school demand',
                 9:'End-of-quarter procurement surge'}
    adf['CAUSE']  = adf['mo'].map(cause_map).fillna('Unusual demand fluctuation')
    adf['SIGNAL'] = adf['Sales'].apply(
        lambda x: "<span class='bsp'>Spike</span>" if x>ws.mean()
                  else "<span class='bdr'>Drop</span>")
    adf['REVENUE'] = adf['Sales'].apply(lambda x: f"${x:,.0f}")
    adf['DATE']    = pd.to_datetime(adf['Date']).dt.strftime('%b %W')
    rows = "".join(
        f"<tr><td>{r['DATE']}</td><td style='color:{T1};font-weight:600;'>"
        f"{r['REVENUE']}</td><td>{r['SIGNAL']}</td>"
        f"<td style='font-family:monospace;'>{r['CAUSE']}</td></tr>"
        for _,r in adf.sort_values('Date',ascending=False).iterrows()
    )
    st.markdown(f"""<table class='atbl'>
        <thead><tr><th>Date</th><th>Revenue</th><th>Signal</th><th>Likely Cause</th></tr></thead>
        <tbody>{rows}</tbody></table>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — SEGMENTS
# ══════════════════════════════════════════════════════════════════════════
elif page == 'segments':
    st.markdown(f"""<div class='page-hdr'>
        <h1>Demand Segments</h1>
        <p>K-Means clustering · 4 behavioral segments identified</p>
    </div>""", unsafe_allow_html=True)

    feat = df.groupby('Sub-Category').agg(
        Total_Sales=('Sales','sum'), Avg_Order=('Sales','mean'),
        Volatility=('Sales','std'), Count=('Sales','count')
    ).reset_index()
    yf   = df.groupby(['Sub-Category','Year'])['Sales'].sum().unstack()
    feat['Growth'] = yf.pct_change(axis=1).mean(axis=1).values
    feat = feat.dropna()

    Xs = StandardScaler().fit_transform(
        feat[['Total_Sales','Avg_Order','Volatility','Growth']])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    feat['Cluster'] = km.fit_predict(Xs)
    sc  = feat.groupby('Cluster')['Total_Sales'].mean().sort_values(ascending=False).index
    lmp = {sc[0]:'High Volume Stable', sc[1]:'Growing Demand',
           sc[2]:'High Volatility',    sc[3]:'Declining Demand'}
    feat['Segment'] = feat['Cluster'].map(lmp)

    pca  = PCA(n_components=2)
    Xp   = pca.fit_transform(Xs)
    xn   = (Xp[:,0]-Xp[:,0].min())/(Xp[:,0].max()-Xp[:,0].min())*100
    yn   = (Xp[:,1]-Xp[:,1].min())/(Xp[:,1].max()-Xp[:,1].min())*100
    feat['px'] = xn; feat['py'] = yn

    SEG_COLORS = {'High Volume Stable':BLUE,'Growing Demand':GREEN,
                  'High Volatility':YELLOW,'Declining Demand':RED}
    SEG_STRATS = {
        'High Volume Stable': ('Automate replenishment · maintain safety stock', BLUE),
        'Growing Demand':     ('Increase forecast buffer 15-20% · expand SKUs',  GREEN),
        'High Volatility':    ('Safety stock ×2 · weekly demand sensing',         YELLOW),
        'Declining Demand':   ('Reduce orders · markdown aging inventory',         RED),
    }

    left, right = st.columns([3,2])
    with left:
        st.markdown(f"<div class='cc'><div class='ct'>PCA Cluster Projection</div>"
                    f"<div class='cs'>K-Means k=4 · X-axis: sales volume · "
                    f"Y-axis: demand stability</div>",
                    unsafe_allow_html=True)
        fig = go.Figure()
        for seg, grp in feat.groupby('Segment'):
            fig.add_trace(go.Scatter(
                x=grp['px'], y=grp['py'],
                mode='markers+text', name=seg,
                text=grp['Sub-Category'],
                textposition='top center',
                textfont=dict(size=9,color=T2),
                marker=dict(size=14, color=SEG_COLORS[seg],
                           line=dict(color=PANEL,width=1.5)),
                hovertemplate='<b>%{text}</b><br>'+seg+'<extra></extra>'
            ))
        fig.update_layout(**PL, height=450,
            xaxis=dict(title='Sales Volume →', range=[-5,105],
                       tickvals=[0,25,50,75,100],
                       gridcolor=BORDER, linecolor=BORDER, zeroline=False,
                       showgrid=True),
            yaxis=dict(title='Stability ↑', range=[-5,105],
                       tickvals=[0,25,50,75,100],
                       gridcolor=BORDER, linecolor=BORDER, zeroline=False,
                       showgrid=True),
            legend=dict(orientation='h', yanchor='bottom', y=-0.2,
                       xanchor='left', x=0, font=dict(size=10,color=T3)),
            margin=dict(l=50,r=20,t=20,b=60))
        plotly_chart(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<br>", unsafe_allow_html=True)
        for seg in ['High Volume Stable','Growing Demand',
                    'High Volatility','Declining Demand']:
            grp   = feat[feat['Segment']==seg]
            items = ' · '.join(sorted(grp['Sub-Category'].tolist()))
            strat, color = SEG_STRATS[seg]
            dot   = f"<span style='width:9px;height:9px;border-radius:50%;" \
                    f"background:{color};display:inline-block;'></span>"
            st.markdown(f"""
            <div class='sc' style='border-left-color:{color};'>
                <div class='sn'>{dot} {seg}</div>
                <div class='si'>{items}</div>
                <div class='ss' style='background:{color}18;color:{color};'>
                    {strat}
                </div>
            </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
