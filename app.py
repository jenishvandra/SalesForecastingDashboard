import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
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

# ============ SALESIQ UI STYLING ============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

.stApp {
    background: #f0f2f5 !important;
}

/* Hide default elements */
#MainMenu, footer, header {
    visibility: hidden;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
    padding: 0 !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04) !important;
}

[data-testid="stSidebar"] * {
    color: #1a1a2e !important;
}

.sidebar-logo {
    padding: 24px 20px 16px 20px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 12px;
}
.sidebar-logo h1 {
    font-size: 22px;
    font-weight: 800;
    color: #1a1a2e;
    letter-spacing: -0.5px;
    margin: 0;
}
.sidebar-logo h1 span {
    color: #4a6cf7;
}
.sidebar-logo p {
    font-size: 11px;
    color: #8888a0;
    margin: 2px 0 0 0;
}

/* ── Navigation ── */
[data-testid="stSidebar"] .stRadio > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
    padding: 0 12px !important;
}

[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #4a4a6a !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: #f0f4ff !important;
    color: #4a6cf7 !important;
}

[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div,
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: #f0f4ff !important;
    color: #4a6cf7 !important;
    border-left: 3px solid #4a6cf7 !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] .stRadio input[type="radio"] {
    display: none !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] {
    display: none !important;
}

/* ── Sidebar Stats ── */
.sidebar-stats {
    padding: 0 16px;
    margin-top: 8px;
}
.sidebar-stats .label {
    font-size: 10px;
    color: #8888a0;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    padding-bottom: 8px;
}
.sidebar-stats .row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #f0f0f0;
}
.sidebar-stats .row:last-child {
    border-bottom: none;
}
.sidebar-stats .row .label-text {
    font-size: 11px;
    color: #8888a0;
}
.sidebar-stats .row .value {
    font-size: 12px;
    color: #1a1a2e;
    font-weight: 600;
}
.sidebar-stats .row .green {
    color: #10b981;
}

/* ── Page Header ── */
.page-header {
    margin-bottom: 24px;
}
.page-header .tag {
    font-size: 10px;
    color: #8888a0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 500;
    margin-bottom: 2px;
}
.page-header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
}
.page-header p {
    font-size: 14px;
    color: #8888a0;
    margin: 0;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 20px;
    transition: all 0.2s;
}
.kpi-card:hover {
    border-color: #4a6cf7;
    box-shadow: 0 2px 12px rgba(74, 108, 247, 0.08);
}
.kpi-card .label {
    font-size: 10px;
    color: #8888a0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 4px;
}
.kpi-card .value {
    font-size: 24px;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -0.5px;
}
.kpi-card .sub {
    font-size: 12px;
    margin-top: 2px;
}
.kpi-card .sub.up {
    color: #10b981;
}
.kpi-card .sub.down {
    color: #ef4444;
}
.kpi-card .sub.neutral {
    color: #4a6cf7;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #e5e7eb;
}
.section-header .icon {
    font-size: 16px;
}
.section-header .title {
    font-size: 14px;
    font-weight: 600;
    color: #1a1a2e;
}
.section-header .badge {
    font-size: 9px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    background: #f0f4ff;
    color: #4a6cf7;
    border: 1px solid #d6e0ff;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Quick Stats Footer ── */
.quick-stats {
    background: #ffffff;
    padding: 12px 20px;
    border-radius: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem 2.5rem;
    margin-top: 24px;
    border: 1px solid #e5e7eb;
    font-size: 13px;
    color: #4a4a6a;
}
.quick-stats strong {
    color: #1a1a2e;
    font-weight: 600;
}
.quick-stats .live {
    background: #10b981;
    color: white;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 600;
    margin-left: 6px;
}

/* ── Metric Boxes ── */
.metric-box {
    background: #ffffff;
    padding: 14px 18px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    text-align: center;
}
.metric-box .label {
    font-size: 10px;
    color: #8888a0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-box .value {
    font-size: 20px;
    font-weight: 700;
    color: #1a1a2e;
    margin-top: 2px;
}
.metric-box .sub {
    font-size: 11px;
    color: #10b981;
}

/* ── Cluster Cards ── */
.cluster-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 3px solid;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 6px 0;
    transition: background 0.2s;
}
.cluster-card:hover {
    background: #f8fafc;
}
.cluster-card .name {
    font-size: 13px;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 4px;
}
.cluster-card .items {
    font-size: 11px;
    color: #8888a0;
    margin-bottom: 5px;
    line-height: 1.5;
}
.cluster-card .strategy {
    font-size: 11px;
    color: #4a6cf7;
    font-weight: 500;
}

/* ── Select Inputs ── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    color: #1a1a2e !important;
    font-size: 13px !important;
}
.stMultiSelect > div > div {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div {
    background: #4a6cf7 !important;
}
.stSlider label {
    color: #4a4a6a !important;
    font-size: 12px !important;
}
.stRadio > div {
    gap: 4px !important;
}
.stRadio > div > label {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    font-size: 12px !important;
    color: #4a4a6a !important;
}

/* ── Tables ── */
[data-testid="stDataFrame"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
}

@media (max-width: 768px) {
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Year']       = df['Order Date'].dt.year
    df['Month']      = df['Order Date'].dt.month
    df['Quarter']    = df['Order Date'].dt.quarter
    df['Ship_Days']  = (df['Ship Date'] - df['Order Date']).dt.days
    return df

df = load_data()
monthly = df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
monthly.columns = ['Date','Sales']
weekly  = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
weekly.columns  = ['Date','Sales']

# ── CHART HELPERS ─────────────────────────────────────────────────────────
BG   = '#f8fafc'
PNL  = '#ffffff'
GRD  = '#e5e7eb'
TK   = '#8888a0'
C1,C2,C3,C4,C5 = '#4a6cf7','#6b8af4','#9bb0f7','#c5d2f9','#1a1a2e'
UP,DN = '#10b981','#ef4444'

def mfig(w=10, h=4, ncols=1, nrows=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor('white')
    alist = list(axes.flat) if hasattr(axes, 'flat') else [axes]
    for ax in alist:
        ax.set_facecolor('white')
        ax.tick_params(colors=TK, labelsize=8, length=0, pad=4)
        ax.xaxis.label.set_color(TK)
        ax.yaxis.label.set_color(TK)
        ax.title.set_color(C5)
        for s in ax.spines.values():
            s.set_edgecolor(GRD)
        ax.grid(axis='y', color=GRD, linewidth=0.6, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    return fig, axes

def finish(fig, tight=True):
    if tight:
        fig.tight_layout(pad=1.8)
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────
yearly_rev = df.groupby('Year')['Sales'].sum()
growth = (yearly_rev.iloc[-1]-yearly_rev.iloc[-2])/yearly_rev.iloc[-2]*100 if len(yearly_rev) >= 2 else 0

with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
        <h1>Sales<span>IQ</span></h1>
        <p>Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;color:#8888a0;text-transform:uppercase;letter-spacing:1px;padding:0 16px 6px;font-weight:600;'>Menu</div>", unsafe_allow_html=True)

    page = st.radio("nav", [
        "📊 Sales Overview",
        "🔮 Forecasting",
        "🚨 Anomaly Report",
        "📦 Demand Segments",
    ], label_visibility='collapsed')

    st.markdown("<hr style='margin:12px 16px;border-color:#e5e7eb;'/>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='sidebar-stats'>
        <div class='label'>Quick Stats</div>
        <div class='row'>
            <span class='label-text'>Revenue</span>
            <span class='value'>${df['Sales'].sum()/1e6:.2f}M</span>
        </div>
        <div class='row'>
            <span class='label-text'>Orders</span>
            <span class='value'>{len(df):,}</span>
        </div>
        <div class='row'>
            <span class='label-text'>YoY Growth</span>
            <span class='value green'>+{growth:.1f}%</span>
        </div>
        <div class='row'>
            <span class='label-text'>Avg Order</span>
            <span class='value'>${df['Sales'].mean():,.0f}</span>
        </div>
        <div class='row'>
            <span class='label-text'>Period</span>
            <span class='value'>2014–2017</span>
        </div>
        <div style='margin-top:12px;padding-top:12px;border-top:1px solid #f0f0f0;'>
            <span style='background:#10b981;color:white;padding:2px 12px;border-radius:20px;font-size:10px;font-weight:600;'>● LIVE</span>
            <span style='font-size:10px;color:#8888a0;margin-left:6px;'>Updated just now</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — SALES OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "📊 Sales Overview":
    st.markdown("""
    <div class='page-header'>
        <div class='tag'>Real-time Performance</div>
        <h1>Sales Overview</h1>
        <p>Real-time retail performance across categories, regions and time.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    total = df['Sales'].sum()
    orders = len(df)
    avg_ord = df['Sales'].mean()
    avg_shp = df['Ship_Days'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Total Revenue</div>
            <div class='value'>${total/1e6:.2f}M</div>
            <div class='sub up'>↑ {growth:.1f}% YoY</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Total Orders</div>
            <div class='value'>{orders:,}</div>
            <div class='sub neutral'>4 Years of Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Avg Order Value</div>
            <div class='value'>${avg_ord:,.0f}</div>
            <div class='sub neutral'>Per Transaction</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='label'>Avg Ship Time</div>
            <div class='value'>{avg_shp:.1f}d</div>
            <div class='sub neutral'>Order to Delivery</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class='kpi-card'>
            <div class='label'>Categories</div>
            <div class='value'>3</div>
            <div class='sub neutral'>Furn · Tech · Office</div>
        </div>
        """, unsafe_allow_html=True)

    # Monthly Revenue Trend
    st.markdown("""
    <div class='section-header'>
        <span class='icon'>📈</span>
        <span class='title'>Monthly Revenue Trend</span>
        <span class='badge'>4 Years</span>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = mfig(11, 4)
    x = monthly['Date']
    y = monthly['Sales']/1e3

    ax.fill_between(x, y, alpha=0.15, color=C2)
    ax.plot(x, y, color=C2, lw=2, zorder=3, label='Monthly Revenue')

    ma = monthly['Sales'].rolling(3).mean()/1e3
    ax.plot(x, ma, color=C3, lw=1.5, ls='--', alpha=0.8, label='3-mo avg')

    pidx = monthly['Sales'].idxmax()
    ax.scatter(x.iloc[pidx], y.iloc[pidx], color=UP, s=100, zorder=6, edgecolors='white', lw=2)
    ax.annotate(f"Peak ${y.iloc[pidx]:.0f}K",
                (x.iloc[pidx], y.iloc[pidx]),
                xytext=(6, 10), textcoords='offset points',
                color=UP, fontsize=9, fontweight='700')

    for yr in [2014, 2015, 2016, 2017]:
        start = pd.Timestamp(f'{yr}-01-01')
        end = pd.Timestamp(f'{yr}-12-31')
        if yr % 2 == 0:
            ax.axvspan(start, end, alpha=0.04, color=C1, zorder=0)
        ax.text(pd.Timestamp(f'{yr}-07-01'), ax.get_ylim()[0]+1,
                str(yr), ha='center', color=TK, fontsize=8, alpha=0.6)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color=TK)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
    ax.set_ylabel('')
    ax.legend(facecolor='white', edgecolor=GRD, labelcolor=TK, fontsize=8)
    finish(fig)
    st.pyplot(fig)

    # Bottom Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='section-header'>
            <span class='icon'>📅</span>
            <span class='title'>Revenue by Year</span>
        </div>
        """, unsafe_allow_html=True)
        
        yearly = df.groupby('Year')['Sales'].sum()
        fig2, ax2 = mfig(5, 3)
        bcolors = [C1, C2, C3, C4]
        bars = ax2.bar(yearly.index.astype(str), yearly.values/1e3,
                       color=bcolors, width=0.5, edgecolor='none', zorder=3)
        for bar, val in zip(bars, yearly.values):
            ax2.text(bar.get_x()+bar.get_width()/2, val/1e3+0.5,
                     f'${val/1e3:.0f}K', ha='center', va='bottom',
                     color=C5, fontsize=9, fontweight='600')
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
        finish(fig2)
        st.pyplot(fig2)

    with col2:
        st.markdown("""
        <div class='section-header'>
            <span class='icon'>🗂️</span>
            <span class='title'>Revenue by Category</span>
        </div>
        """, unsafe_allow_html=True)
        
        cat_sel = st.multiselect("", df['Category'].unique().tolist(),
                                 default=df['Category'].unique().tolist(),
                                 label_visibility='collapsed')
        cat = df[df['Category'].isin(cat_sel)].groupby('Category')['Sales'].sum().sort_values()
        fig3, ax3 = mfig(5, 3)
        colors = [C1, C2, C3]
        for i, (idx, val) in enumerate(cat.items()):
            ax3.barh(idx, val/1e3, color=colors[i%3], height=0.38, zorder=3)
            ax3.text(val/1e3+0.3, i, f'${val/1e3:.0f}K', va='center',
                     color=C5, fontsize=9, fontweight='600')
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
        finish(fig3)
        st.pyplot(fig3)

    # Region
    st.markdown("""
    <div class='section-header'>
        <span class='icon'>🌍</span>
        <span class='title'>Revenue by Region</span>
        <span class='badge'>Interactive</span>
    </div>
    """, unsafe_allow_html=True)
    
    reg_sel = st.multiselect("", df['Region'].unique().tolist(),
                              default=df['Region'].unique().tolist(),
                              label_visibility='collapsed')
    reg = df[df['Region'].isin(reg_sel)].groupby('Region')['Sales'].sum().sort_values()
    fig4, ax4 = mfig(11, 2.6)
    colors = [C1, C2, C3, C4]
    for i, (idx, val) in enumerate(reg.items()):
        ax4.barh(idx, val/1e3, color=colors[i%4], height=0.38, zorder=3)
        ax4.text(val/1e3+0.3, i, f'${val/1e3:.0f}K', va='center',
                 color=C5, fontsize=9, fontweight='600')
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
    finish(fig4)
    st.pyplot(fig4)

    # Quick Stats Footer
    st.markdown(f"""
    <div class='quick-stats'>
        <span><strong>Revenue:</strong> ${total/1e6:.2f}M</span>
        <span><strong>Orders:</strong> {orders:,}</span>
        <span><strong>YoY Growth:</strong> <span style='color:#10b981;'>+{growth:.1f}%</span></span>
        <span><strong>Avg Order:</strong> ${avg_ord:,.0f}</span>
        <span><strong>Period:</strong> 2014–2017</span>
        <span style='margin-left:auto;'>
            <span class='live'>● LIVE</span> Updated just now
        </span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecasting":
    st.markdown("""
    <div class='page-header'>
        <div class='tag'>Prophet Model</div>
        <h1>Forecast Explorer</h1>
        <p>Select a segment and horizon to generate an AI-powered demand forecast.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1: 
        seg_type = st.selectbox("Segment Type", ["Category", "Region"])
    with c2:
        opts = df['Category'].unique().tolist() if seg_type == "Category" else df['Region'].unique().tolist()
        seg_val = st.selectbox(f"Select {seg_type}", opts)
        seg_df = df[df['Category'] == seg_val] if seg_type == "Category" else df[df['Region'] == seg_val]
    with c3: 
        horizon = st.slider("Months", 1, 3, 3)

    seg_m = seg_df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    seg_m.columns = ['ds', 'y']

    with st.spinner("Running Prophet..."):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='additive')
        m.fit(seg_m)
        future = m.make_future_dataframe(periods=horizon, freq='ME')
        fc = m.predict(future)

    st.markdown("""
    <div class='section-header'>
        <span class='icon'>📈</span>
        <span class='title'>Sales Forecast</span>
        <span class='badge'>Prophet</span>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = mfig(11, 4)
    ax.plot(seg_m['ds'], seg_m['y']/1e3, color=C2, lw=2, label='Historical', zorder=3)
    ax.fill_between(seg_m['ds'], seg_m['y']/1e3, alpha=0.1, color=C2)

    fo = fc.tail(horizon)
    ax.plot(fo['ds'], fo['yhat']/1e3, color=C4, lw=2.2, marker='o',
            markersize=8, label='Forecast', zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    ax.fill_between(fo['ds'], fo['yhat_lower']/1e3, fo['yhat_upper']/1e3,
                    alpha=0.2, color=C4, label='95% Confidence')
    ax.axvline(seg_m['ds'].iloc[-1], color=C3, ls=':', lw=1.2, alpha=0.6)

    for _, row in fo.iterrows():
        ax.annotate(f"${row['yhat']/1e3:.1f}K",
                    (row['ds'], row['yhat']/1e3),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', color=C4, fontsize=8, fontweight='700')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color=TK)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
    ax.legend(facecolor='white', edgecolor=GRD, labelcolor=TK, fontsize=8)
    finish(fig)
    st.pyplot(fig)

    tp = fc['yhat'].iloc[:len(seg_m)].values
    mae = np.mean(np.abs(seg_m['y'].values - tp))
    rmse = np.sqrt(np.mean((seg_m['y'].values - tp)**2))

    st.markdown("""
    <div class='section-header'>
        <span class='icon'>📐</span>
        <span class='title'>Model Accuracy</span>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='label'>MAE</div>
            <div class='value'>${mae/1e3:.1f}K</div>
            <div class='sub'>Mean Abs. Error</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='label'>RMSE</div>
            <div class='value'>${rmse/1e3:.1f}K</div>
            <div class='sub'>Root Mean Sq. Error</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='label'>Month +1</div>
            <div class='value'>${fo['yhat'].iloc[0]/1e3:.1f}K</div>
            <div class='sub'>Next Month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m4:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='label'>Month +3</div>
            <div class='value'>${fo['yhat'].iloc[-1]/1e3:.1f}K</div>
            <div class='sub'>3 Months Out</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-header'>
        <span class='icon'>📋</span>
        <span class='title'>Forecast Table</span>
    </div>
    """, unsafe_allow_html=True)
    
    tbl = fo[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    tbl.columns = ['Month', 'Forecast', 'Lower Bound', 'Upper Bound']
    tbl['Month'] = tbl['Month'].dt.strftime('%B %Y')
    for c in ['Forecast', 'Lower Bound', 'Upper Bound']:
        tbl[c] = tbl[c].apply(lambda x: f"${x:,.0f}")
    st.dataframe(tbl.set_index('Month'), use_container_width=True)

    total = df['Sales'].sum()
    st.markdown(f"""
    <div class='quick-stats'>
        <span><strong>Revenue:</strong> ${total/1e6:.2f}M</span>
        <span><strong>Orders:</strong> {len(df):,}</span>
        <span><strong>YoY Growth:</strong> <span style='color:#10b981;'>+{growth:.1f}%</span></span>
        <span><strong>Avg Order:</strong> ${df['Sales'].mean():,.0f}</span>
        <span><strong>Period:</strong> 2014–2017</span>
        <span style='margin-left:auto;'>
            <span class='live'>● LIVE</span> Updated just now
        </span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALIES
# ══════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Report":
    st.markdown("""
    <div class='page-header'>
        <div class='tag'>Detection Engine</div>
        <h1>Anomaly Report</h1>
        <p>Isolation Forest + Z-S
