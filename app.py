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
    page_title="SalesIQ — Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ NEW UI STYLING (SalesIQ Style) ============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
.stApp { background: #f8fafc; }

/* Hide default elements */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] * { color: #1a1a2e !important; }

/* ── Navigation (SalesIQ Style) ── */
[data-testid="stSidebar"] .stRadio > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
    padding: 0 8px !important;
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
[data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"] { display: flex; }
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div,
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: #f0f4ff !important;
    color: #4a6cf7 !important;
    border-left: 3px solid #4a6cf7 !important;
    border-radius: 10px !important;
}

/* Hide radio circle */
[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] { display: none !important; }

/* ── Sidebar Header ── */
.sidebar-header {
    padding: 24px 20px 16px 20px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 12px;
}
.sidebar-logo {
    font-size: 22px;
    font-weight: 800;
    color: #1a1a2e;
    letter-spacing: -0.5px;
}
.sidebar-logo span { color: #4a6cf7; }
.sidebar-sub {
    font-size: 11px;
    color: #8888a0;
    margin-top: 2px;
}

/* ── Sidebar Stats ── */
.sidebar-stats {
    padding: 0 16px;
    margin-top: 8px;
}
.ss-label-stats {
    font-size: 10px;
    color: #8888a0;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    padding-bottom: 8px;
}
.ss-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #f0f0f0;
}
.ss-row:last-child { border-bottom: none; }
.ss-label { font-size: 11px; color: #8888a0; }
.ss-value { font-size: 12px; color: #1a1a2e; font-weight: 600; }
.ss-green { color: #10b981 !important; }

/* ── Inputs ── */
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
.stSlider > div > div > div { background: #4a6cf7 !important; }
.stSlider label { color: #4a4a6a !important; font-size: 12px !important; }
.stRadio > div { gap: 4px !important; }
.stRadio > div > label {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    font-size: 12px !important;
    color: #4a4a6a !important;
}

/* ── Text ── */
h1 {
    color: #1a1a2e !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.5px;
    margin-bottom: 2px !important;
}
h2 {
    color: #1a1a2e !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    margin-top: 20px !important;
}
p {
    color: #8888a0 !important;
    font-size: 13px !important;
}
hr { border-color: #e5e7eb !important; margin: 12px 0 !important; }

/* ── Cards ── */
.stat-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 20px;
    transition: border-color 0.2s, box-shadow 0.15s;
    height: 100%;
}
.stat-card:hover {
    border-color: #4a6cf7;
    box-shadow: 0 2px 8px rgba(74, 108, 247, 0.08);
}
.stat-label {
    font-size: 10px;
    color: #8888a0;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 6px;
}
.stat-value {
    font-size: 24px;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}
.stat-sub {
    font-size: 12px;
    color: #4a6cf7;
    font-weight: 500;
}
.stat-up { color: #10b981 !important; }
.stat-down { color: #ef4444 !important; }

/* ── Section Headers ── */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #e5e7eb;
}
.sec-icon { font-size: 15px; }
.sec-title {
    font-size: 13px;
    font-weight: 600;
    color: #1a1a2e;
}
.sec-badge {
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
.cluster-card:hover { background: #f8fafc; }
.c-name {
    font-size: 13px;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 4px;
}
.c-items {
    font-size: 11px;
    color: #8888a0;
    margin-bottom: 5px;
    line-height: 1.5;
}
.c-strat {
    font-size: 11px;
    color: #4a6cf7;
    font-weight: 500;
}

/* ── Page Tag ── */
.page-tag {
    font-size: 10px;
    color: #8888a0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 500;
    margin-bottom: 2px;
}

/* ── Quick Stats Footer ── */
.quick-stats {
    background: #ffffff;
    padding: 12px 20px;
    border-radius: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem 2.5rem;
    margin-top: 1.5rem;
    border: 1px solid #e5e7eb;
    font-size: 0.85rem;
    color: #4a4a6a;
}
.quick-stats strong { color: #1a1a2e; font-weight: 600; }
.live-badge {
    background: #10b981;
    color: white;
    padding: 2px 12px;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

/* ── Metric Boxes ── */
.metric-box {
    background: #ffffff;
    padding: 12px 16px;
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

/* ── Anomaly Badges ── */
.badge-spike {
    background: #fef3c7;
    color: #d97706;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}
.badge-drop {
    background: #fee2e2;
    color: #dc2626;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────
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

# ── Chart helpers (keeping same as original) ─────────────────────────────
BG   = '#f8fafc'
PNL  = '#ffffff'
GRD  = '#e5e7eb'
TK   = '#8888a0'
C1,C2,C3,C4,C5 = '#4a6cf7','#6b8af4','#9bb0f7','#c5d2f9','#1a1a2e'
UP,DN = '#10b981','#ef4444'

def mfig(w=10, h=4, ncols=1, nrows=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    alist = list(axes.flat) if hasattr(axes,'flat') else [axes]
    for ax in alist:
        ax.set_facecolor(PNL)
        ax.tick_params(colors=TK, labelsize=8, length=0, pad=4)
        ax.xaxis.label.set_color(TK)
        ax.yaxis.label.set_color(TK)
        ax.title.set_color(C5)
        for s in ax.spines.values(): s.set_edgecolor(GRD)
        ax.grid(axis='y', color=GRD, linewidth=0.6, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    return fig, axes

def finish(fig, tight=True):
    if tight: fig.tight_layout(pad=1.8)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────
yearly_rev = df.groupby('Year')['Sales'].sum()
growth = (yearly_rev.iloc[-1]-yearly_rev.iloc[-2])/yearly_rev.iloc[-2]*100 if len(yearly_rev) >= 2 else 0

with st.sidebar:
    st.markdown("""
    <div class='sidebar-header'>
        <div class='sidebar-logo'>Sales<span>IQ</span></div>
        <div class='sidebar-sub'>Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;color:#8888a0;text-transform:uppercase;letter-spacing:1px;padding:0 16px 6px;font-weight:600;'>Menu</div>", unsafe_allow_html=True)

    page = st.radio("nav", [
        "📊 Sales Overview",
        "🔮 Forecasting",
        "🚨 Anomaly Report",
        "📦 Demand Segments",
    ], label_visibility='collapsed')

    st.markdown("<hr style='margin:12px 16px;'/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='sidebar-stats'>
        <div class='ss-label-stats'>Quick Stats</div>
        <div class='ss-row'><span class='ss-label'>Revenue</span><span class='ss-value'>${df['Sales'].sum()/1e6:.2f}M</span></div>
        <div class='ss-row'><span class='ss-label'>Orders</span><span class='ss-value'>{len(df):,}</span></div>
        <div class='ss-row'><span class='ss-label'>YoY Growth</span><span class='ss-value ss-green'>+{growth:.1f}%</span></div>
        <div class='ss-row'><span class='ss-label'>Avg Order</span><span class='ss-value'>${df['Sales'].mean():,.0f}</span></div>
        <div class='ss-row'><span class='ss-label'>Period</span><span class='ss-value'>2014–2017</span></div>
        <div style='margin-top:10px;'>
            <span style='background:#10b981;color:white;padding:2px 12px;border-radius:20px;font-size:10px;font-weight:600;'>LIVE</span>
            <span style='font-size:10px;color:#8888a0;margin-left:6px;'>Updated just now</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "📊 Sales Overview":
    st.markdown("<div class='page-tag'>Real-time Performance</div>", unsafe_allow_html=True)
    st.title("Sales Overview")
    st.markdown("<p>Real-time retail performance across categories, regions and time.</p>", unsafe_allow_html=True)

    # KPI Cards
    total   = df['Sales'].sum()
    orders  = len(df)
    avg_ord = df['Sales'].mean()
    avg_shp = df['Ship_Days'].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,lbl,val,sub,up in [
        (c1,"Total Revenue",   f"${total/1e6:.2f}M",  f"↑ {growth:.1f}% YoY", True),
        (c2,"Total Orders",    f"{orders:,}",          "4 Years of Data",       False),
        (c3,"Avg Order Value", f"${avg_ord:,.0f}",     "Per Transaction",       False),
        (c4,"Avg Ship Time",   f"{avg_shp:.1f}d",      "Order to Delivery",     False),
        (c5,"Categories",      "3",                    "Furn · Tech · Office",  False),
    ]:
        with col:
            sc = "stat-up" if up else "stat-sub"
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>{lbl}</div>
                <div class='stat-value'>{val}</div>
                <div class='{sc}'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Monthly trend
    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📈</span>
        <span class='sec-title'>Monthly Revenue Trend</span>
        <span class='sec-badge'>4 Years</span>
    </div>""", unsafe_allow_html=True)

    fig, ax = mfig(11, 4)
    x = monthly['Date']
    y = monthly['Sales']/1e3

    ax.fill_between(x, y, alpha=0.15, color=C2)
    ax.plot(x, y, color=C2, lw=2, zorder=3)

    ma = monthly['Sales'].rolling(3).mean()/1e3
    ax.plot(x, ma, color=C3, lw=1.2, ls='--', alpha=0.7, label='3-mo avg')

    pidx = monthly['Sales'].idxmax()
    ax.scatter(x.iloc[pidx], y.iloc[pidx], color=UP, s=80, zorder=6, edgecolors=PNL, lw=1.5)
    ax.annotate(f" Peak ${y.iloc[pidx]:.0f}K",
                (x.iloc[pidx], y.iloc[pidx]),
                xytext=(6, 6), textcoords='offset points',
                color=UP, fontsize=8, fontweight='700')

    for yr in [2014,2015,2016,2017]:
        start = pd.Timestamp(f'{yr}-01-01')
        end   = pd.Timestamp(f'{yr}-12-31')
        if yr % 2 == 0:
            ax.axvspan(start, end, alpha=0.04, color=C1, zorder=0)
        ax.text(pd.Timestamp(f'{yr}-07-01'), ax.get_ylim()[0]+1,
                str(yr), ha='center', color=TK, fontsize=8, alpha=0.6)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color=TK)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
    ax.set_ylabel('')
    legend = ax.legend(facecolor=PNL, edgecolor=GRD, labelcolor=TK, fontsize=8)
    finish(fig)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class='section-hdr'>
            <span class='sec-icon'>📅</span><span class='sec-title'>Revenue by Year</span>
        </div>""", unsafe_allow_html=True)
        yearly = df.groupby('Year')['Sales'].sum()
        fig2, ax2 = mfig(5, 3)
        bcolors = [C1,C2,C3,C4]
        bars = ax2.bar(yearly.index.astype(str), yearly.values/1e3,
                       color=bcolors, width=0.5, edgecolor='none', zorder=3)
        for bar,val in zip(bars, yearly.values):
            ax2.text(bar.get_x()+bar.get_width()/2, val/1e3+0.5,
                     f'${val/1e3:.0f}K', ha='center', va='bottom',
                     color=C5, fontsize=9, fontweight='600')
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
        finish(fig2)
        st.pyplot(fig2)

    with col2:
        st.markdown("""<div class='section-hdr'>
            <span class='sec-icon'>🗂</span><span class='sec-title'>Revenue by Category</span>
        </div>""", unsafe_allow_html=True)
        cat_sel = st.multiselect("", df['Category'].unique().tolist(),
                                 default=df['Category'].unique().tolist(),
                                 label_visibility='collapsed')
        cat = df[df['Category'].isin(cat_sel)].groupby('Category')['Sales'].sum().sort_values()
        fig3, ax3 = mfig(5, 3)
        colors = [C1,C2,C3]
        for i,(idx,val) in enumerate(cat.items()):
            ax3.barh(idx, val/1e3, color=colors[i%3], height=0.38, zorder=3)
            ax3.text(val/1e3+0.3, i, f'${val/1e3:.0f}K', va='center',
                     color=C5, fontsize=9, fontweight='600')
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
        finish(fig3)
        st.pyplot(fig3)

    # Region (keeping original interaction)
    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>🌍</span><span class='sec-title'>Revenue by Region</span>
        <span class='sec-badge'>Interactive</span>
    </div>""", unsafe_allow_html=True)
    reg_sel = st.multiselect("", df['Region'].unique().tolist(),
                              default=df['Region'].unique().tolist(),
                              label_visibility='collapsed')
    reg = df[df['Region'].isin(reg_sel)].groupby('Region')['Sales'].sum().sort_values()
    fig4, ax4 = mfig(11, 2.6)
    colors = [C1,C2,C3,C4]
    for i,(idx,val) in enumerate(reg.items()):
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
            <span style="margin-left: auto;">
                <span class="live-badge">LIVE</span> Updated just now
            </span>
        </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FORECASTING (EXACTLY YOUR ORIGINAL LOGIC)
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecasting":
    st.markdown("<div class='page-tag'>Prophet Model</div>", unsafe_allow_html=True)
    st.title("Forecast Explorer")
    st.markdown("<p>Select a segment and horizon to generate an AI-powered demand forecast.</p>", unsafe_allow_html=True)

    # YOUR ORIGINAL FORECASTING LOGIC - EXACTLY SAME
    c1,c2,c3 = st.columns([2,2,1])
    with c1: seg_type = st.selectbox("Segment Type", ["Category","Region"])
    with c2:
        opts = df['Category'].unique().tolist() if seg_type=="Category" else df['Region'].unique().tolist()
        seg_val = st.selectbox(f"Select {seg_type}", opts)
        seg_df  = df[df['Category']==seg_val] if seg_type=="Category" else df[df['Region']==seg_val]
    with c3: horizon = st.slider("Months", 1, 3, 3)

    seg_m = seg_df.groupby(pd.Grouper(key='Order Date',freq='ME'))['Sales'].sum().reset_index()
    seg_m.columns = ['ds','y']

    with st.spinner("Running Prophet..."):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='additive')
        m.fit(seg_m)
        future = m.make_future_dataframe(periods=horizon, freq='ME')
        fc = m.predict(future)

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📈</span><span class='sec-title'>Sales Forecast</span>
        <span class='sec-badge'>Prophet</span>
    </div>""", unsafe_allow_html=True)

    # YOUR ORIGINAL CHART - EXACTLY SAME
    fig, ax = mfig(11, 4)
    ax.plot(seg_m['ds'], seg_m['y']/1e3, color=C2, lw=2, label='Historical', zorder=3)
    ax.fill_between(seg_m['ds'], seg_m['y']/1e3, alpha=0.1, color=C2)

    fo = fc.tail(horizon)
    ax.plot(fo['ds'], fo['yhat']/1e3, color=C4, lw=2.2, marker='o',
            markersize=8, label='Forecast', zorder=5, markeredgecolor=PNL, markeredgewidth=1.5)
    ax.fill_between(fo['ds'], fo['yhat_lower']/1e3, fo['yhat_upper']/1e3,
                    alpha=0.2, color=C4, label='95% Confidence')
    ax.axvline(seg_m['ds'].iloc[-1], color=C3, ls=':', lw=1.2, alpha=0.6)

    for _,row in fo.iterrows():
        ax.annotate(f"${row['yhat']/1e3:.1f}K",
                    (row['ds'], row['yhat']/1e3),
                    xytext=(0,10), textcoords='offset points',
                    ha='center', color=C4, fontsize=8, fontweight='700')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color=TK)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
    legend = ax.legend(facecolor=PNL, edgecolor=GRD, labelcolor=TK, fontsize=8)
    finish(fig)
    st.pyplot(fig)

    # YOUR ORIGINAL METRICS - EXACTLY SAME
    tp   = fc['yhat'].iloc[:len(seg_m)].values
    mae  = np.mean(np.abs(seg_m['y'].values - tp))
    rmse = np.sqrt(np.mean((seg_m['y'].values - tp)**2))

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📐</span><span class='sec-title'>Model Accuracy</span>
    </div>""", unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    for col,lbl,val,sub in [
        (m1,"MAE",         f"${mae/1e3:.1f}K",              "Mean Abs. Error"),
        (m2,"RMSE",        f"${rmse/1e3:.1f}K",             "Root Mean Sq. Error"),
        (m3,"Month +1",    f"${fo['yhat'].iloc[0]/1e3:.1f}K","Next Month"),
        (m4,"Month +3",    f"${fo['yhat'].iloc[-1]/1e3:.1f}K","3 Months Out"),
    ]:
        with col:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>{lbl}</div>
                <div class='stat-value' style='font-size:22px;'>{val}</div>
                <div class='stat-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    # YOUR ORIGINAL TABLE - EXACTLY SAME
    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📋</span><span class='sec-title'>Forecast Table</span>
    </div>""", unsafe_allow_html=True)
    tbl = fo[['ds','yhat','yhat_lower','yhat_upper']].copy()
    tbl.columns = ['Month','Forecast','Lower Bound','Upper Bound']
    tbl['Month'] = tbl['Month'].dt.strftime('%B %Y')
    for c in ['Forecast','Lower Bound','Upper Bound']:
        tbl[c] = tbl[c].apply(lambda x: f"${x:,.0f}")
    st.dataframe(tbl.set_index('Month'), use_container_width=True)

    # Quick Stats Footer
    total = df['Sales'].sum()
    st.markdown(f"""
        <div class='quick-stats'>
            <span><strong>Revenue:</strong> ${total/1e6:.2f}M</span>
            <span><strong>Orders:</strong> {len(df):,}</span>
            <span><strong>YoY Growth:</strong> <span style='color:#10b981;'>+{growth:.1f}%</span></span>
            <span><strong>Avg Order:</strong> ${df['Sales'].mean():,.0f}</span>
            <span><strong>Period:</strong> 2014–2017</span>
            <span style="margin-left: auto;">
                <span class="live-badge">LIVE</span> Updated just now
            </span>
        </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALIES (EXACTLY YOUR ORIGINAL LOGIC)
# ══════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Report":
    st.markdown("<div class='page-tag'>Detection Engine</div>", unsafe_allow_html=True)
    st.title("Anomaly Report")
    st.markdown("<p>Isolation Forest + Z-Score applied to 200+ weeks of sales data.</p>", unsafe_allow_html=True)

    # YOUR ORIGINAL ANOMALY DETECTION - EXACTLY SAME
    ws = weekly.set_index('Date')['Sales']
    iso = IsolationForest(contamination=0.07, random_state=42)
    iso_lbl = iso.fit_predict(ws.values.reshape(-1,1))
    iso_an  = ws[iso_lbl==-1]

    rm  = ws.rolling(8, center=True).mean()
    rs  = ws.rolling(8, center=True).std()
    z   = (ws - rm) / rs
    z_an = ws[z.abs() > 2]
    both = len(set(iso_an.index.date) & set(z_an.index.date))

    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val,sub,kind in [
        (c1,"Isolation Forest",   str(len(iso_an)), "Anomalies Found", "dn"),
        (c2,"Z-Score Method",     str(len(z_an)),   "Anomalies Found", "dn"),
        (c3,"Consensus",          str(both),         "Both Agree",      "up"),
        (c4,"Weeks Analyzed",     str(len(ws)),      "Full Dataset",    ""),
    ]:
        with col:
            sc = "stat-up" if kind=="up" else ("stat-down" if kind=="dn" else "stat-sub")
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>{lbl}</div>
                <div class='stat-value' style='font-size:22px;'>{val}</div>
                <div class='{sc}' style='font-size:12px;'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📉</span><span class='sec-title'>Weekly Sales + Anomalies</span>
        <span class='sec-badge'>Interactive</span>
    </div>""", unsafe_allow_html=True)

    # YOUR ORIGINAL METHOD SELECTOR
    method = st.radio("", ["🌲 Isolation Forest","📊 Z-Score (|z|>2)"], horizontal=True)
    anom   = iso_an if "Isolation" in method else z_an

    # YOUR ORIGINAL CHART - EXACTLY SAME
    fig, ax = mfig(11, 4)
    ax.fill_between(ws.index, ws.values/1e3, alpha=0.08, color=C2)
    ax.plot(ws.index, ws.values/1e3, color=C2, lw=1.5, alpha=0.9, label='Weekly Sales', zorder=2)

    rm_plot = ws.rolling(4).mean()/1e3
    ax.plot(ws.index, rm_plot, color=C3, lw=1.2, ls='--', alpha=0.6, label='4-wk avg')

    spikes = anom[anom > ws.mean()]
    drops  = anom[anom <= ws.mean()]
    if len(spikes):
        ax.scatter(spikes.index, spikes.values/1e3, color=UP, s=80, zorder=5,
                   marker='^', label='Spike', edgecolors=PNL, lw=1)
    if len(drops):
        ax.scatter(drops.index, drops.values/1e3, color=DN, s=80, zorder=5,
                   marker='v', label='Drop', edgecolors=PNL, lw=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color=TK)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}K'))
    legend = ax.legend(facecolor=PNL, edgecolor=GRD, labelcolor=TK, fontsize=8)
    finish(fig)
    st.pyplot(fig)

    # YOUR ORIGINAL ANOMALY LOG - EXACTLY SAME
    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📋</span><span class='sec-title'>Anomaly Log</span>
    </div>""", unsafe_allow_html=True)
    adf = anom.reset_index()
    adf.columns = ['Date','Sales']
    adf['Month'] = pd.to_datetime(adf['Date']).dt.month
    adf['Signal'] = adf['Sales'].apply(lambda x: '🔺 Spike' if x > ws.mean() else '🔻 Drop')
    cmap = {11:'Holiday season surge',12:'Christmas / year-end',
            1:'Post-holiday dip',2:'Post-holiday slowdown',
            7:'Summer promotions',8:'Back-to-school'}
    adf['Cause'] = adf['Month'].map(cmap).fillna('Unusual fluctuation')
    adf['Revenue'] = adf['Sales'].apply(lambda x: f"${x:,.0f}")
    adf = adf[['Date','Revenue','Signal','Cause']].sort_values('Date', ascending=False)
    st.dataframe(adf.set_index('Date'), use_container_width=True)

    # Quick Stats Footer
    total = df['Sales'].sum()
    st.markdown(f"""
        <div class='quick-stats'>
            <span><strong>Revenue:</strong> ${total/1e6:.2f}M</span>
            <span><strong>Orders:</strong> {len(df):,}</span>
            <span><strong>YoY Growth:</strong> <span style='color:#10b981;'>+{growth:.1f}%</span></span>
            <span><strong>Avg Order:</strong> ${df['Sales'].mean():,.0f}</span>
            <span><strong>Period:</strong> 2014–2017</span>
            <span style="margin-left: auto;">
                <span class="live-badge">LIVE</span> Updated just now
            </span>
        </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — SEGMENTS (EXACTLY YOUR ORIGINAL LOGIC)
# ══════════════════════════════════════════════════════════════════════════
elif page == "📦 Demand Segments":
    st.markdown("<div class='page-tag'>K-Means Clustering</div>", unsafe_allow_html=True)
    st.title("Demand Segments")
    st.markdown("<p>17 sub-categories segmented by volume, growth, volatility & order value.</p>", unsafe_allow_html=True)

    # YOUR ORIGINAL CLUSTERING - EXACTLY SAME
    feat = df.groupby('Sub-Category').agg(
        Total_Sales=('Sales','sum'), Avg_Order=('Sales','mean'),
        Volatility=('Sales','std'),  Count=('Sales','count')
    ).reset_index()
    yoy = df.groupby(['Sub-Category','Year'])['Sales'].sum().unstack()
    feat['Growth'] = yoy.pct_change(axis=1).mean(axis=1).values
    feat = feat.dropna()

    Xs = StandardScaler().fit_transform(feat[['Total_Sales','Avg_Order','Volatility','Growth']])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    feat['Cluster'] = km.fit_predict(Xs)

    sc  = feat.groupby('Cluster')['Total_Sales'].mean().sort_values(ascending=False).index
    lmp = {sc[0]:'🏆 High Volume, Stable', sc[1]:'📈 Growing Demand',
           sc[2]:'⚡ High Volatility',     sc[3]:'📉 Declining Demand'}
    feat['Segment'] = feat['Cluster'].map(lmp)

    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(Xs)
    feat['x'] = Xp[:,0]; feat['y'] = Xp[:,1]

    pal = {'🏆 High Volume, Stable':C2,'📈 Growing Demand':UP,
           '⚡ High Volatility':C4,    '📉 Declining Demand':DN}

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>🎯</span><span class='sec-title'>Cluster Map (PCA)</span>
        <span class='sec-badge'>2D Projection</span>
    </div>""", unsafe_allow_html=True)

    # YOUR ORIGINAL CLUSTER PLOT - EXACTLY SAME
    fig, ax = mfig(11, 5)
    ax.grid(axis='both', color=GRD, linewidth=0.5, linestyle='--', alpha=0.5)
    for seg, grp in feat.groupby('Segment'):
        ax.scatter(grp['x'], grp['y'], label=seg, color=pal[seg],
                   s=140, edgecolors=PNL, lw=1.5, alpha=0.95, zorder=4)
        for _,row in grp.iterrows():
            ax.annotate(row['Sub-Category'], (row['x'], row['y']),
                        fontsize=7.5, color=C5, ha='center', va='bottom',
                        xytext=(0,8), textcoords='offset points')
    ax.set_xlabel(f"PC1 — {pca.explained_variance_ratio_[0]*100:.0f}% variance", fontsize=9)
    ax.set_ylabel(f"PC2 — {pca.explained_variance_ratio_[1]*100:.0f}% variance", fontsize=9)
    legend = ax.legend(facecolor=PNL, edgecolor=GRD, labelcolor=TK, fontsize=9, loc='upper right')
    finish(fig)
    st.pyplot(fig)

    # YOUR ORIGINAL STRATEGY CARDS - EXACTLY SAME
    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📋</span><span class='sec-title'>Strategy per Segment</span>
    </div>""", unsafe_allow_html=True)

    strat = {
        '🏆 High Volume, Stable':  'Automate reorders. Maintain consistent safety stock. These never go out of demand.',
        '📈 Growing Demand':        'Increase procurement 15–20%. Prioritise shelf space. Monitor acceleration monthly.',
        '⚡ High Volatility':       'Use just-in-time inventory only. Avoid bulk orders. Demand is unpredictable.',
        '📉 Declining Demand':      'Reduce stock now. Run clearance promotions before Q4 peak season.',
    }
    bord = [C2, UP, C4, DN]

    col1,col2 = st.columns(2)
    for i,(seg,grp) in enumerate(feat.groupby('Segment')):
        items = ', '.join(sorted(grp['Sub-Category'].tolist()))
        bc    = bord[i%4]
        with (col1 if i%2==0 else col2):
            st.markdown(f"""
            <div class='cluster-card' style='border-left-color:{bc};'>
                <div class='c-name'>{seg}</div>
                <div class='c-items'>📦 {items}</div>
                <div class='c-strat'>📌 {strat[seg]}</div>
            </div>""", unsafe_allow_html=True)

    # Quick Stats Footer
    total = df['Sales'].sum()
    st.markdown(f"""
        <div class='quick-stats'>
            <span><strong>Revenue:</strong> ${total/1e6:.2f}M</span>
            <span><strong>Orders:</strong> {len(df):,}</span>
            <span><strong>YoY Growth:</strong> <span style='color:#10b981;'>+{growth:.1f}%</span></span>
            <span><strong>Avg Order:</strong> ${df['Sales'].mean():,.0f}</span>
            <span><strong>Period:</strong> 2014–2017</span>
            <span style="margin-left: auto;">
                <span class="live-badge">LIVE</span> Updated just now
            </span>
        </div>
    """, unsafe_allow_html=True)
