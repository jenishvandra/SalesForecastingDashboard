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

# ══════════════════════════════════════════════════════════════════════
# THEME & STYLING
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { 
    font-family: 'Inter', sans-serif !important; 
    box-sizing: border-box; 
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
    color: #e6edf3 !important;
}

.stApp { 
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
    color: #e6edf3 !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #21262d 100%) !important;
    border-right: 1px solid #30363d !important;
}

[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Radio buttons as navigation */
[data-testid="stSidebar"] .stRadio > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
}

[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #8b949e !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    display: flex !important;
    align-items: center !important;
}

[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(112, 145, 230, 0.1) !important;
    color: #7091E6 !important;
    border-color: #7091E6 !important;
}

[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(112, 145, 230, 0.15) !important;
    color: #7091E6 !important;
    border-left: 3px solid #7091E6 !important;
    border-color: #7091E6 !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] { display: none !important; }

/* ── INPUTS & CONTROLS ── */
.stSelectbox > div > div {
    background: rgba(33, 38, 45, 0.8) !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 13px !important;
}

.stMultiSelect > div > div {
    background: rgba(33, 38, 45, 0.8) !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

.stSlider > div > div > div { 
    background: linear-gradient(90deg, #7091E6, #3fb950) !important; 
}

.stSlider label { 
    color: #8b949e !important; 
    font-size: 12px !important;
    font-weight: 500 !important;
}

.stRadio > div { gap: 6px !important; }
.stRadio > div > label {
    background: rgba(33, 38, 45, 0.6) !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    font-size: 12px !important;
    color: #8b949e !important;
    transition: all 0.15s !important;
}

.stRadio > div > label:hover {
    background: rgba(33, 38, 45, 1) !important;
    border-color: #7091E6 !important;
}

/* ── TEXT ── */
h1 { 
    color: #e6edf3 !important; 
    font-weight: 800 !important; 
    font-size: 2rem !important; 
    letter-spacing: -0.5px !important;
    margin-bottom: 8px !important;
}

h2 { 
    color: #c9d1d9 !important; 
    font-weight: 600 !important; 
    font-size: 1.1rem !important; 
    margin-top: 24px !important;
    margin-bottom: 16px !important;
}

p { 
    color: #8b949e !important; 
    font-size: 13px !important;
    line-height: 1.6 !important;
}

hr { 
    border-color: #21262d !important; 
    margin: 16px 0 !important;
}

/* ── KPI CARDS ── */
.stat-card {
    background: linear-gradient(135deg, rgba(22, 27, 34, 0.8), rgba(33, 38, 45, 0.6)) !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    height: 100% !important;
    backdrop-filter: blur(10px) !important;
}

.stat-card:hover { 
    border-color: #7091E6 !important;
    transform: translateY(-4px) !important;
    box-shadow: 0 8px 24px rgba(112, 145, 230, 0.15) !important;
}

.stat-label { 
    font-size: 10px !important; 
    color: #8b949e !important; 
    text-transform: uppercase !important; 
    letter-spacing: 1px !important; 
    font-weight: 600 !important; 
    margin-bottom: 10px !important;
}

.stat-value { 
    font-size: 28px !important; 
    font-weight: 800 !important; 
    color: #e6edf3 !important; 
    letter-spacing: -1px !important; 
    margin-bottom: 6px !important;
}

.stat-sub { 
    font-size: 12px !important; 
    color: #7091E6 !important; 
    font-weight: 500 !important;
}

.stat-up { color: #3fb950 !important; }
.stat-down { color: #f85149 !important; }

/* ── SECTION HEADERS ── */
.section-hdr {
    display: flex !important; 
    align-items: center !important; 
    gap: 12px !important;
    margin: 28px 0 18px 0 !important; 
    padding-bottom: 12px !important;
    border-bottom: 1.5px solid #21262d !important;
}

.sec-icon { 
    font-size: 16px !important; 
}

.sec-title { 
    font-size: 14px !important; 
    font-weight: 600 !important; 
    color: #c9d1d9 !important;
}

.sec-badge {
    font-size: 9px !important; 
    font-weight: 700 !important; 
    padding: 4px 10px !important;
    border-radius: 20px !important; 
    background: rgba(112, 145, 230, 0.15) !important; 
    color: #7091E6 !important;
    border: 1px solid rgba(112, 145, 230, 0.3) !important; 
    text-transform: uppercase !important; 
    letter-spacing: 0.5px !important;
}

/* ── CLUSTER CARDS ── */
.cluster-card {
    background: rgba(22, 27, 34, 0.7) !important;
    border: 1px solid #30363d !important;
    border-left: 3px solid !important;
    border-radius: 10px !important;
    padding: 16px !important; 
    margin: 8px 0 !important; 
    transition: all 0.2s !important;
    backdrop-filter: blur(5px) !important;
}

.cluster-card:hover { 
    background: rgba(33, 38, 45, 0.8) !important;
    border-color: #7091E6 !important;
}

.c-name { 
    font-size: 13px !important; 
    font-weight: 700 !important; 
    color: #e6edf3 !important; 
    margin-bottom: 6px !important;
}

.c-items { 
    font-size: 11px !important; 
    color: #8b949e !important; 
    margin-bottom: 8px !important; 
    line-height: 1.6 !important;
}

.c-strat { 
    font-size: 11px !important; 
    color: #7091E6 !important; 
    font-weight: 500 !important;
}

/* ── SIDEBAR STATS ── */
.sidebar-stats { padding: 4px; }

.ss-row {
    display: flex !important; 
    justify-content: space-between !important; 
    align-items: center !important;
    padding: 10px 0 !important; 
    border-bottom: 1px solid #21262d !important;
}

.ss-label { 
    font-size: 11px !important; 
    color: #8b949e !important;
    font-weight: 500 !important;
}

.ss-value { 
    font-size: 12px !important; 
    color: #e6edf3 !important; 
    font-weight: 700 !important;
}

.ss-green { color: #3fb950 !important; }

/* ── PAGE TAG ── */
.page-tag {
    font-size: 10px !important; 
    color: #8b949e !important; 
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important; 
    font-weight: 600 !important; 
    margin-bottom: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# DATA LOADING & PROCESSING
# ══════════════════════════════════════════════════════════════════════
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
weekly = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
weekly.columns = ['Date','Sales']

# ══════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ══════════════════════════════════════════════════════════════════════
BG   = '#0d1117'
PNL  = '#161b22'
GRD  = '#21262d'
TK   = '#8b949e'
C1,C2,C3,C4,C5 = '#3D52A0','#7091E6','#8697C4','#ADBBDA','#EDE8F5'
UP,DN = '#3fb950','#f85149'

# ══════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════
def mfig(w=10, h=4, ncols=1, nrows=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    alist = list(axes.flat) if hasattr(axes, 'flat') else [axes]
    for ax in alist:
        ax.set_facecolor(PNL)
        ax.tick_params(colors=TK, labelsize=9, length=0, pad=5)
        ax.xaxis.label.set_color(TK)
        ax.yaxis.label.set_color(TK)
        ax.title.set_color(C5)
        for s in ax.spines.values():
            s.set_edgecolor(GRD)
            s.set_linewidth(0.8)
        ax.grid(axis='y', color=GRD, linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
    return fig, axes

def finish(fig, tight=True):
    if tight:
        fig.tight_layout(pad=2)
    return fig

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
yearly_rev = df.groupby('Year')['Sales'].sum()
growth = (yearly_rev.iloc[-1] - yearly_rev.iloc[-2]) / yearly_rev.iloc[-2] * 100

with st.sidebar:
    st.markdown("""
    <div style='padding:24px 16px 20px; border-bottom: 1.5px solid #30363d; margin-bottom: 16px;'>
        <div style='font-size: 22px; font-weight: 800; color: #e6edf3; letter-spacing: -1px;'>
            Sales<span style='color: #7091E6;'>IQ</span>
        </div>
        <div style='font-size: 12px; color: #8b949e; margin-top: 4px; font-weight: 500;'>
            📊 Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; padding: 0 4px 8px; font-weight: 700;'>NAVIGATION</div>", unsafe_allow_html=True)

    page = st.radio("nav", [
        "📊 Sales Overview",
        "🔮 Forecast Explorer",
        "🚨 Anomaly Report",
        "📦 Demand Segments",
    ], label_visibility='collapsed')

    st.markdown("<hr style='margin: 16px 0;'/>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='sidebar-stats'>
        <div style='font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; padding-bottom: 10px;'>QUICK STATS</div>
        <div class='ss-row'><span class='ss-label'>💰 Revenue</span><span class='ss-value'>${df['Sales'].sum()/1e6:.2f}M</span></div>
        <div class='ss-row'><span class='ss-label'>📦 Orders</span><span class='ss-value'>{len(df):,}</span></div>
        <div class='ss-row'><span class='ss-label'>📈 YoY Growth</span><span class='ss-value ss-green'>+{growth:.1f}%</span></div>
        <div class='ss-row'><span class='ss-label'>💵 Avg Order</span><span class='ss-value'>${df['Sales'].mean():,.0f}</span></div>
        <div class='ss-row' style='border: none;'><span class='ss-label'>📅 Period</span><span class='ss-value'>{df['Order Date'].dt.year.min()}–{df['Order Date'].dt.year.max()}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 16px 0;'/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 11px; color: #8b949e; margin-top: 16px;'>✨ Built with Streamlit, Prophet & Scikit-learn</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1: SALES OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == "📊 Sales Overview":
    st.markdown("<div class='page-tag'>XYlofy AI · Analytics Dashboard</div>", unsafe_allow_html=True)
    st.title("Sales Overview")
    st.markdown("<p>Real-time retail performance across categories, regions and time periods.</p>", unsafe_allow_html=True)

    # KPIs
    total = df['Sales'].sum()
    orders = len(df)
    avg_ord = df['Sales'].mean()
    avg_shp = df['Ship_Days'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    
    kpis = [
        (c1, "Total Revenue", f"${total/1e6:.2f}M", f"↑ {growth:.1f}% YoY", True, "💰"),
        (c2, "Total Orders", f"{orders:,}", f"Full Dataset", False, "📦"),
        (c3, "Avg Order Value", f"${avg_ord:,.0f}", f"Per Transaction", False, "💵"),
        (c4, "Avg Ship Time", f"{avg_shp:.1f}d", f"Order to Delivery", False, "⏱️"),
        (c5, "Categories", "3", f"Tech · Furn · Office", False, "🏷️"),
    ]
    
    for col, lbl, val, sub, is_up, icon in kpis:
        with col:
            sc = "stat-up" if is_up else "stat-sub"
            st.markdown(f"""<div class='stat-card'>
                <div style='font-size: 18px; margin-bottom: 8px;'>{icon}</div>
                <div class='stat-label'>{lbl}</div>
                <div class='stat-value'>{val}</div>
                <div class='{sc}'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Monthly Revenue Trend
    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📈</span>
        <span class='sec-title'>Monthly Revenue Trend</span>
        <span class='sec-badge'>Interactive</span>
    </div>""", unsafe_allow_html=True)

    fig, ax = mfig(12, 4.5)
    x = monthly['Date']
    y = monthly['Sales'] / 1e3

    ax.fill_between(x, y, alpha=0.12, color=C2)
    ax.plot(x, y, color=C2, lw=2.5, zorder=3, label='Revenue')

    ma = monthly['Sales'].rolling(3).mean() / 1e3
    ax.plot(x, ma, color=C4, lw=1.5, ls='--', alpha=0.6, label='3-mo avg', zorder=2)

    pidx = monthly['Sales'].idxmax()
    ax.scatter(x.iloc[pidx], y.iloc[pidx], color=UP, s=100, zorder=6, edgecolors=BG, lw=2)
    ax.annotate(f"Peak ${y.iloc[pidx]:.0f}K",
                (x.iloc[pidx], y.iloc[pidx]),
                xytext=(8, 12), textcoords='offset points',
                color=UP, fontsize=9, fontweight='700',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=BG, edgecolor=UP, alpha=0.7))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color=TK)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
    legend = ax.legend(facecolor=PNL, edgecolor=GRD, labelcolor=C5, fontsize=9, loc='upper left', framealpha=0.9)
    finish(fig)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""<div class='section-hdr'>
            <span class='sec-icon'>📅</span><span class='sec-title'>Revenue by Year</span>
        </div>""", unsafe_allow_html=True)
        yearly = df.groupby('Year')['Sales'].sum()
        fig2, ax2 = mfig(5.5, 3.5)
        bcolors = [C1, C2, C3, C4]
        bars = ax2.bar(yearly.index.astype(str), yearly.values / 1e3,
                       color=bcolors, width=0.5, edgecolor='none', zorder=3)
        for bar, val in zip(bars, yearly.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, val / 1e3 + 1,
                     f'${val/1e3:.0f}K', ha='center', va='bottom',
                     color=C5, fontsize=10, fontweight='700')
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
        finish(fig2)
        st.pyplot(fig2)

    with col2:
        st.markdown("""<div class='section-hdr'>
            <span class='sec-icon'>🗂️</span><span class='sec-title'>Revenue by Category</span>
        </div>""", unsafe_allow_html=True)
        cat_sel = st.multiselect("Select categories", df['Category'].unique().tolist(),
                                 default=df['Category'].unique().tolist(),
                                 key="cat_select")
        cat = df[df['Category'].isin(cat_sel)].groupby('Category')['Sales'].sum().sort_values()
        fig3, ax3 = mfig(5.5, 3.5)
        for i, (idx, val) in enumerate(cat.items()):
            ax3.barh(idx, val / 1e3, color=[C2, C3, C4][i % 3], height=0.4, zorder=3)
            ax3.text(val / 1e3 + 1, i, f'${val/1e3:.0f}K', va='center',
                     color=C5, fontsize=10, fontweight='700')
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
        finish(fig3)
        st.pyplot(fig3)

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>🌍</span><span class='sec-title'>Revenue by Region</span>
        <span class='sec-badge'>Geographic</span>
    </div>""", unsafe_allow_html=True)
    reg_sel = st.multiselect("Select regions", df['Region'].unique().tolist(),
                              default=df['Region'].unique().tolist(),
                              key="reg_select")
    reg = df[df['Region'].isin(reg_sel)].groupby('Region')['Sales'].sum().sort_values()
    fig4, ax4 = mfig(12, 3)
    for i, (idx, val) in enumerate(reg.items()):
        ax4.barh(idx, val / 1e3, color=[C1, C2, C3, C4][i % 4], height=0.4, zorder=3)
        ax4.text(val / 1e3 + 2, i, f'${val/1e3:.0f}K', va='center',
                 color=C5, fontsize=10, fontweight='700')
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
    finish(fig4)
    st.pyplot(fig4)

# ══════════════════════════════════════════════════════════════════════
# PAGE 2: FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast Explorer":
    st.markdown("<div class='page-tag'>Prophet Time Series Model</div>", unsafe_allow_html=True)
    st.title("Forecast Explorer")
    st.markdown("<p>AI-powered demand forecasting with confidence intervals.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1.5])
    with col1:
        seg_type = st.selectbox("Segment Type", ["Category", "Region"], key="seg_type")
    with col2:
        opts = df['Category'].unique().tolist() if seg_type == "Category" else df['Region'].unique().tolist()
        seg_val = st.selectbox(f"Select {seg_type}", opts, key="seg_val")
        seg_df = df[df['Category'] == seg_val] if seg_type == "Category" else df[df['Region'] == seg_val]
    with col3:
        horizon = st.slider("Forecast Months", 1, 6, 3, key="horizon")

    seg_m = seg_df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    seg_m.columns = ['ds', 'y']

    with st.spinner("🔄 Running Prophet model..."):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='additive', interval_width=0.95)
        m.fit(seg_m)
        future = m.make_future_dataframe(periods=horizon, freq='ME')
        fc = m.predict(future)

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📊</span><span class='sec-title'>Sales Forecast</span>
        <span class='sec-badge'>Prophet</span>
    </div>""", unsafe_allow_html=True)

    fig, ax = mfig(12, 4.5)
    ax.plot(seg_m['ds'], seg_m['y'] / 1e3, color=C2, lw=2.5, label='Historical', zorder=3)
    ax.fill_between(seg_m['ds'], seg_m['y'] / 1e3, alpha=0.1, color=C2)

    fo = fc.tail(horizon)
    ax.plot(fo['ds'], fo['yhat'] / 1e3, color=C4, lw=2.5, marker='o',
            markersize=8, label='Forecast', zorder=5, markeredgecolor=BG, markeredgewidth=2)
    ax.fill_between(fo['ds'], fo['yhat_lower'] / 1e3, fo['yhat_upper'] / 1e3,
                    alpha=0.2, color=C4, label='95% CI', zorder=2)
    ax.axvline(seg_m['ds'].iloc[-1], color=C3, ls=':', lw=1.5, alpha=0.7)

    for _, row in fo.iterrows():
        ax.annotate(f"${row['yhat']/1e3:.1f}K",
                    (row['ds'], row['yhat'] / 1e3),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', color=C4, fontsize=8, fontweight='700')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color=TK)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
    legend = ax.legend(facecolor=PNL, edgecolor=GRD, labelcolor=C5, fontsize=9, loc='upper left', framealpha=0.9)
    finish(fig)
    st.pyplot(fig)

    tp = fc['yhat'].iloc[:len(seg_m)].values
    mae = np.mean(np.abs(seg_m['y'].values - tp))
    rmse = np.sqrt(np.mean((seg_m['y'].values - tp) ** 2))

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📐</span><span class='sec-title'>Model Accuracy</span>
    </div>""", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (m1, "MAE", f"${mae/1e3:.1f}K", "Mean Abs. Error", "📊"),
        (m2, "RMSE", f"${rmse/1e3:.1f}K", "Root Mean Sq.", "📈"),
        (m3, "Month +1", f"${fo['yhat'].iloc[0]/1e3:.1f}K", "Next Month", "📅"),
        (m4, "Month +{horizon}", f"${fo['yhat'].iloc[-1]/1e3:.1f}K", f"+{horizon}M Outlook", "🔮"),
    ]
    
    for col, lbl, val, sub, icon in metrics:
        with col:
            st.markdown(f"""<div class='stat-card'>
                <div style='font-size: 16px; margin-bottom: 6px;'>{icon}</div>
                <div class='stat-label'>{lbl}</div>
                <div class='stat-value' style='font-size: 24px;'>{val}</div>
                <div class='stat-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📋</span><span class='sec-title'>Forecast Breakdown</span>
    </div>""", unsafe_allow_html=True)
    
    tbl = fo[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    tbl.columns = ['Month', 'Forecast', 'Lower Bound', 'Upper Bound']
    tbl['Month'] = tbl['Month'].dt.strftime('%b %Y')
    for c in ['Forecast', 'Lower Bound', 'Upper Bound']:
        tbl[c] = tbl[c].apply(lambda x: f"${x:,.0f}")
    st.dataframe(tbl.set_index('Month'), use_container_width=True, height=250)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Report":
    st.markdown("<div class='page-tag'>ML Detection Engine</div>", unsafe_allow_html=True)
    st.title("Anomaly Report")
    st.markdown("<p>Isolation Forest & Z-Score analysis on 200+ weeks of sales data.</p>", unsafe_allow_html=True)

    ws = weekly.set_index('Date')['Sales']
    iso = IsolationForest(contamination=0.07, random_state=42)
    iso_lbl = iso.fit_predict(ws.values.reshape(-1, 1))
    iso_an = ws[iso_lbl == -1]

    rm = ws.rolling(8, center=True).mean()
    rs = ws.rolling(8, center=True).std()
    z = (ws - rm) / rs
    z_an = ws[z.abs() > 2]
    both = len(set(iso_an.index.date) & set(z_an.index.date))

    c1, c2, c3, c4 = st.columns(4)
    anomaly_stats = [
        (c1, "Isolation Forest", str(len(iso_an)), "Anomalies", "dn", "🌲"),
        (c2, "Z-Score Method", str(len(z_an)), "Anomalies", "dn", "📊"),
        (c3, "Consensus", str(both), "Both Agree", "up", "✅"),
        (c4, "Weeks Analyzed", str(len(ws)), "Full Dataset", "", "📅"),
    ]
    
    for col, lbl, val, sub, kind, icon in anomaly_stats:
        with col:
            sc = "stat-up" if kind == "up" else ("stat-down" if kind == "dn" else "stat-sub")
            st.markdown(f"""<div class='stat-card'>
                <div style='font-size: 16px; margin-bottom: 6px;'>{icon}</div>
                <div class='stat-label'>{lbl}</div>
                <div class='stat-value' style='font-size: 24px;'>{val}</div>
                <div class='{sc}' style='font-size: 12px;'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📉</span><span class='sec-title'>Weekly Sales + Anomalies</span>
        <span class='sec-badge'>Interactive</span>
    </div>""", unsafe_allow_html=True)

    method = st.radio("Detection Method", ["🌲 Isolation Forest", "📊 Z-Score (|z|>2)"], horizontal=True, key="anom_method")
    anom = iso_an if "Isolation" in method else z_an

    fig, ax = mfig(12, 4.5)
    ax.fill_between(ws.index, ws.values / 1e3, alpha=0.08, color=C2)
    ax.plot(ws.index, ws.values / 1e3, color=C2, lw=1.8, alpha=0.9, label='Weekly Sales', zorder=2)

    rm_plot = ws.rolling(4).mean() / 1e3
    ax.plot(ws.index, rm_plot, color=C3, lw=1.5, ls='--', alpha=0.6, label='4-wk avg', zorder=2)

    spikes = anom[anom > ws.mean()]
    drops = anom[anom <= ws.mean()]
    if len(spikes):
        ax.scatter(spikes.index, spikes.values / 1e3, color=UP, s=100, zorder=5,
                   marker='^', label='Spike ↑', edgecolors=BG, lw=1.5)
    if len(drops):
        ax.scatter(drops.index, drops.values / 1e3, color=DN, s=100, zorder=5,
                   marker='v', label='Drop ↓', edgecolors=BG, lw=1.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', color=TK)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}K'))
    legend = ax.legend(facecolor=PNL, edgecolor=GRD, labelcolor=C5, fontsize=9, loc='upper left', framealpha=0.9)
    finish(fig)
    st.pyplot(fig)

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📋</span><span class='sec-title'>Anomaly Log</span>
    </div>""", unsafe_allow_html=True)
    
    adf = anom.reset_index()
    adf.columns = ['Date', 'Sales']
    adf['Month'] = pd.to_datetime(adf['Date']).dt.month
    adf['Signal'] = adf['Sales'].apply(lambda x: '🔺 Spike' if x > ws.mean() else '🔻 Drop')
    cmap = {11: 'Holiday surge', 12: 'Year-end', 1: 'Post-holiday dip', 2: 'Q1 slowdown',
            7: 'Summer promo', 8: 'Back-to-school', 3: 'Spring', 4: 'Easter'}
    adf['Cause'] = adf['Month'].map(cmap).fillna('Irregular')
    adf['Revenue'] = adf['Sales'].apply(lambda x: f"${x:,.0f}")
    adf = adf[['Date', 'Revenue', 'Signal', 'Cause']].sort_values('Date', ascending=False)
    st.dataframe(adf.set_index('Date'), use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════
# PAGE 4: DEMAND SEGMENTS
# ══════════════════════════════════════════════════════════════════════
elif page == "📦 Demand Segments":
    st.markdown("<div class='page-tag'>K-Means Clustering + PCA</div>", unsafe_allow_html=True)
    st.title("Demand Segments")
    st.markdown("<p>Sub-categories segmented by volume, growth, volatility & order value.</p>", unsafe_allow_html=True)

    feat = df.groupby('Sub-Category').agg(
        Total_Sales=('Sales', 'sum'), Avg_Order=('Sales', 'mean'),
        Volatility=('Sales', 'std'), Count=('Sales', 'count')
    ).reset_index()
    yoy = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack()
    feat['Growth'] = yoy.pct_change(axis=1).mean(axis=1).values
    feat = feat.dropna()

    Xs = StandardScaler().fit_transform(feat[['Total_Sales', 'Avg_Order', 'Volatility', 'Growth']])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    feat['Cluster'] = km.fit_predict(Xs)

    sc = feat.groupby('Cluster')['Total_Sales'].mean().sort_values(ascending=False).index
    lmp = {sc[0]: '🏆 High Volume, Stable', sc[1]: '📈 Growing Demand',
           sc[2]: '⚡ High Volatility', sc[3]: '📉 Declining Demand'}
    feat['Segment'] = feat['Cluster'].map(lmp)

    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    feat['x'] = Xp[:, 0]
    feat['y'] = Xp[:, 1]

    pal = {'🏆 High Volume, Stable': C2, '📈 Growing Demand': UP,
           '⚡ High Volatility': C4, '📉 Declining Demand': DN}

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>🎯</span><span class='sec-title'>Cluster Map (PCA)</span>
        <span class='sec-badge'>2D Projection</span>
    </div>""", unsafe_allow_html=True)

    fig, ax = mfig(12, 5.5)
    ax.grid(axis='both', color=GRD, linewidth=0.4, linestyle='--', alpha=0.4)
    for seg, grp in feat.groupby('Segment'):
        ax.scatter(grp['x'], grp['y'], label=seg, color=pal[seg],
                   s=150, edgecolors=PNL, lw=1.5, alpha=0.85, zorder=4)
        for _, row in grp.iterrows():
            ax.annotate(row['Sub-Category'], (row['x'], row['y']),
                        fontsize=8, color=C5, ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points', fontweight='600')
    ax.set_xlabel(f"PC1 — {pca.explained_variance_ratio_[0]*100:.0f}% variance", fontsize=10, fontweight='600')
    ax.set_ylabel(f"PC2 — {pca.explained_variance_ratio_[1]*100:.0f}% variance", fontsize=10, fontweight='600')
    legend = ax.legend(facecolor=BG, edgecolor=GRD, labelcolor=C5, fontsize=10, loc='upper right', framealpha=0.95)
    finish(fig)
    st.pyplot(fig)

    st.markdown("""<div class='section-hdr'>
        <span class='sec-icon'>📋</span><span class='sec-title'>Segment Strategies</span>
    </div>""", unsafe_allow_html=True)

    strat = {
        '🏆 High Volume, Stable': 'Automate reorders. Maintain consistent safety stock.',
        '📈 Growing Demand': 'Increase procurement 15–20%. Prioritize shelf space.',
        '⚡ High Volatility': 'Use just-in-time inventory. Avoid bulk orders.',
        '📉 Declining Demand': 'Reduce stock. Run clearance promotions.',
    }
    bord = [C2, UP, C4, DN]

    col1, col2 = st.columns(2)
    for i, (seg, grp) in enumerate(feat.groupby('Segment')):
        items = ', '.join(sorted(grp['Sub-Category'].tolist())[:5])
        if len(grp) > 5:
            items += f", +{len(grp)-5} more"
        bc = bord[i % 4]
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class='cluster-card' style='border-left-color: {bc};'>
                <div class='c-name'>{seg}</div>
                <div class='c-items'>📦 {items}</div>
                <div class='c-strat'>📌 {strat[seg]}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #8b949e; font-size: 11px; padding: 16px 0;'>
    <p>Made with ❤️ using Streamlit • Data spans {0}–{1}</p>
</div>
""".format(df['Order Date'].dt.year.min(), df['Order Date'].dt.year.max()), unsafe_allow_html=True)
