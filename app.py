import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
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

# ══════════════════════════════════════════════════════
#  FULL CUSTOM CSS — Modern SaaS Style
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ── App background ── */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Hide streamlit default header ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #8b949e !important; }
[data-testid="stSidebar"] .stRadio label { 
    color: #8b949e !important; 
    font-size: 13px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

/* ── Headings ── */
h1 { color: #e6edf3 !important; font-weight: 800 !important; font-size: 1.75rem !important; letter-spacing: -0.5px; margin-bottom: 4px !important; }
h2 { color: #c9d1d9 !important; font-weight: 600 !important; font-size: 1rem !important; margin-top: 28px !important; margin-bottom: 12px !important; }
h3 { color: #8b949e !important; font-weight: 500 !important; font-size: 0.9rem !important; }
p  { color: #8b949e !important; font-size: 13px !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
[data-testid="metric-container"]:hover { border-color: #3D52A0 !important; }
[data-testid="metric-container"] label { color: #8b949e !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.8px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 24px !important; font-weight: 700 !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 13px !important;
}
.stMultiSelect > div > div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Slider ── */
.stSlider > div > div > div { background: #3D52A0 !important; }
.stSlider label { color: #8b949e !important; font-size: 12px !important; }

/* ── Radio ── */
.stRadio > div { gap: 4px !important; }
.stRadio > div > label {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 14px !important;
    font-size: 12px !important;
    color: #8b949e !important;
    cursor: pointer;
    transition: all 0.15s;
}
.stRadio > div > label:hover { border-color: #7091E6 !important; color: #e6edf3 !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #30363d !important; }
[data-testid="stDataFrame"] * { color: #c9d1d9 !important; font-size: 12px !important; }
[data-testid="stDataFrame"] th { background: #21262d !important; color: #8b949e !important; }
[data-testid="stDataFrame"] tr:hover td { background: #21262d !important; }

/* ── Divider ── */
hr { border-color: #21262d !important; margin: 16px 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7091E6 !important; }

/* ── Alert ── */
.stAlert { border-radius: 10px !important; border: none !important; background: #21262d !important; }

/* ── Custom components ── */
.stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 22px;
    transition: border-color 0.2s, transform 0.2s;
    cursor: default;
}
.stat-card:hover { border-color: #3D52A0; transform: translateY(-2px); }
.stat-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500; }
.stat-value { font-size: 28px; font-weight: 800; color: #e6edf3; margin: 6px 0 4px 0; letter-spacing: -1px; }
.stat-sub   { font-size: 12px; color: #3D52A0; font-weight: 500; }
.stat-up    { color: #3fb950 !important; }
.stat-down  { color: #f85149 !important; }

.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #21262d;
}
.section-icon { font-size: 16px; }
.section-title { font-size: 14px; font-weight: 600; color: #c9d1d9; }
.section-badge {
    font-size: 10px; font-weight: 600; padding: 2px 8px;
    border-radius: 20px; background: #3D52A020; color: #7091E6;
    border: 1px solid #3D52A050; text-transform: uppercase; letter-spacing: 0.5px;
}

.cluster-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid;
    border-radius: 10px;
    padding: 16px 18px;
    margin: 8px 0;
    transition: all 0.2s;
}
.cluster-card:hover { background: #21262d; }
.cluster-name { font-size: 14px; font-weight: 600; color: #e6edf3; margin-bottom: 5px; }
.cluster-items { font-size: 12px; color: #8b949e; margin-bottom: 6px; }
.cluster-strategy { font-size: 12px; color: #7091E6; font-weight: 500; }

.forecast-pill {
    display: inline-block;
    background: #3D52A020;
    border: 1px solid #3D52A050;
    color: #7091E6;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 16px;
}

.anomaly-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}
.badge-up   { background: #3fb95020; color: #3fb950; }
.badge-down { background: #f8514920; color: #f85149; }

/* Sidebar brand */
.brand-wrap {
    padding: 20px 16px 16px 16px;
    border-bottom: 1px solid #30363d;
    margin-bottom: 8px;
}
.brand-logo { font-size: 22px; font-weight: 800; color: #e6edf3 !important; letter-spacing: -0.5px; }
.brand-logo span { color: #7091E6 !important; }
.brand-sub  { font-size: 11px; color: #8b949e !important; margin-top: 2px; }

.nav-label {
    font-size: 10px; font-weight: 600; color: #8b949e !important;
    text-transform: uppercase; letter-spacing: 1px;
    padding: 8px 0 4px 0; display: block;
}
</style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────
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

# ── Chart style ────────────────────────────────────────────────────────────
P = {'bg':'#0d1117','panel':'#161b22','grid':'#21262d',
     'tick':'#8b949e','text':'#c9d1d9',
     'c1':'#3D52A0','c2':'#7091E6','c3':'#8697C4','c4':'#ADBBDA','c5':'#EDE8F5',
     'up':'#3fb950','down':'#f85149'}

def mfig(w=10, h=4, ncols=1, nrows=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(P['bg'])
    ax_list = axes.flat if hasattr(axes,'flat') else [axes]
    for ax in ax_list:
        ax.set_facecolor(P['panel'])
        ax.tick_params(colors=P['tick'], labelsize=8, length=0)
        ax.xaxis.label.set_color(P['tick'])
        ax.yaxis.label.set_color(P['tick'])
        ax.title.set_color(P['text'])
        for spine in ax.spines.values():
            spine.set_edgecolor(P['grid'])
        ax.grid(axis='y', color=P['grid'], linewidth=0.6, linestyle='--')
        ax.set_axisbelow(True)
    return fig, axes

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
        <div class="brand-logo">Sales<span>IQ</span></div>
        <div class="brand-sub">Demand Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="nav-label">Navigation</span>', unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Overview",
        "🔮  Forecasting",
        "🚨  Anomalies",
        "📦  Segments"
    ], label_visibility='collapsed')

    st.markdown("""<hr/>""", unsafe_allow_html=True)

    yearly_rev = df.groupby('Year')['Sales'].sum()
    growth = (yearly_rev.iloc[-1] - yearly_rev.iloc[-2]) / yearly_rev.iloc[-2] * 100
    st.markdown(f"""
    <div style="padding:0 4px;">
        <div class="nav-label">Quick Stats</div>
        <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #21262d;">
            <span style="font-size:12px;color:#8b949e;">Total Revenue</span>
            <span style="font-size:12px;color:#e6edf3;font-weight:600;">${df['Sales'].sum()/1e6:.1f}M</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #21262d;">
            <span style="font-size:12px;color:#8b949e;">Orders</span>
            <span style="font-size:12px;color:#e6edf3;font-weight:600;">{len(df):,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #21262d;">
            <span style="font-size:12px;color:#8b949e;">YoY Growth</span>
            <span style="font-size:12px;color:#3fb950;font-weight:600;">+{growth:.1f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:6px 0;">
            <span style="font-size:12px;color:#8b949e;">Date Range</span>
            <span style="font-size:12px;color:#e6edf3;font-weight:600;">2014–2017</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "📊  Overview":

    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
            XYlofy AI · Week 3 & 4
        </div>
        <h1 style="margin:0;">Sales Overview</h1>
        <p style="margin-top:4px;">Real-time retail performance across categories, regions and time.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    total_sales  = df['Sales'].sum()
    total_orders = len(df)
    avg_order    = df['Sales'].mean()
    avg_ship     = df['Ship_Days'].mean()
    yearly_sales = df.groupby('Year')['Sales'].sum()
    yoy_growth   = (yearly_sales.iloc[-1]-yearly_sales.iloc[-2])/yearly_sales.iloc[-2]*100

    c1,c2,c3,c4,c5 = st.columns(5)
    stats = [
        (c1, "Total Revenue",    f"${total_sales/1e6:.2f}M", f"↑ {yoy_growth:.1f}% YoY", "up"),
        (c2, "Total Orders",     f"{total_orders:,}",         "4 Years of Data",           ""),
        (c3, "Avg Order Value",  f"${avg_order:,.0f}",        "Per Transaction",           ""),
        (c4, "Avg Ship Time",    f"{avg_ship:.1f}d",          "Order to Delivery",         ""),
        (c5, "Product Categories","3",                        "Furn · Tech · Office",      ""),
    ]
    for col, label, val, sub, kind in stats:
        sub_class = "stat-up" if kind=="up" else ("stat-down" if kind=="down" else "")
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{val}</div>
                <div class="stat-sub {sub_class}">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── Monthly Trend ──
    st.markdown("""<div class="section-header">
        <span class="section-icon">📈</span>
        <span class="section-title">Monthly Revenue Trend</span>
        <span class="section-badge">4 Years</span>
    </div>""", unsafe_allow_html=True)

    fig, ax = mfig(10, 3.5)
    ax.plot(monthly['Date'], monthly['Sales']/1e3, color=P['c2'], lw=2, zorder=3)
    ax.fill_between(monthly['Date'], monthly['Sales']/1e3, alpha=0.12, color=P['c2'])

    # Annotate peak
    peak_idx = monthly['Sales'].idxmax()
    ax.scatter(monthly.loc[peak_idx,'Date'], monthly.loc[peak_idx,'Sales']/1e3,
               color=P['up'], s=60, zorder=5)
    ax.annotate(f"Peak: ${monthly.loc[peak_idx,'Sales']/1e3:.0f}K",
                (monthly.loc[peak_idx,'Date'], monthly.loc[peak_idx,'Sales']/1e3),
                xytext=(10,10), textcoords='offset points',
                color=P['up'], fontsize=8, fontweight='600')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=30, ha='right', color=P['tick'])
    ax.set_ylabel("Revenue ($K)", fontsize=9)
    ax.set_title("")
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)

    # ── Year + Category ──
    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("""<div class="section-header">
            <span class="section-icon">📅</span>
            <span class="section-title">Revenue by Year</span>
        </div>""", unsafe_allow_html=True)
        yearly = df.groupby('Year')['Sales'].sum()
        fig2, ax2 = mfig(5, 3.2)
        bar_colors = [P['c1'],P['c2'],P['c3'],P['c4']]
        bars = ax2.bar(yearly.index.astype(str), yearly.values/1e3,
                       color=bar_colors, width=0.5, edgecolor='none', zorder=3)
        for bar, val in zip(bars, yearly.values):
            ax2.text(bar.get_x()+bar.get_width()/2, val/1e3+1,
                     f'${val/1e3:.0f}K', ha='center', va='bottom',
                     color=P['c5'], fontsize=9, fontweight='600')
        ax2.set_ylabel("Revenue ($K)", fontsize=9)
        fig2.tight_layout(pad=1.5)
        st.pyplot(fig2)

    with col2:
        st.markdown("""<div class="section-header">
            <span class="section-icon">🗂</span>
            <span class="section-title">Revenue by Category</span>
        </div>""", unsafe_allow_html=True)
        cat_sel = st.multiselect("", df['Category'].unique().tolist(),
                                 default=df['Category'].unique().tolist(),
                                 label_visibility='collapsed')
        cat = df[df['Category'].isin(cat_sel)].groupby('Category')['Sales'].sum().sort_values()
        fig3, ax3 = mfig(5, 3.2)
        for i, (idx, val) in enumerate(cat.items()):
            ax3.barh(idx, val/1e3, color=[P['c2'],P['c3'],P['c4']][i], height=0.4, zorder=3)
            ax3.text(val/1e3+0.5, i, f'${val/1e3:.0f}K', va='center',
                     color=P['c5'], fontsize=9, fontweight='600')
        ax3.set_xlabel("Revenue ($K)", fontsize=9)
        fig3.tight_layout(pad=1.5)
        st.pyplot(fig3)

    # ── Region ──
    st.markdown("""<div class="section-header">
        <span class="section-icon">🌍</span>
        <span class="section-title">Revenue by Region</span>
        <span class="section-badge">Interactive</span>
    </div>""", unsafe_allow_html=True)
    reg_sel = st.multiselect("", df['Region'].unique().tolist(),
                              default=df['Region'].unique().tolist(),
                              label_visibility='collapsed')
    reg = df[df['Region'].isin(reg_sel)].groupby('Region')['Sales'].sum().sort_values()
    fig4, ax4 = mfig(10, 2.5)
    for i,(idx,val) in enumerate(reg.items()):
        color = [P['c1'],P['c2'],P['c3'],P['c4']][i%4]
        ax4.barh(idx, val/1e3, color=color, height=0.4, zorder=3)
        ax4.text(val/1e3+0.3, i, f'${val/1e3:.0f}K', va='center',
                 color=P['c5'], fontsize=9, fontweight='600')
    ax4.set_xlabel("Revenue ($K)", fontsize=9)
    fig4.tight_layout(pad=1.5)
    st.pyplot(fig4)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮  Forecasting":
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Prophet Model</div>
        <h1 style="margin:0;">Forecast Explorer</h1>
        <p style="margin-top:4px;">Select a segment and horizon to generate a demand forecast.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        segment_type = st.selectbox("Segment Type", ["Category","Region"])
    with col2:
        opts = df['Category'].unique().tolist() if segment_type=="Category" else df['Region'].unique().tolist()
        segment_val = st.selectbox(f"Select {segment_type}", opts)
        seg_df = df[df['Category']==segment_val] if segment_type=="Category" else df[df['Region']==segment_val]
    with col3:
        horizon = st.slider("Months Ahead", 1, 3, 3)

    seg_monthly = seg_df.groupby(pd.Grouper(key='Order Date',freq='ME'))['Sales'].sum().reset_index()
    seg_monthly.columns = ['ds','y']

    with st.spinner("Running Prophet model..."):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='additive')
        m.fit(seg_monthly)
        future = m.make_future_dataframe(periods=horizon, freq='ME')
        fc = m.predict(future)

    # ── Forecast chart ──
    st.markdown("""<div class="section-header">
        <span class="section-icon">📈</span>
        <span class="section-title">Sales Forecast</span>
        <span class="section-badge">Prophet</span>
    </div>""", unsafe_allow_html=True)

    fig, ax = mfig(10, 3.8)
    ax.plot(seg_monthly['ds'], seg_monthly['y']/1e3,
            color=P['c2'], lw=2, label='Historical', zorder=3)
    fo = fc.tail(horizon)
    ax.plot(fo['ds'], fo['yhat']/1e3,
            color=P['c4'], lw=2.2, marker='o', markersize=8,
            label='Forecast', zorder=4)
    ax.fill_between(fo['ds'], fo['yhat_lower']/1e3, fo['yhat_upper']/1e3,
                    alpha=0.18, color=P['c4'], label='95% CI')
    ax.axvline(seg_monthly['ds'].iloc[-1], color=P['c3'],
               ls=':', lw=1.2, alpha=0.8, label='Forecast start')

    # Annotate forecast values
    for _, row in fo.iterrows():
        ax.annotate(f"${row['yhat']/1e3:.1f}K",
                    (row['ds'], row['yhat']/1e3),
                    xytext=(0,10), textcoords='offset points',
                    ha='center', color=P['c4'], fontsize=8, fontweight='600')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30, ha='right', color=P['tick'])
    ax.set_ylabel("Revenue ($K)", fontsize=9)
    legend = ax.legend(facecolor=P['bg'], edgecolor=P['grid'],
                       labelcolor=P['c4'], fontsize=8, loc='upper left')
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)

    # ── Metrics ──
    train_pred = fc['yhat'].iloc[:len(seg_monthly)].values
    mae  = np.mean(np.abs(seg_monthly['y'].values - train_pred))
    rmse = np.sqrt(np.mean((seg_monthly['y'].values - train_pred)**2))

    st.markdown("""<div class="section-header">
        <span class="section-icon">📐</span>
        <span class="section-title">Model Accuracy</span>
    </div>""", unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    metrics = [
        (m1,"MAE",              f"${mae/1e3:.1f}K",  "Mean Absolute Error"),
        (m2,"RMSE",             f"${rmse/1e3:.1f}K", "Root Mean Sq. Error"),
        (m3,"Forecast M+1",     f"${fo['yhat'].iloc[0]/1e3:.1f}K","Next Month"),
        (m4,"Forecast M+3",     f"${fo['yhat'].iloc[-1]/1e3:.1f}K","Month +3"),
    ]
    for col,label,val,sub in metrics:
        with col:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="font-size:22px;">{val}</div>
                <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── Table ──
    st.markdown("""<div class="section-header">
        <span class="section-icon">📋</span>
        <span class="section-title">Forecast Breakdown</span>
    </div>""", unsafe_allow_html=True)
    tbl = fo[['ds','yhat','yhat_lower','yhat_upper']].copy()
    tbl.columns = ['Month','Forecast','Lower Bound','Upper Bound']
    tbl['Month'] = tbl['Month'].dt.strftime('%B %Y')
    for c in ['Forecast','Lower Bound','Upper Bound']:
        tbl[c] = tbl[c].apply(lambda x: f"${x:,.0f}")
    st.dataframe(tbl.set_index('Month'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALIES
# ══════════════════════════════════════════════════════════════════════════
elif page == "🚨  Anomalies":
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Detection Engine</div>
        <h1 style="margin:0;">Anomaly Report</h1>
        <p style="margin-top:4px;">Isolation Forest + Z-Score applied to weekly sales data.</p>
    </div>
    """, unsafe_allow_html=True)

    ws = weekly.set_index('Date')['Sales']

    iso = IsolationForest(contamination=0.07, random_state=42)
    iso_labels = iso.fit_predict(ws.values.reshape(-1,1))
    iso_anom = ws[iso_labels==-1]

    rm = ws.rolling(8, center=True).mean()
    rs = ws.rolling(8, center=True).std()
    z  = (ws - rm) / rs
    z_anom = ws[z.abs() > 2]
    both   = len(set(iso_anom.index.date) & set(z_anom.index.date))

    # ── KPIs ──
    c1,c2,c3,c4 = st.columns(4)
    for col, label, val, sub, kind in [
        (c1,"Isolation Forest",  str(len(iso_anom)), "Anomalies Detected", "down"),
        (c2,"Z-Score Detections",str(len(z_anom)),   "Anomalies Detected", "down"),
        (c3,"Consensus",         str(both),           "Both Methods Agree", "up"),
        (c4,"Weeks Analyzed",    str(len(ws)),        "Full Dataset",       ""),
    ]:
        sub_class = "stat-up" if kind=="up" else ("stat-down" if kind=="down" else "stat-sub")
        with col:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="font-size:22px;">{val}</div>
                <div class="{sub_class}" style="font-size:12px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── Chart ──
    st.markdown("""<div class="section-header">
        <span class="section-icon">📉</span>
        <span class="section-title">Weekly Sales + Anomalies</span>
        <span class="section-badge">Interactive</span>
    </div>""", unsafe_allow_html=True)

    method = st.radio("", ["🌲 Isolation Forest","📊 Z-Score (|z|>2)"], horizontal=True)
    anom   = iso_anom if "Isolation" in method else z_anom

    fig, ax = mfig(10, 3.8)
    ax.plot(ws.index, ws.values/1e3, color=P['c2'], lw=1.6, alpha=0.9, label='Weekly Sales', zorder=2)

    spikes = anom[anom > ws.mean()]
    drops  = anom[anom <= ws.mean()]
    if len(spikes):
        ax.scatter(spikes.index, spikes.values/1e3, color=P['up'],
                   s=70, zorder=5, marker='^', label='Spike', edgecolors=P['bg'], lw=0.5)
    if len(drops):
        ax.scatter(drops.index, drops.values/1e3, color=P['down'],
                   s=70, zorder=5, marker='v', label='Drop', edgecolors=P['bg'], lw=0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30, ha='right', color=P['tick'])
    ax.set_ylabel("Revenue ($K)", fontsize=9)
    legend = ax.legend(facecolor=P['bg'], edgecolor=P['grid'],
                       labelcolor=P['c5'], fontsize=8)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)

    # ── Table ──
    st.markdown("""<div class="section-header">
        <span class="section-icon">📋</span>
        <span class="section-title">Anomaly Log</span>
    </div>""", unsafe_allow_html=True)

    anom_df = anom.reset_index()
    anom_df.columns = ['Date','Sales']
    anom_df['Month'] = pd.to_datetime(anom_df['Date']).dt.month
    anom_df['Signal'] = anom_df['Sales'].apply(
        lambda x: '🔺 Spike' if x > ws.mean() else '🔻 Drop')
    cause_map = {11:'Holiday season surge',12:'Christmas / year-end',
                 1:'Post-holiday dip',2:'Post-holiday slowdown',
                 7:'Summer promotions',8:'Back-to-school'}
    anom_df['Likely Cause'] = anom_df['Month'].map(cause_map).fillna('Unusual fluctuation')
    anom_df['Revenue'] = anom_df['Sales'].apply(lambda x: f"${x:,.0f}")
    anom_df = anom_df[['Date','Revenue','Signal','Likely Cause']].sort_values('Date', ascending=False)
    st.dataframe(anom_df.set_index('Date'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — SEGMENTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📦  Segments":
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">K-Means Clustering</div>
        <h1 style="margin:0;">Demand Segments</h1>
        <p style="margin-top:4px;">17 sub-categories clustered by volume, growth, volatility & order value.</p>
    </div>
    """, unsafe_allow_html=True)

    features = df.groupby('Sub-Category').agg(
        Total_Sales=('Sales','sum'), Avg_Order=('Sales','mean'),
        Volatility=('Sales','std'),  Count=('Sales','count')
    ).reset_index()
    yoy = df.groupby(['Sub-Category','Year'])['Sales'].sum().unstack()
    features['Growth'] = yoy.pct_change(axis=1).mean(axis=1).values
    features = features.dropna()

    X_s = StandardScaler().fit_transform(
        features[['Total_Sales','Avg_Order','Volatility','Growth']])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    features['Cluster'] = km.fit_predict(X_s)

    sc = features.groupby('Cluster')['Total_Sales'].mean().sort_values(ascending=False).index
    lmap = {sc[0]:'🏆 High Volume, Stable', sc[1]:'📈 Growing Demand',
            sc[2]:'⚡ High Volatility',     sc[3]:'📉 Declining Demand'}
    features['Segment'] = features['Cluster'].map(lmap)

    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(X_s)
    features['x'] = Xp[:,0]
    features['y'] = Xp[:,1]

    palette = {'🏆 High Volume, Stable':P['c2'],
               '📈 Growing Demand':P['up'],
               '⚡ High Volatility':P['c4'],
               '📉 Declining Demand':P['down']}

    st.markdown("""<div class="section-header">
        <span class="section-icon">🎯</span>
        <span class="section-title">Cluster Visualization (PCA)</span>
        <span class="section-badge">2D Projection</span>
    </div>""", unsafe_allow_html=True)

    fig, ax = mfig(10, 4.8)
    for seg, grp in features.groupby('Segment'):
        ax.scatter(grp['x'], grp['y'], label=seg, color=palette[seg],
                   s=130, edgecolors=P['panel'], lw=1.2, alpha=0.95, zorder=4)
        for _, row in grp.iterrows():
            ax.annotate(row['Sub-Category'], (row['x'], row['y']),
                        fontsize=7.5, color=P['c5'], ha='center', va='bottom',
                        xytext=(0,8), textcoords='offset points')
    ax.set_xlabel(f"PC1 — {pca.explained_variance_ratio_[0]*100:.0f}% variance", fontsize=9)
    ax.set_ylabel(f"PC2 — {pca.explained_variance_ratio_[1]*100:.0f}% variance", fontsize=9)
    legend = ax.legend(facecolor=P['bg'], edgecolor=P['grid'],
                       labelcolor=P['c5'], fontsize=8, loc='upper right')
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)

    # ── Cluster cards ──
    st.markdown("""<div class="section-header">
        <span class="section-icon">📋</span>
        <span class="section-title">Segment Details & Stocking Strategy</span>
    </div>""", unsafe_allow_html=True)

    strategies = {
        '🏆 High Volume, Stable': 'Maintain consistent safety stock. Automate reorders. These products never go out of demand.',
        '📈 Growing Demand':      'Increase procurement by 15–20%. Monitor monthly. Prioritise shelf space and marketing.',
        '⚡ High Volatility':     'Keep lean just-in-time inventory. Avoid bulk orders. Demand is unpredictable.',
        '📉 Declining Demand':    'Reduce stock levels immediately. Run clearance promotions before Q4 season.',
    }
    bcolors = [P['c2'], P['up'], P['c4'], P['down']]

    col1, col2 = st.columns(2)
    for i, (seg, grp) in enumerate(features.groupby('Segment')):
        col = col1 if i % 2 == 0 else col2
        items = ', '.join(sorted(grp['Sub-Category'].tolist()))
        bc = bcolors[i % 4]
        with col:
            st.markdown(f"""
            <div class="cluster-card" style="border-left-color:{bc};">
                <div class="cluster-name">{seg}</div>
                <div class="cluster-items">📦 {items}</div>
                <div class="cluster-strategy">📌 {strategies[seg]}</div>
            </div>""", unsafe_allow_html=True)
