import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: rgba(100,200,255,0.4);
    }
    [data-testid="metric-container"] label {
        color: #a0aec0 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Section headers */
    h1 { 
        color: #ffffff !important; 
        font-weight: 700 !important;
        font-size: 2rem !important;
        letter-spacing: -0.5px;
    }
    h2 { 
        color: #e2e8f0 !important; 
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        border-bottom: 2px solid rgba(99,179,237,0.4);
        padding-bottom: 8px;
        margin-top: 24px !important;
    }
    h3 { color: #cbd5e0 !important; }

    /* Selectbox & Slider */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }

    /* Radio buttons */
    .stRadio > div {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 12px;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* Info/success boxes */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }

    /* Page title badge */
    .page-badge {
        display: inline-block;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* KPI card custom */
    .kpi-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: #63b3ed;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 12px;
        color: #a0aec0;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .kpi-delta {
        font-size: 13px;
        color: #68d391;
        margin-top: 4px;
    }

    /* Section card */
    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 24px;
        margin: 16px 0;
    }

    /* Sidebar nav */
    .nav-item {
        padding: 10px 16px;
        border-radius: 10px;
        margin: 4px 0;
        cursor: pointer;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Year']    = df['Order Date'].dt.year
    df['Month']   = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
    df['Ship_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    return df

df = load_data()
monthly = df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
monthly.columns = ['Date','Sales']
weekly  = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
weekly.columns  = ['Date','Sales']

# ── Plot style ─────────────────────────────────────────────────────────────
def dark_fig(figsize=(12,5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#a0aec0', labelsize=9)
    ax.xaxis.label.set_color('#a0aec0')
    ax.yaxis.label.set_color('#a0aec0')
    ax.title.set_color('#e2e8f0')
    for spine in ax.spines.values():
        spine.set_edgecolor('rgba(255,255,255,0.08)')
    ax.grid(axis='y', color='rgba(255,255,255,0.05)', linewidth=0.7)
    return fig, ax

def dark_fig2(figsize=(12,5), nrows=1, ncols=2):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor('#1a1a2e')
    for ax in (axes.flat if hasattr(axes,'flat') else [axes]):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#a0aec0', labelsize=9)
        ax.xaxis.label.set_color('#a0aec0')
        ax.yaxis.label.set_color('#a0aec0')
        ax.title.set_color('#e2e8f0')
        for spine in ax.spines.values():
            spine.set_edgecolor('rgba(255,255,255,0.05)')
        ax.grid(axis='y', color='rgba(255,255,255,0.05)', linewidth=0.7)
    return fig, axes

ACCENT   = '#667eea'
ACCENT2  = '#f6ad55'
ACCENT3  = '#68d391'
ACCENT4  = '#fc8181'
ACCENT5  = '#76e4f7'

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:40px;'>📈</div>
        <div style='font-size:18px; font-weight:700; color:#ffffff; margin-top:8px;'>
            Sales Intelligence
        </div>
        <div style='font-size:12px; color:#718096; margin-top:4px;'>
            XYlofy AI · Week 3 & 4
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.1); margin:16px 0;'/>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Sales Overview",
        "🔮  Forecast Explorer",
        "🚨  Anomaly Report",
        "📦  Demand Segments"
    ], label_visibility='collapsed')

    st.markdown("""
    <hr style='border-color:rgba(255,255,255,0.1); margin:20px 0 12px 0;'/>
    <div style='font-size:11px; color:#4a5568; text-align:center; line-height:1.8;'>
        Dataset: Kaggle Superstore<br>
        9,800 orders · 4 years<br>
        3 Categories · 4 Regions
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — SALES OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠  Sales Overview":
    st.markdown('<div class="page-badge">Overview</div>', unsafe_allow_html=True)
    st.title("Sales Overview Dashboard")
    st.markdown("*4 years of Superstore retail data — categories, regions, and trends*")

    st.markdown("---")

    # KPI Row
    total_sales  = df['Sales'].sum()
    total_orders = len(df)
    avg_order    = df['Sales'].mean()
    avg_ship     = df['Ship_Days'].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Total Revenue</div>
            <div class='kpi-value'>${total_sales/1e6:.2f}M</div>
            <div class='kpi-delta'>↑ All 4 Years</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Total Orders</div>
            <div class='kpi-value'>{total_orders:,}</div>
            <div class='kpi-delta'>↑ Growing YoY</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Avg Order Value</div>
            <div class='kpi-value'>${avg_order:,.0f}</div>
            <div class='kpi-delta'>Per Transaction</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Avg Ship Days</div>
            <div class='kpi-value'>{avg_ship:.1f}</div>
            <div class='kpi-delta'>Order to Delivery</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly Trend
    st.subheader("📊 Monthly Sales Trend")
    fig, ax = dark_fig((9, 3.5))
    ax.plot(monthly['Date'], monthly['Sales'], color=ACCENT, lw=2.5)
    ax.fill_between(monthly['Date'], monthly['Sales'], alpha=0.15, color=ACCENT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45, color='#a0aec0')
    ax.set_ylabel('Sales ($)', color='#a0aec0')
    ax.set_title('Monthly Revenue — 2014 to 2017', color='#e2e8f0', pad=12)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Sales by Year
    with col1:
        st.subheader("📅 Sales by Year")
        yearly = df.groupby('Year')['Sales'].sum()
        fig2, ax2 = dark_fig((5, 3.2))
        colors_bar = [ACCENT, ACCENT2, ACCENT3, ACCENT5]
        bars = ax2.bar(yearly.index.astype(str), yearly.values,
                       color=colors_bar, width=0.55, edgecolor='none')
        for bar, val in zip(bars, yearly.values):
            ax2.text(bar.get_x()+bar.get_width()/2, val+1500,
                     f'${val/1e3:.0f}K', ha='center', va='bottom',
                     color='#e2e8f0', fontsize=10, fontweight='600')
        ax2.set_ylabel('Sales ($)', color='#a0aec0')
        plt.tight_layout()
        st.pyplot(fig2)

    # Sales by Category with filter
    with col2:
        st.subheader("🗂 Sales by Category")
        cat_filter = st.multiselect("Filter Categories",
            df['Category'].unique().tolist(),
            default=df['Category'].unique().tolist())
        cat = df[df['Category'].isin(cat_filter)].groupby('Category')['Sales'].sum()
        fig3, ax3 = dark_fig((5, 3.2))
        bars3 = ax3.barh(cat.index, cat.values,
                         color=[ACCENT, ACCENT2, ACCENT3], edgecolor='none', height=0.5)
        for bar, val in zip(bars3, cat.values):
            ax3.text(val+1000, bar.get_y()+bar.get_height()/2,
                     f'${val/1e3:.0f}K', va='center',
                     color='#e2e8f0', fontsize=10, fontweight='600')
        ax3.set_xlabel('Sales ($)', color='#a0aec0')
        plt.tight_layout()
        st.pyplot(fig3)

    # Region filter
    st.subheader("🌍 Sales by Region")
    reg_filter = st.multiselect("Filter Regions",
        df['Region'].unique().tolist(),
        default=df['Region'].unique().tolist())
    reg = df[df['Region'].isin(reg_filter)].groupby('Region')['Sales'].sum().sort_values()
    fig4, ax4 = dark_fig((9, 2.8))
    colors_reg = [ACCENT4, ACCENT2, ACCENT3, ACCENT]
    bars4 = ax4.barh(reg.index, reg.values,
                     color=colors_reg[:len(reg)], edgecolor='none', height=0.5)
    for bar, val in zip(bars4, reg.values):
        ax4.text(val+500, bar.get_y()+bar.get_height()/2,
                 f'${val/1e3:.0f}K', va='center',
                 color='#e2e8f0', fontsize=10, fontweight='600')
    ax4.set_xlabel('Total Sales ($)', color='#a0aec0')
    plt.tight_layout()
    st.pyplot(fig4)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮  Forecast Explorer":
    st.markdown('<div class="page-badge">Forecasting</div>', unsafe_allow_html=True)
    st.title("Forecast Explorer")
    st.markdown("*Prophet model — select a segment and forecast horizon*")
    st.markdown("---")

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        segment_type = st.selectbox("Segment Type", ["Category", "Region"])
    with col2:
        if segment_type == "Category":
            segment_val = st.selectbox("Select Category", df['Category'].unique().tolist())
            seg_df = df[df['Category'] == segment_val]
        else:
            segment_val = st.selectbox("Select Region", df['Region'].unique().tolist())
            seg_df = df[df['Region'] == segment_val]
    with col3:
        horizon = st.slider("Months", 1, 3, 3)

    seg_monthly = seg_df.groupby(
        pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    seg_monthly.columns = ['ds','y']

    with st.spinner("Running Prophet forecast..."):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='additive')
        m.fit(seg_monthly)
        future = m.make_future_dataframe(periods=horizon, freq='ME')
        fc = m.predict(future)

    # Plot
    fig, ax = dark_fig((9, 3.8))
    ax.plot(seg_monthly['ds'], seg_monthly['y'],
            color=ACCENT, lw=2.5, label='Historical Sales')
    forecast_only = fc.tail(horizon)
    ax.plot(forecast_only['ds'], forecast_only['yhat'],
            color=ACCENT2, lw=2.5, marker='o', markersize=8, label='Forecast')
    ax.fill_between(forecast_only['ds'],
                    forecast_only['yhat_lower'], forecast_only['yhat_upper'],
                    alpha=0.25, color=ACCENT2, label='Confidence Interval')
    ax.axvline(seg_monthly['ds'].iloc[-1], color='#718096',
               ls='--', lw=1.2, label='Forecast Start')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, color='#a0aec0')
    ax.set_title(f'{segment_val} — {horizon}-Month Prophet Forecast',
                 color='#e2e8f0', pad=12)
    ax.set_ylabel('Sales ($)', color='#a0aec0')
    legend = ax.legend(facecolor='#1a1a2e', edgecolor='rgba(255,255,255,0.1)',
                       labelcolor='#e2e8f0', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    # Metrics
    train_pred = fc['yhat'].iloc[:len(seg_monthly)].values
    mae  = np.mean(np.abs(seg_monthly['y'].values - train_pred))
    rmse = np.sqrt(np.mean((seg_monthly['y'].values - train_pred)**2))

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Model MAE</div>
            <div class='kpi-value' style='color:#f6ad55;'>${mae:,.0f}</div>
            <div class='kpi-delta'>Mean Abs. Error</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Model RMSE</div>
            <div class='kpi-value' style='color:#f6ad55;'>${rmse:,.0f}</div>
            <div class='kpi-delta'>Root Mean Sq. Error</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Next Month Forecast</div>
            <div class='kpi-value' style='color:#68d391;'>${forecast_only['yhat'].iloc[0]:,.0f}</div>
            <div class='kpi-delta'>Prophet Prediction</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Forecast Table")
    fc_table = forecast_only[['ds','yhat','yhat_lower','yhat_upper']].copy()
    fc_table.columns = ['Month','Forecast ($)','Lower Bound ($)','Upper Bound ($)']
    fc_table['Month'] = fc_table['Month'].dt.strftime('%B %Y')
    for col in ['Forecast ($)','Lower Bound ($)','Upper Bound ($)']:
        fc_table[col] = fc_table[col].apply(lambda x: f"${x:,.0f}")
    st.dataframe(fc_table.set_index('Month'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALY REPORT
# ══════════════════════════════════════════════════════════════════════════
elif page == "🚨  Anomaly Report":
    st.markdown('<div class="page-badge">Anomaly Detection</div>', unsafe_allow_html=True)
    st.title("Anomaly Detection Report")
    st.markdown("*Isolation Forest + Z-Score methods applied to weekly sales data*")
    st.markdown("---")

    weekly_sales = weekly.set_index('Date')['Sales']

    iso = IsolationForest(contamination=0.07, random_state=42)
    iso_labels = iso.fit_predict(weekly_sales.values.reshape(-1,1))
    iso_anom = weekly_sales[iso_labels == -1]

    roll_mean = weekly_sales.rolling(8, center=True).mean()
    roll_std  = weekly_sales.rolling(8, center=True).std()
    z_score   = (weekly_sales - roll_mean) / roll_std
    z_anom    = weekly_sales[z_score.abs() > 2]

    both = len(set(iso_anom.index.date) & set(z_anom.index.date))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Isolation Forest</div>
            <div class='kpi-value' style='color:#fc8181;'>{len(iso_anom)}</div>
            <div class='kpi-delta'>Anomalies Found</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Z-Score Method</div>
            <div class='kpi-value' style='color:#f6ad55;'>{len(z_anom)}</div>
            <div class='kpi-delta'>Anomalies Found</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-label'>Both Agree</div>
            <div class='kpi-value' style='color:#68d391;'>{both}</div>
            <div class='kpi-delta'>High Confidence</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    method = st.radio("Detection Method",
                      ["🌲 Isolation Forest", "📊 Z-Score (|z| > 2)"],
                      horizontal=True)
    anom = iso_anom if "Isolation" in method else z_anom
    anom_color = ACCENT4 if "Isolation" in method else ACCENT2

    fig, ax = dark_fig((9, 3.8))
    ax.plot(weekly_sales.index, weekly_sales.values,
            color=ACCENT, lw=1.5, alpha=0.85, label='Weekly Sales')
    ax.scatter(anom.index, anom.values, color=anom_color,
               s=70, zorder=5, marker='D', label='Anomaly', edgecolors='white', lw=0.5)
    ax.set_title('Weekly Sales with Detected Anomalies', color='#e2e8f0', pad=12)
    ax.set_ylabel('Sales ($)', color='#a0aec0')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, color='#a0aec0')
    legend = ax.legend(facecolor='#1a1a2e', edgecolor='rgba(255,255,255,0.1)',
                       labelcolor='#e2e8f0', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Anomaly Details")
    anom_df = anom.reset_index()
    anom_df.columns = ['Date','Sales']
    anom_df['Month'] = pd.to_datetime(anom_df['Date']).dt.month
    anom_df['Type'] = anom_df['Sales'].apply(
        lambda x: '🔺 Spike' if x > weekly_sales.mean() else '🔻 Drop')
    anom_df['Likely Cause'] = anom_df['Month'].map({
        11:'🎉 Holiday season surge (Nov)',
        12:'🎄 Christmas & year-end spike',
        1:'📉 Post-holiday demand drop',
        2:'📉 Post-holiday slowdown',
        7:'☀️ Summer promotions',
        8:'🎒 Back-to-school demand'
    }).fillna('⚠️ Unusual fluctuation')
    anom_df['Sales'] = anom_df['Sales'].apply(lambda x: f"${x:,.0f}")
    anom_df = anom_df.drop('Month', axis=1).sort_values('Date', ascending=False)
    st.dataframe(anom_df.set_index('Date'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — DEMAND SEGMENTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📦  Demand Segments":
    st.markdown('<div class="page-badge">Segmentation</div>', unsafe_allow_html=True)
    st.title("Product Demand Segments")
    st.markdown("*K-Means clustering on 17 sub-categories — demand behavior analysis*")
    st.markdown("---")

    features = df.groupby('Sub-Category').agg(
        Total_Sales   = ('Sales','sum'),
        Avg_Order_Val = ('Sales','mean'),
        Volatility    = ('Sales','std'),
        Order_Count   = ('Sales','count')
    ).reset_index()
    yoy = df.groupby(['Sub-Category','Year'])['Sales'].sum().unstack()
    features['Growth_Rate'] = yoy.pct_change(axis=1).mean(axis=1).values
    features = features.dropna()

    X = features[['Total_Sales','Avg_Order_Val','Volatility','Growth_Rate']]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    features['Cluster'] = km.fit_predict(X_s)

    sorted_c = features.groupby('Cluster')['Total_Sales'].mean().sort_values(ascending=False).index
    label_map = {
        sorted_c[0]: '🏆 High Volume, Stable',
        sorted_c[1]: '📈 Growing Demand',
        sorted_c[2]: '⚡ Low Volume, High Volatility',
        sorted_c[3]: '📉 Declining / Low Demand'
    }
    features['Segment'] = features['Cluster'].map(label_map)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_s)
    features['PCA1'] = X_pca[:,0]
    features['PCA2'] = X_pca[:,1]

    palette = {
        '🏆 High Volume, Stable'       : ACCENT,
        '📈 Growing Demand'             : ACCENT3,
        '⚡ Low Volume, High Volatility': ACCENT2,
        '📉 Declining / Low Demand'     : ACCENT4
    }

    fig, ax = dark_fig((9, 5))
    for label, grp in features.groupby('Segment'):
        ax.scatter(grp['PCA1'], grp['PCA2'], label=label,
                   color=palette[label], s=120,
                   edgecolors='white', lw=0.8, alpha=0.9)
        for _, row in grp.iterrows():
            ax.annotate(row['Sub-Category'],
                        (row['PCA1'], row['PCA2']),
                        fontsize=8, color='#e2e8f0', ha='center', va='bottom',
                        xytext=(0,7), textcoords='offset points')
    ax.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]*100:.0f}% variance)',
                  color='#a0aec0')
    ax.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]*100:.0f}% variance)',
                  color='#a0aec0')
    ax.set_title('Product Sub-Category Demand Clusters (K-Means + PCA)',
                 color='#e2e8f0', pad=12)
    legend = ax.legend(facecolor='#1a1a2e', edgecolor='rgba(255,255,255,0.1)',
                       labelcolor='#e2e8f0', fontsize=9, loc='upper right')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Segment Details & Strategy")

    for seg, grp in features.groupby('Segment'):
        color = palette[seg].replace('#','')
        items = ', '.join(grp['Sub-Category'].tolist())
        strategies = {
            '🏆 High Volume, Stable'       : 'Maintain consistent safety stock. Automate reorders at fixed intervals.',
            '📈 Growing Demand'             : 'Increase procurement 15–20%. Monitor monthly for acceleration.',
            '⚡ Low Volume, High Volatility': 'Keep lean stock. Use just-in-time ordering to avoid overstock.',
            '📉 Declining / Low Demand'     : 'Reduce stock levels. Run promotions to clear existing inventory.'
        }
        st.markdown(f"""
        <div class='section-card' style='border-left: 4px solid #{color};'>
            <div style='font-size:15px; font-weight:600; color:#e2e8f0; margin-bottom:6px;'>{seg}</div>
            <div style='font-size:13px; color:#a0aec0; margin-bottom:8px;'>
                <b style='color:#cbd5e0;'>Products:</b> {items}
            </div>
            <div style='font-size:13px; color:#68d391;'>
                📌 <b>Strategy:</b> {strategies[seg]}
            </div>
        </div>
        """, unsafe_allow_html=True)
