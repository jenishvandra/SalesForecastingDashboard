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

st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Color Palette (from image) ─────────────────────────────────────────────
C1 = '#3D52A0'   # deep blue
C2 = '#7091E6'   # medium blue
C3 = '#8697C4'   # muted blue
C4 = '#ADBBDA'   # light blue
C5 = '#EDE8F5'   # very light lavender

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background: linear-gradient(160deg, #1a2150 0%, #2a3a7a 50%, #3D52A0 100%);
    color: #EDE8F5;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #1a1f4e 0%, #2a3070 100%);
    border-right: 1px solid {C3}40;
}}
[data-testid="stSidebar"] * {{ color: {C4} !important; }}

[data-testid="metric-container"] {{
    background: {C1}80;
    border: 1px solid {C3}60;
    border-radius: 14px;
    padding: 18px;
    backdrop-filter: blur(10px);
}}
[data-testid="metric-container"] label {{
    color: {C4} !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {C5} !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}}

h1 {{ color: {C5} !important; font-weight: 700 !important; font-size: 1.9rem !important; }}
h2 {{ 
    color: {C4} !important; font-weight: 600 !important; font-size: 1.15rem !important;
    border-bottom: 2px solid {C2}60; padding-bottom: 6px; margin-top: 20px !important;
}}
h3 {{ color: {C4} !important; }}
p, li {{ color: {C4}; }}

.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {C1}80 !important;
    border: 1px solid {C3}80 !important;
    border-radius: 10px !important;
    color: {C5} !important;
}}
.stSlider > div > div > div {{ background: {C2} !important; }}

.stRadio > div {{
    background: {C1}50;
    border-radius: 12px;
    padding: 10px;
}}

.stDataFrame {{ border-radius: 12px; overflow: hidden; }}
hr {{ border-color: {C3}40 !important; }}

.page-badge {{
    display: inline-block;
    background: linear-gradient(90deg, {C2}, {C1});
    color: {C5};
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
}}

.kpi-card {{
    background: {C1}70;
    border: 1px solid {C3}60;
    border-radius: 16px;
    padding: 18px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s;
}}
.kpi-card:hover {{ transform: translateY(-3px); border-color: {C2}; }}
.kpi-value {{ font-size: 28px; font-weight: 700; color: {C5}; margin: 4px 0; }}
.kpi-label {{ font-size: 11px; color: {C4}; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }}
.kpi-delta {{ font-size: 12px; color: {C2}; margin-top: 4px; }}

.section-card {{
    background: {C1}50;
    border: 1px solid {C3}50;
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0;
}}
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

# ── Dark chart helper ──────────────────────────────────────────────────────
BG_DARK  = '#1e2a5e'
BG_PANEL = '#2a3570'
TICK_COL = '#ADBBDA'
GRID_COL = '#3D52A0'

def make_fig(w=9, h=3.8, ncols=1):
    if ncols > 1:
        fig, axes = plt.subplots(1, ncols, figsize=(w, h))
        fig.patch.set_facecolor(BG_DARK)
        for ax in axes:
            _style_ax(ax)
        return fig, axes
    else:
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor(BG_DARK)
        _style_ax(ax)
        return fig, ax

def _style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TICK_COL, labelsize=8)
    ax.xaxis.label.set_color(TICK_COL)
    ax.yaxis.label.set_color(TICK_COL)
    ax.title.set_color(C5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(axis='y', color=GRID_COL, linewidth=0.5, alpha=0.5)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:38px;'>📈</div>
        <div style='font-size:17px; font-weight:700; color:{C5}; margin-top:8px;'>Sales Intelligence</div>
        <div style='font-size:11px; color:{C3}; margin-top:4px;'>XYlofy AI · Week 3 & 4</div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Sales Overview",
        "🔮  Forecast Explorer",
        "🚨  Anomaly Report",
        "📦  Demand Segments"
    ], label_visibility='collapsed')

    st.markdown(f"""
    <hr/>
    <div style='font-size:11px; color:{C3}; text-align:center; line-height:2;'>
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
    st.markdown(f"<p style='color:{C4}; margin-top:-10px;'>4 years of Superstore retail data — categories, regions, and trends</p>", unsafe_allow_html=True)
    st.markdown("---")

    total_sales  = df['Sales'].sum()
    total_orders = len(df)
    avg_order    = df['Sales'].mean()
    avg_ship     = df['Ship_Days'].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, delta in zip(
        [c1,c2,c3,c4],
        ['Total Revenue','Total Orders','Avg Order Value','Avg Ship Days'],
        [f'${total_sales/1e6:.2f}M', f'{total_orders:,}', f'${avg_order:,.0f}', f'{avg_ship:.1f} days'],
        ['All 4 Years','Growing YoY','Per Transaction','Order to Delivery']
    ):
        with col:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value'>{value}</div>
                <div class='kpi-delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly Trend
    st.subheader("📊 Monthly Sales Trend")
    fig, ax = make_fig(9, 3.5)
    ax.plot(monthly['Date'], monthly['Sales'], color=C2, lw=2.2)
    ax.fill_between(monthly['Date'], monthly['Sales'], alpha=0.2, color=C2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45, color=TICK_COL)
    ax.set_ylabel('Sales ($)', color=TICK_COL)
    ax.set_title('Monthly Revenue — 2014 to 2017', pad=10)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Sales by Year")
        yearly = df.groupby('Year')['Sales'].sum()
        fig2, ax2 = make_fig(5, 3.2)
        bars = ax2.bar(yearly.index.astype(str), yearly.values,
                       color=[C1, C2, C3, C4], width=0.5, edgecolor='none')
        for bar, val in zip(bars, yearly.values):
            ax2.text(bar.get_x()+bar.get_width()/2, val+500,
                     f'${val/1e3:.0f}K', ha='center', va='bottom',
                     color=C5, fontsize=9, fontweight='600')
        ax2.set_ylabel('Sales ($)', color=TICK_COL)
        plt.tight_layout()
        st.pyplot(fig2)

    with col2:
        st.subheader("🗂 Sales by Category")
        cat_filter = st.multiselect("Filter", df['Category'].unique().tolist(),
            default=df['Category'].unique().tolist())
        cat = df[df['Category'].isin(cat_filter)].groupby('Category')['Sales'].sum()
        fig3, ax3 = make_fig(5, 3.2)
        ax3.barh(cat.index, cat.values, color=[C2, C3, C4], edgecolor='none', height=0.45)
        for i, (idx, val) in enumerate(cat.items()):
            ax3.text(val+200, i, f'${val/1e3:.0f}K', va='center',
                     color=C5, fontsize=9, fontweight='600')
        ax3.set_xlabel('Sales ($)', color=TICK_COL)
        plt.tight_layout()
        st.pyplot(fig3)

    st.subheader("🌍 Sales by Region")
    reg_filter = st.multiselect("Filter Regions", df['Region'].unique().tolist(),
        default=df['Region'].unique().tolist())
    reg = df[df['Region'].isin(reg_filter)].groupby('Region')['Sales'].sum().sort_values()
    fig4, ax4 = make_fig(9, 2.8)
    ax4.barh(reg.index, reg.values, color=[C1, C2, C3, C4][:len(reg)],
             edgecolor='none', height=0.45)
    for i, (idx, val) in enumerate(reg.items()):
        ax4.text(val+200, i, f'${val/1e3:.0f}K', va='center',
                 color=C5, fontsize=9, fontweight='600')
    ax4.set_xlabel('Total Sales ($)', color=TICK_COL)
    plt.tight_layout()
    st.pyplot(fig4)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔮  Forecast Explorer":
    st.markdown('<div class="page-badge">Forecasting</div>', unsafe_allow_html=True)
    st.title("Forecast Explorer")
    st.markdown(f"<p style='color:{C4}; margin-top:-10px;'>Prophet model — select a segment and forecast horizon</p>", unsafe_allow_html=True)
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

    fig, ax = make_fig(9, 3.8)
    ax.plot(seg_monthly['ds'], seg_monthly['y'], color=C2, lw=2.2, label='Historical')
    forecast_only = fc.tail(horizon)
    ax.plot(forecast_only['ds'], forecast_only['yhat'],
            color=C4, lw=2.2, marker='o', markersize=7, label='Forecast')
    ax.fill_between(forecast_only['ds'],
                    forecast_only['yhat_lower'], forecast_only['yhat_upper'],
                    alpha=0.2, color=C4, label='Confidence Interval')
    ax.axvline(seg_monthly['ds'].iloc[-1], color=C3, ls='--', lw=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, color=TICK_COL)
    ax.set_title(f'{segment_val} — {horizon}-Month Forecast', pad=10)
    ax.set_ylabel('Sales ($)', color=TICK_COL)
    legend = ax.legend(facecolor=BG_DARK, edgecolor=GRID_COL,
                       labelcolor=C5, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    train_pred = fc['yhat'].iloc[:len(seg_monthly)].values
    mae  = np.mean(np.abs(seg_monthly['y'].values - train_pred))
    rmse = np.sqrt(np.mean((seg_monthly['y'].values - train_pred)**2))

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, label, val, delta in zip(
        [m1, m2, m3],
        ['Model MAE', 'Model RMSE', 'Next Month Forecast'],
        [f'${mae:,.0f}', f'${rmse:,.0f}', f'${forecast_only["yhat"].iloc[0]:,.0f}'],
        ['Mean Abs. Error', 'Root Mean Sq. Error', 'Prophet Prediction']
    ):
        with col:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Forecast Table")
    fc_table = forecast_only[['ds','yhat','yhat_lower','yhat_upper']].copy()
    fc_table.columns = ['Month','Forecast ($)','Lower Bound ($)','Upper Bound ($)']
    fc_table['Month'] = fc_table['Month'].dt.strftime('%B %Y')
    for c in ['Forecast ($)','Lower Bound ($)','Upper Bound ($)']:
        fc_table[c] = fc_table[c].apply(lambda x: f"${x:,.0f}")
    st.dataframe(fc_table.set_index('Month'), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALY REPORT
# ══════════════════════════════════════════════════════════════════════════
elif page == "🚨  Anomaly Report":
    st.markdown('<div class="page-badge">Anomaly Detection</div>', unsafe_allow_html=True)
    st.title("Anomaly Detection Report")
    st.markdown(f"<p style='color:{C4}; margin-top:-10px;'>Isolation Forest + Z-Score applied to weekly sales data</p>", unsafe_allow_html=True)
    st.markdown("---")

    weekly_sales = weekly.set_index('Date')['Sales']

    iso = IsolationForest(contamination=0.07, random_state=42)
    iso_labels = iso.fit_predict(weekly_sales.values.reshape(-1,1))
    iso_anom = weekly_sales[iso_labels == -1]

    roll_mean = weekly_sales.rolling(8, center=True).mean()
    roll_std  = weekly_sales.rolling(8, center=True).std()
    z_score   = (weekly_sales - roll_mean) / roll_std
    z_anom    = weekly_sales[z_score.abs() > 2]
    both      = len(set(iso_anom.index.date) & set(z_anom.index.date))

    c1, c2, c3 = st.columns(3)
    for col, label, val, delta in zip(
        [c1,c2,c3],
        ['Isolation Forest','Z-Score Method','Both Agree'],
        [len(iso_anom), len(z_anom), both],
        ['Anomalies Found','Anomalies Found','High Confidence']
    ):
        with col:
            st.markdown(f"""<div class='kpi-card'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    method = st.radio("Detection Method",
                      ["🌲 Isolation Forest", "📊 Z-Score (|z| > 2)"], horizontal=True)
    anom       = iso_anom if "Isolation" in method else z_anom
    anom_color = C4 if "Isolation" in method else '#ADBBDA'

    fig, ax = make_fig(9, 3.8)
    ax.plot(weekly_sales.index, weekly_sales.values, color=C2, lw=1.5, alpha=0.85, label='Weekly Sales')
    ax.scatter(anom.index, anom.values, color=anom_color, s=65, zorder=5,
               marker='D', label='Anomaly', edgecolors=BG_DARK, lw=0.5)
    ax.set_title('Weekly Sales with Detected Anomalies', pad=10)
    ax.set_ylabel('Sales ($)', color=TICK_COL)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45, color=TICK_COL)
    legend = ax.legend(facecolor=BG_DARK, edgecolor=GRID_COL, labelcolor=C5, fontsize=9)
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
        11:'🎉 Holiday season surge',12:'🎄 Christmas spike',
        1:'📉 Post-holiday drop',2:'📉 Post-holiday slowdown',
        7:'☀️ Summer promotions',8:'🎒 Back-to-school'
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
    st.markdown(f"<p style='color:{C4}; margin-top:-10px;'>K-Means clustering on 17 sub-categories</p>", unsafe_allow_html=True)
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

    X_s = StandardScaler().fit_transform(
        features[['Total_Sales','Avg_Order_Val','Volatility','Growth_Rate']])
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
        '🏆 High Volume, Stable'        : C1,
        '📈 Growing Demand'              : C2,
        '⚡ Low Volume, High Volatility' : C3,
        '📉 Declining / Low Demand'      : C4
    }

    fig, ax = make_fig(9, 5)
    for label, grp in features.groupby('Segment'):
        ax.scatter(grp['PCA1'], grp['PCA2'], label=label,
                   color=palette[label], s=110, edgecolors=C5, lw=0.6, alpha=0.9)
        for _, row in grp.iterrows():
            ax.annotate(row['Sub-Category'], (row['PCA1'], row['PCA2']),
                        fontsize=7.5, color=C5, ha='center', va='bottom',
                        xytext=(0,6), textcoords='offset points')
    ax.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]*100:.0f}% variance)', color=TICK_COL)
    ax.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]*100:.0f}% variance)', color=TICK_COL)
    ax.set_title('Product Sub-Category Demand Clusters', pad=10)
    legend = ax.legend(facecolor=BG_DARK, edgecolor=GRID_COL, labelcolor=C5, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Segment Strategy")

    strategies = {
        '🏆 High Volume, Stable'        : 'Maintain consistent safety stock. Automate reorders.',
        '📈 Growing Demand'              : 'Increase procurement 15–20%. Monitor monthly.',
        '⚡ Low Volume, High Volatility' : 'Keep lean stock. Use just-in-time ordering.',
        '📉 Declining / Low Demand'      : 'Reduce stock. Run promotions to clear inventory.'
    }
    border_colors = [C1, C2, C3, C4]

    for (seg, grp), bc in zip(features.groupby('Segment'), border_colors):
        items = ', '.join(grp['Sub-Category'].tolist())
        st.markdown(f"""
        <div class='section-card' style='border-left: 4px solid {bc};'>
            <div style='font-size:15px; font-weight:600; color:{C5}; margin-bottom:6px;'>{seg}</div>
            <div style='font-size:13px; color:{C4}; margin-bottom:6px;'>
                <b style='color:{C5};'>Products:</b> {items}
            </div>
            <div style='font-size:13px; color:{C2};'>
                📌 <b>Strategy:</b> {strategies[seg]}
            </div>
        </div>
        """, unsafe_allow_html=True)
