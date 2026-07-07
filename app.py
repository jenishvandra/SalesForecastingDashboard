import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# ── Load Data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Year']    = df['Order Date'].dt.year
    df['Month']   = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    return df

df = load_data()

monthly = df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
monthly.columns = ['Date','Sales']
weekly  = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
weekly.columns  = ['Date','Sales']

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Page 1 — Sales Overview",
    "Page 2 — Forecast Explorer",
    "Page 3 — Anomaly Report",
    "Page 4 — Product Demand Segments"
])

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — SALES OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "Page 1 — Sales Overview":
    st.title("📊 Sales Overview Dashboard")
    st.markdown("Superstore Sales Dataset — 4 Years of Retail Data")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue",    f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Orders",     f"{len(df):,}")
    col3.metric("Avg Order Value",  f"${df['Sales'].mean():,.0f}")

    st.subheader("Total Sales by Year")
    yearly = df.groupby('Year')['Sales'].sum()
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(yearly.index.astype(str), yearly.values, color='#1565C0', width=0.5)
    for bar, val in zip(bars, yearly.values):
        ax.text(bar.get_x()+bar.get_width()/2, val+1000,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("Sales ($)")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig)

    st.subheader("Monthly Sales Trend")
    fig2, ax2 = plt.subplots(figsize=(12,4))
    ax2.plot(monthly['Date'], monthly['Sales'], color='#1565C0', lw=2)
    ax2.fill_between(monthly['Date'], monthly['Sales'], alpha=0.15, color='#1565C0')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45)
    ax2.set_ylabel("Sales ($)")
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    st.pyplot(fig2)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Sales by Region")
        region_filter = st.multiselect("Select Regions",
            df['Region'].unique().tolist(), default=df['Region'].unique().tolist())
        reg = df[df['Region'].isin(region_filter)].groupby('Region')['Sales'].sum()
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.bar(reg.index, reg.values, color='#E65100')
        ax3.set_ylabel("Sales ($)")
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
        st.pyplot(fig3)

    with col_b:
        st.subheader("Sales by Category")
        cat_filter = st.multiselect("Select Categories",
            df['Category'].unique().tolist(), default=df['Category'].unique().tolist())
        cat = df[df['Category'].isin(cat_filter)].groupby('Category')['Sales'].sum()
        fig4, ax4 = plt.subplots(figsize=(6,4))
        ax4.bar(cat.index, cat.values, color='#2E7D32')
        ax4.set_ylabel("Sales ($)")
        ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
        st.pyplot(fig4)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "Page 2 — Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.markdown("Select a segment and forecast horizon to view Prophet predictions.")

    col1, col2 = st.columns(2)
    with col1:
        segment_type = st.selectbox("Segment Type", ["Category", "Region"])
    with col2:
        if segment_type == "Category":
            segment_val = st.selectbox("Select", df['Category'].unique().tolist())
            seg_df = df[df['Category'] == segment_val]
        else:
            segment_val = st.selectbox("Select", df['Region'].unique().tolist())
            seg_df = df[df['Region'] == segment_val]

    horizon = st.slider("Forecast Horizon (months)", 1, 3, 3)

    seg_monthly = seg_df.groupby(
        pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    seg_monthly.columns = ['ds','y']

    with st.spinner("Running Prophet forecast..."):
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False)
        m.fit(seg_monthly)
        future = m.make_future_dataframe(periods=horizon, freq='ME')
        fc = m.predict(future)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(seg_monthly['ds'], seg_monthly['y'],
            label='Historical', color='#1565C0', lw=2)
    forecast_only = fc.tail(horizon)
    ax.plot(forecast_only['ds'], forecast_only['yhat'],
            label='Forecast', color='#E65100', lw=2, marker='o')
    ax.fill_between(forecast_only['ds'],
                    forecast_only['yhat_lower'], forecast_only['yhat_upper'],
                    alpha=0.2, color='#E65100')
    ax.set_title(f'{segment_val} — {horizon}-Month Sales Forecast')
    ax.set_ylabel('Sales ($)')
    ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig)

    # Metrics
    train_actual = seg_monthly['y'].values
    train_pred   = fc['yhat'].iloc[:len(seg_monthly)].values
    mae  = np.mean(np.abs(train_actual - train_pred))
    rmse = np.sqrt(np.mean((train_actual - train_pred)**2))

    c1, c2, c3 = st.columns(3)
    c1.metric("Model MAE",  f"${mae:,.0f}")
    c2.metric("Model RMSE", f"${rmse:,.0f}")
    c3.metric("Forecast (next month)", f"${forecast_only['yhat'].iloc[0]:,.0f}")

    st.subheader("Forecast Values")
    fc_table = forecast_only[['ds','yhat','yhat_lower','yhat_upper']].copy()
    fc_table.columns = ['Month','Forecast','Lower Bound','Upper Bound']
    fc_table = fc_table.set_index('Month')
    fc_table = fc_table.applymap(lambda x: f"${x:,.0f}")
    st.dataframe(fc_table)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALY REPORT
# ══════════════════════════════════════════════════════════════════════════
elif page == "Page 3 — Anomaly Report":
    st.title("🚨 Anomaly Detection Report")

    weekly_sales = weekly.set_index('Date')['Sales']

    iso = IsolationForest(contamination=0.07, random_state=42)
    iso_labels = iso.fit_predict(weekly_sales.values.reshape(-1,1))
    iso_anom = weekly_sales[iso_labels == -1]

    roll_mean = weekly_sales.rolling(8, center=True).mean()
    roll_std  = weekly_sales.rolling(8, center=True).std()
    z_score   = (weekly_sales - roll_mean) / roll_std
    z_anom    = weekly_sales[z_score.abs() > 2]

    col1, col2 = st.columns(2)
    col1.metric("Isolation Forest Anomalies", len(iso_anom))
    col2.metric("Z-Score Anomalies", len(z_anom))

    method = st.radio("Select Detection Method",
                      ["Isolation Forest", "Z-Score"])
    anom = iso_anom if method == "Isolation Forest" else z_anom

    fig, ax = plt.subplots(figsize=(13,5))
    ax.plot(weekly_sales.index, weekly_sales.values,
            color='#1565C0', lw=1.2, alpha=0.8, label='Weekly Sales')
    ax.scatter(anom.index, anom.values, color='#E53935',
               s=60, zorder=5, marker='D', label='Anomaly')
    ax.set_title(f'Weekly Sales — {method} Anomalies')
    ax.set_ylabel('Sales ($)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig)

    st.subheader("Detected Anomaly Dates")
    anom_df = anom.reset_index()
    anom_df.columns = ['Date','Sales']
    anom_df['Sales'] = anom_df['Sales'].apply(lambda x: f"${x:,.0f}")
    anom_df['Month'] = pd.to_datetime(anom_df['Date']).dt.month
    anom_df['Likely Cause'] = anom_df['Month'].map({
        11:'Holiday season spike', 12:'Holiday season spike',
        1:'Post-holiday dip', 2:'Post-holiday dip',
        7:'Summer promo', 8:'Back-to-school demand'
    }).fillna('Unusual fluctuation')
    anom_df = anom_df.drop('Month', axis=1)
    st.dataframe(anom_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PRODUCT DEMAND SEGMENTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "Page 4 — Product Demand Segments":
    st.title("📦 Product Demand Segments")
    st.markdown("Sub-categories clustered by demand behavior using K-Means.")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

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
        sorted_c[0]:'High Volume, Stable Demand',
        sorted_c[1]:'Growing Demand',
        sorted_c[2]:'Low Volume, High Volatility',
        sorted_c[3]:'Declining / Low Demand'
    }
    features['Segment'] = features['Cluster'].map(label_map)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_s)
    features['PCA1'] = X_pca[:,0]
    features['PCA2'] = X_pca[:,1]

    palette = {'High Volume, Stable Demand':'#1565C0',
               'Growing Demand':'#2E7D32',
               'Low Volume, High Volatility':'#E65100',
               'Declining / Low Demand':'#C62828'}

    fig, ax = plt.subplots(figsize=(11,7))
    for label, grp in features.groupby('Segment'):
        ax.scatter(grp['PCA1'], grp['PCA2'], label=label,
                   color=palette[label], s=100, edgecolors='white', lw=0.8)
        for _, row in grp.iterrows():
            ax.annotate(row['Sub-Category'], (row['PCA1'], row['PCA2']),
                        fontsize=7.5, ha='center', va='bottom',
                        xytext=(0,5), textcoords='offset points')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Product Demand Segments (K-Means + PCA)')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig)

    st.subheader("Sub-Category Cluster Table")
    display_df = features[['Sub-Category','Segment','Total_Sales','Growth_Rate']].copy()
    display_df['Total_Sales'] = display_df['Total_Sales'].apply(lambda x: f"${x:,.0f}")
    display_df['Growth_Rate'] = display_df['Growth_Rate'].apply(lambda x: f"{x*100:+.1f}%")
    display_df = display_df.sort_values('Segment')
    st.dataframe(display_df, use_container_width=True)

    st.subheader("Stocking Strategy per Segment")
    strategies = {
        'High Volume, Stable Demand':     'Maintain consistent safety stock. Automate reorders at fixed intervals.',
        'Growing Demand':                  'Increase procurement by 15-20%. Monitor monthly for acceleration.',
        'Low Volume, High Volatility':     'Keep lean stock. Use just-in-time ordering to avoid overstock.',
        'Declining / Low Demand':          'Reduce stock levels. Consider promotions to clear existing inventory.',
    }
    for seg, strat in strategies.items():
        st.markdown(f"**{seg}:** {strat}")
