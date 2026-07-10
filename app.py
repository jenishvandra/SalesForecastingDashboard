import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="SalesIQ — Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme ─────────────────────────────────────────────────────────────────
BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
T1 = "#e6edf3"
T2 = "#c9d1d9"
T3 = "#8b949e"
BLUE = "#7091E6"
BLUE2 = "#3D52A0"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
PURPLE = "#a371f7"

PL = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(family="Inter,sans-serif", color=T2, size=11),
    margin=dict(l=50, r=30, t=30, b=40),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showgrid=True),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, showgrid=True, griddash="dot"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(size=10, color=T3),
    ),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=T1, size=11)),
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*{{ font-family:'Inter',sans-serif!important; box-sizing:border-box; }}
.stApp{{ background:{BG}; color:{T1}; }}

/* Hide default Streamlit sidebar (we provide our own navigation) */
#MainMenu{{visibility:hidden;}}
footer{{visibility:hidden;}}
[data-testid="stSidebar"]{{display:none !important;}}

/* Brand + layout */
.sidebar-col{{
    background:{PANEL}; border-right:1px solid {BORDER};
    min-height:100vh; padding:0; position:fixed;
    left:0; top:0; width:230px; z-index:999;
    display:flex; flex-direction:column;
}}
.brand{{padding:20px 16px 16px; border-bottom:1px solid {BORDER};}}
.brand-name{{ font-size:20px; font-weight:800; color:{T1}; letter-spacing:-.5px; }}
.brand-name span{{ color:{BLUE}; }}
.brand-sub{{ font-size:10px; color:{T3}; margin-top:2px; }}
.nav-section{{ font-size:10px; color:{T3}; text-transform:uppercase; letter-spacing:1px; font-weight:600; padding:12px 16px 6px; }}
.nav-item{{
    display:flex; align-items:center; gap:10px;
    padding:10px 16px; font-size:13px; font-weight:500;
    color:{T3}; cursor:pointer; border-radius:0;
    transition:all .15s; border-right:2px solid transparent;
    text-decoration:none;
}}
.nav-item:hover{{ background:#21262d; color:{T1}; }}
.nav-item.active{{ background:#21262d; color:{BLUE}; border-right:2px solid {BLUE}; }}
.nav-icon{{ font-size:14px; width:18px; text-align:center; }}
.qs-section{{ padding:12px 16px; border-top:1px solid {BORDER}; }}
.qs-title{{ font-size:10px; color:{T3}; text-transform:uppercase; letter-spacing:1px; font-weight:600; margin-bottom:8px; }}
.qs-row{{ display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #21262d; }}
.qs-label{{ font-size:11px; color:{T3}; }}
.qs-val{{ font-size:11px; font-weight:700; color:{T2}; }}

.content-area{{ margin-left:230px; padding:24px 28px; min-height:100vh; }}
.page-hdr{{ margin-bottom:20px; padding-bottom:14px; border-bottom:1px solid {BORDER}; }}
.page-hdr h1{{ color:{T1}; font-weight:700; font-size:1.6rem; letter-spacing:-.3px; margin:0; }}
.page-hdr p{{ color:{T3}; font-size:12px; font-family:monospace; margin-top:6px; }}

.kpi-card{{
    background:{PANEL}; border:1px solid {BORDER};
    border-radius:10px; padding:16px 18px;
    position:relative; overflow:hidden; height:100%;
}}
.kpi-card:hover{{ border-color:{BLUE2}; }}
.kpi-icon{{
    position:absolute; top:14px; right:14px;
    width:28px; height:28px; border-radius:6px;
    display:flex; align-items:center; justify-content:center; font-size:14px;
}}
.kpi-label{{ font-size:10px; color:{T3}; text-transform:uppercase; letter-spacing:.8px; font-weight:600; margin-bottom:8px; }}
.kpi-value{{ font-size:26px; font-weight:800; color:{T1}; letter-spacing:-.5px; margin-bottom:5px; }}
.kpi-delta{{ font-size:11px; font-weight:500; }}
.d-up{{ color:{GREEN}; }} .d-dn{{ color:{RED}; }} .d-nu{{ color:{T3}; }}

.cc{{ background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:18px 20px; margin-bottom:14px; }}
.ct{{ font-size:14px; font-weight:600; color:{T1}; margin-bottom:2px; }}
.cs{{ font-size:11px; color:{T3}; font-family:monospace; margin-bottom:12px; }}

.atbl{{ width:100%; border-collapse:collapse; }}
.atbl th{{ text-align:left; padding:8px 12px; font-size:10px; color:{T3}; text-transform:uppercase; letter-spacing:.8px; border-bottom:1px solid {BORDER}; }}
.atbl td{{ padding:8px 12px; border-bottom:1px solid {BORDER}; font-size:12px; color:{T2}; }}
.atbl tr:hover td{{ background:#21262d; }}

.bsp{{ background:rgba(63,185,80,.15); color:{GREEN}; border:1px solid rgba(63,185,80,.3); border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600; }}
.bdr{{ background:rgba(248,81,73,.15); color:{RED}; border:1px solid rgba(248,81,73,.3); border-radius:4px; padding:2px 8px; font-size:11px; font-weight:600; }}

/* Hide Streamlit default header spacing under our fixed sidebar */
div[data-testid="stAppViewContainer"]{{ padding-top:0px !important; }}
</style>
""",
    unsafe_allow_html=True,
)


# ── Data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    # Defensive parsing
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["Order Date", "Sales"]).copy()
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    df["Ship_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df


df = load_data()

monthly = df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index()
monthly.columns = ["Date", "Sales"]
weekly = df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"].sum().reset_index()
weekly.columns = ["Date", "Sales"]

# YoY using last two years (guard against short ranges)
years = sorted(df["Year"].unique())
yoy_pct = 0.0
if len(years) >= 2:
    yr_rev = df.groupby("Year")["Sales"].sum().reindex(years)
    if yr_rev.iloc[-2] != 0:
        yoy_pct = float((yr_rev.iloc[-1] - yr_rev.iloc[-2]) / yr_rev.iloc[-2] * 100)

# ── Navigation state ───────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "overview"

nav_items = [
    ("overview", "📊", "Sales Overview"),
    ("forecast", "📈", "Forecasting"),
    ("anomaly", "🔔", "Anomaly Report"),
    ("segments", "🗂", "Demand Segments"),
]

with st.container():
    nav_html = ""
    for key, icon, label in nav_items:
        active = "active" if st.session_state.page == key else ""
        nav_html += (
            f"<a class='nav-item {active}' href='?page={key}'>"
            f"<span class='nav-icon'>{icon}</span>{label}</a>"
        )

    st.markdown(
        f"""
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
        <div class='qs-row'><span class='qs-label'>Revenue</span><span class='qs-val'>${df['Sales'].sum()/1e6:.2f}M</span></div>
        <div class='qs-row'><span class='qs-label'>Orders</span><span class='qs-val'>{len(df):,}</span></div>
        <div class='qs-row'><span class='qs-label'>YoY Growth</span><span class='qs-val' style='color:{GREEN};'>+{yoy_pct:.1f}%</span></div>
        <div class='qs-row'><span class='qs-label'>Avg Order</span><span class='qs-val'>${df['Sales'].mean():,.0f}</span></div>
        <div class='qs-row' style='border:none;'><span class='qs-label'>Period</span><span class='qs-val'>{df['Year'].min()}–{df['Year'].max()}</span></div>
    </div>
</div>
<div class='content-area'>
""",
        unsafe_allow_html=True,
    )

# Read ?page query param (works reliably)
params = st.query_params
if "page" in params:
    st.session_state.page = params.get("page", "overview")

page = st.session_state.page


# ── Helpers ─────────────────────────────────────────────────────────────────

def plotly_chart(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def kpi_card(col, label, value, delta, dtype, icon, ibg):
    dc = "d-up" if dtype == "up" else ("d-dn" if dtype == "dn" else "d-nu")
    ar = "↗" if dtype == "up" else ("↘" if dtype == "dn" else "")
    with col:
        st.markdown(
            f"""
<div class='kpi-card'>
  <div class='kpi-icon' style='background:{ibg}20;'>{icon}</div>
  <div class='kpi-label'>{label}</div>
  <div class='kpi-value'>{value}</div>
  <div class='kpi-delta {dc}'>{ar} {delta}</div>
</div>
""",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "overview":
    st.markdown(
        """
<div class='page-hdr'>
  <h1>Sales Overview</h1>
  <p>Full-year performance across all channels and regions</p>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Total Revenue", f"${df['Sales'].sum()/1e6:.2f}M", f"{yoy_pct:+.1f}% YoY", "up" if yoy_pct >= 0 else "dn", "💲", BLUE2)
    kpi_card(c2, "Total Orders", f"{len(df):,}", "+12.1% YoY", "up", "🛒", GREEN)
    kpi_card(c3, "Avg Order Value", f"${df['Sales'].mean():,.0f}", "+3.8% YoY", "up", "📈", "#a371f7")
    ship_mean = float(df["Ship_Days"].dropna().mean()) if df["Ship_Days"].notna().any() else 0.0
    kpi_card(c4, "Avg Ship Time", f"{ship_mean:.1f} days", "-0.1% YoY", "dn", "⏱", YELLOW)
    kpi_card(c5, "Categories", "3", "Furn · Tech · Office", "nu", "🏷", RED)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
<div class='cc'>
  <div class='ct'>Monthly Revenue Trend</div>
  <div class='cs'>{df['Year'].min()}–{df['Year'].max()} · 3-month moving average overlay · $K</div>
""",
        unsafe_allow_html=True,
    )

    ma3 = monthly["Sales"].rolling(3).mean()
    pidx = int(monthly["Sales"].idxmax()) if len(monthly) else 0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["Date"],
            y=monthly["Sales"] / 1e3,
            name="Monthly Revenue",
            line=dict(color=BLUE, width=2),
            fill="tozeroy",
            fillcolor="rgba(112,145,230,0.08)",
            hovertemplate="%{x|%b %Y}<br>$%{y:.0f}K<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["Date"],
            y=ma3 / 1e3,
            name="3-Month Moving Avg",
            line=dict(color=YELLOW, width=1.5, dash="dash"),
            hovertemplate="3mo Avg: $%{y:.0f}K<extra></extra>",
        )
    )
    if len(monthly):
        fig.add_vline(
            x=monthly.loc[pidx, "Date"],
            line_dash="dot",
            line_color=GREEN,
            line_width=1.2,
            opacity=0.7,
        )
        fig.add_annotation(
            x=monthly.loc[pidx, "Date"],
            y=monthly.loc[pidx, "Sales"] / 1e3,
            text="Peak",
            showarrow=False,
            font=dict(color=GREEN, size=11),
            xshift=28,
        )

    fig.update_layout(
        **PL,
        height=300,
        yaxis=dict(tickprefix="$", ticksuffix="K", griddash="dot", linecolor=BORDER, gridcolor=BORDER, zeroline=False),
        xaxis=dict(tickformat="%b %y", gridcolor=BORDER, linecolor=BORDER, zeroline=False),
    )
    plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    ca, cb, cc = st.columns(3)

    with ca:
        st.markdown("<div class='cc'><div class='ct'>By Year</div>", unsafe_allow_html=True)
        yr = df.groupby("Year")["Sales"].sum().sort_index()
        fig2 = go.Figure(
            go.Bar(
                x=[str(x) for x in yr.index],
                y=yr.values / 1e6,
                marker_color=[BLUE2, BLUE, GREEN, PURPLE],
                text=[f"${v/1e6:.1f}M" for v in yr.values],
                textposition="outside",
                textfont=dict(size=10, color=T2),
                hovertemplate="%{x}<br>$%{y:.2f}M<extra></extra>",
            )
        )
        fig2.update_layout(
            **PL,
            height=220,
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=30),
            yaxis=dict(tickprefix="$", ticksuffix="M", gridcolor=BORDER, zeroline=False, griddash="dot", linecolor=BORDER),
            xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False),
        )
        plotly_chart(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    with cb:
        st.markdown("<div class='cc'><div class='ct'>Revenue by Category</div>", unsafe_allow_html=True)
        cat_sel = st.multiselect(
            "",
            df["Category"].dropna().unique().tolist(),
            default=df["Category"].dropna().unique().tolist(),
            label_visibility="collapsed",
            key="cs1",
        )
        cat_df = df[df["Category"].isin(cat_sel)]
        cat = cat_df.groupby("Category")["Sales"].sum().sort_values()
        fig3 = go.Figure(
            go.Bar(
                y=cat.index,
                x=cat.values / 1e3,
                orientation="h",
                marker_color=[BLUE, BLUE2, "#8697C4"],
                text=[f"${v/1e3:.0f}K" for v in cat.values],
                textposition="outside",
                textfont=dict(size=10, color=T2),
                hovertemplate="%{y}<br>$%{x:.0f}K<extra></extra>",
            )
        )
        fig3.update_layout(
            **PL,
            height=220,
