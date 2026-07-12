import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json, os
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="SalesIQ", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{font-family:'Inter',sans-serif!important;}
.stApp{background:#0d1117!important;}
#MainMenu,footer,header{visibility:hidden;}

/* Sidebar */
[data-testid="stSidebar"]{
    background:#161b22!important;
    border-right:1px solid #30363d!important;
    min-width:220px!important; max-width:220px!important;
}
[data-testid="stSidebar"] *{color:#8b949e!important;}
[data-testid="collapsedControl"]{display:none!important;}

/* Sidebar buttons as nav items */
[data-testid="stSidebar"] .stButton>button{
    width:100%!important;
    background:transparent!important;
    border:none!important;
    border-right:2px solid transparent!important;
    border-radius:0!important;
    padding:10px 16px!important;
    text-align:left!important;
    font-size:13px!important;
    font-weight:500!important;
    color:#8b949e!important;
    cursor:pointer!important;
    transition:all .15s!important;
    display:flex!important;
    align-items:center!important;
}
[data-testid="stSidebar"] .stButton>button:hover{
    background:#21262d!important;
    color:#e6edf3!important;
}
[data-testid="stSidebar"] .stButton[data-active="true"]>button,
[data-testid="stSidebar"] .stButton>button[kind="primary"]{
    background:#21262d!important;
    color:#7091E6!important;
    border-right:2px solid #7091E6!important;
}

/* Main content */
.main .block-container{
    padding:0 24px 24px 24px!important;
    max-width:100%!important;
}

/* Bottom nav buttons - HIDE */
div[data-testid="stHorizontalBlock"]:last-of-type{display:none!important;}

/* iframe full */
iframe{border:none!important; display:block!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if 'page'       not in st.session_state: st.session_state.page       = 'overview'
if 'seg_choice' not in st.session_state: st.session_state.seg_choice = 'Technology'
if 'horizon'    not in st.session_state: st.session_state.horizon    = 3

# ── Data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Year']     = df['Order Date'].dt.year
    df['Month']    = df['Order Date'].dt.month
    df['Ship_Days']= (df['Ship Date']-df['Order Date']).dt.days
    return df

df      = load_data()
monthly = df.groupby(pd.Grouper(key='Order Date',freq='ME'))['Sales'].sum().reset_index()
monthly.columns=['Date','Sales']
weekly  = df.groupby(pd.Grouper(key='Order Date',freq='W'))['Sales'].sum().reset_index()
weekly.columns=['Date','Sales']
yr_rev  = df.groupby('Year')['Sales'].sum()
yoy     = float((yr_rev.iloc[-1]-yr_rev.iloc[-2])/yr_rev.iloc[-2]*100)

# ── Compute ────────────────────────────────────────────────────────────────
ma3  = monthly['Sales'].rolling(3).mean()
pidx = int(monthly['Sales'].idxmax())

ws      = weekly.set_index('Date')['Sales']
iso     = IsolationForest(contamination=0.07, random_state=42)
iso_lbl = iso.fit_predict(ws.values.reshape(-1,1))
iso_an  = ws[iso_lbl==-1]
rm_     = ws.rolling(8,center=True).mean()
rs_     = ws.rolling(8,center=True).std()
z_      = (ws-rm_)/rs_
z_an    = ws[z_.abs()>2]
both_ct = len(set(iso_an.index.date)&set(z_an.index.date))
rm4     = ws.rolling(4).mean()

feat = df.groupby('Sub-Category').agg(
    Total_Sales=('Sales','sum'),Avg_Order=('Sales','mean'),
    Volatility=('Sales','std'),Count=('Sales','count')).reset_index()
yf   = df.groupby(['Sub-Category','Year'])['Sales'].sum().unstack()
feat['Growth'] = yf.pct_change(axis=1).mean(axis=1).values
feat = feat.dropna()
Xs   = StandardScaler().fit_transform(feat[['Total_Sales','Avg_Order','Volatility','Growth']])
km   = KMeans(n_clusters=4,random_state=42,n_init=10)
feat['Cluster'] = km.fit_predict(Xs)
sc   = feat.groupby('Cluster')['Total_Sales'].mean().sort_values(ascending=False).index
lmp  = {sc[0]:'High Volume Stable',sc[1]:'Growing Demand',
        sc[2]:'High Volatility',sc[3]:'Declining Demand'}
feat['Segment'] = feat['Cluster'].map(lmp)
pca  = PCA(n_components=2); Xp = pca.fit_transform(Xs)
xn   = (Xp[:,0]-Xp[:,0].min())/(Xp[:,0].max()-Xp[:,0].min())*100
yn   = (Xp[:,1]-Xp[:,1].min())/(Xp[:,1].max()-Xp[:,1].min())*100
feat['px']=xn; feat['py']=yn

def safe(v):
    if v is None: return None
    try:
        f = float(v)
        return None if np.isnan(f) else round(f,2)
    except: return None

# ── JSON ───────────────────────────────────────────────────────────────────
monthly_json = json.dumps([{"x":str(r.Date)[:10],"y":safe(r.Sales/1e3)} for r in monthly.itertuples()])
ma3_json     = json.dumps([{"x":str(monthly.iloc[i]['Date'])[:10],"y":safe(ma3.iloc[i]/1e3)} for i in range(len(ma3))])
yr_json      = json.dumps([{"x":int(k),"y":round(float(v)/1e6,2)} for k,v in yr_rev.items()])
cat_json     = json.dumps([{"x":k,"y":round(float(v)/1e3,0)} for k,v in df.groupby('Category')['Sales'].sum().sort_values().items()])
reg_json     = json.dumps([{"x":k,"y":round(float(v)/1e3,0)} for k,v in df.groupby('Region')['Sales'].sum().sort_values().items()])

cmap_anom={11:'Holiday season surge',12:'Christmas/year-end peak',
           1:'Post-holiday dip',2:'Post-holiday slowdown',
           3:'Promotional campaign',6:'Regional supply disruption',
           7:'Summer promotions',8:'Back-to-school demand',9:'End-of-quarter surge'}

def make_anom(an):
    return json.dumps([{"x":str(i)[:10],"y":safe(float(v)/1e3),
        "type":"spike" if float(v)>float(ws.mean()) else "drop"} for i,v in an.items()])

def make_tbl(an):
    out=[]
    for dt,val in an.items():
        mo=pd.to_datetime(dt).month
        out.append({"date":str(dt)[:10],"rev":f"${float(val):,.0f}",
            "sig":"spike" if float(val)>float(ws.mean()) else "drop",
            "cause":cmap_anom.get(mo,"Unusual fluctuation")})
    return json.dumps(out)

iso_json = make_anom(iso_an); z_json = make_anom(z_an)
ws_json  = json.dumps([{"x":str(i)[:10],"y":safe(float(v)/1e3)} for i,v in ws.items()])
rm4_json = json.dumps([{"x":str(i)[:10],"y":safe(float(v)/1e3)} for i,v in rm4.items()])
clust_json=json.dumps([{"name":r['Sub-Category'],"x":round(float(r['px']),1),
    "y":round(float(r['py']),1),"seg":r['Segment']} for _,r in feat.iterrows()])
seg_cards={seg:sorted(feat[feat['Segment']==seg]['Sub-Category'].tolist())
           for seg in ['High Volume Stable','Growing Demand','High Volatility','Declining Demand']}

# ── Prophet ────────────────────────────────────────────────────────────────
@st.cache_data
def run_prophet(seg,hz):
    sdf = df[df['Category']==seg] if seg in ["Technology","Furniture","Office Supplies"] \
          else df[df['Region']==seg]
    sm  = sdf.groupby(pd.Grouper(key='Order Date',freq='ME'))['Sales'].sum().reset_index()
    sm.columns=['ds','y']
    m   = Prophet(yearly_seasonality=True,weekly_seasonality=False,
                  daily_seasonality=False,seasonality_mode='additive')
    m.fit(sm); fut=m.make_future_dataframe(periods=hz,freq='ME'); fc=m.predict(fut)
    return sm,fc

seg_choice=st.session_state.seg_choice; horizon=st.session_state.horizon
seg_m,fc  = run_prophet(seg_choice,horizon)
fo        = fc.tail(horizon); prev=float(seg_m['y'].iloc[-1])
tp        = fc['yhat'].iloc[:len(seg_m)].values
mae       = float(np.mean(np.abs(seg_m['y'].values-tp)))
rmse_val  = float(np.sqrt(np.mean((seg_m['y'].values-tp)**2)))
m1v       = float(fo['yhat'].iloc[0]); m3v=float(fo['yhat'].iloc[-1])

hist_json  = json.dumps([{"x":str(r.ds)[:10],"y":safe(r.y/1e3)} for r in seg_m.itertuples()])
fc_json    = json.dumps([{"x":str(r.ds)[:10],"y":safe(r.yhat/1e3),"lo":safe(r.yhat_lower/1e3),"hi":safe(r.yhat_upper/1e3)} for r in fo.itertuples()])
fc_tbl_json= json.dumps([{"month":pd.to_datetime(str(r.ds)[:10]).strftime('%b %y'),
    "fc":f"${float(r.yhat)/1e3:.0f}K","lo":f"${float(r.yhat_lower)/1e3:.0f}K",
    "hi":f"${float(r.yhat_upper)/1e3:.0f}K"} for r in fo.itertuples()])

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:20px 16px 16px;border-bottom:1px solid #30363d;'>
        <div style='font-size:20px;font-weight:800;color:#e6edf3;letter-spacing:-.5px;'>
            Sales<span style='color:#7091E6;'>IQ</span></div>
        <div style='font-size:10px;color:#8b949e;margin-top:2px;'>Analytics Platform</div>
    </div>
    <div style='font-size:10px;color:#8b949e;text-transform:uppercase;
        letter-spacing:1px;font-weight:600;padding:12px 16px 4px;'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.session_state.page

    # Nav items — styled as sidebar links
    nav_items = [
        ('overview',  '📊', 'Sales Overview'),
        ('forecast',  '📈', 'Forecasting'),
        ('anomaly',   '🔔', 'Anomaly Report'),
        ('segments',  '🗂', 'Demand Segments'),
    ]
    for key, icon, label in nav_items:
        is_active = (page == key)
        style = f"""
            display:flex;align-items:center;gap:10px;
            padding:10px 16px;font-size:13px;font-weight:500;
            cursor:pointer;border-right:2px solid {'#7091E6' if is_active else 'transparent'};
            background:{'#21262d' if is_active else 'transparent'};
            color:{'#7091E6' if is_active else '#8b949e'};
            border-radius:0;transition:all .15s;
        """
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown(f"""
    <div style='border-top:1px solid #30363d;padding:14px 16px 8px;margin-top:8px;'>
        <div style='font-size:10px;color:#8b949e;text-transform:uppercase;
            letter-spacing:1px;font-weight:600;margin-bottom:10px;'>Quick Stats</div>
        <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #21262d;'>
            <span style='font-size:11px;color:#8b949e;'>Revenue</span>
            <span style='font-size:11px;font-weight:700;color:#c9d1d9;'>${df['Sales'].sum()/1e6:.2f}M</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #21262d;'>
            <span style='font-size:11px;color:#8b949e;'>Orders</span>
            <span style='font-size:11px;font-weight:700;color:#c9d1d9;'>{len(df):,}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #21262d;'>
            <span style='font-size:11px;color:#8b949e;'>YoY Growth</span>
            <span style='font-size:11px;font-weight:700;color:#3fb950;'>+{yoy:.1f}%</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #21262d;'>
            <span style='font-size:11px;color:#8b949e;'>Avg Order</span>
            <span style='font-size:11px;font-weight:700;color:#c9d1d9;'>${df['Sales'].mean():,.0f}</span>
        </div>
        <div style='display:flex;justify-content:space-between;padding:5px 0;'>
            <span style='font-size:11px;color:#8b949e;'>Period</span>
            <span style='font-size:11px;font-weight:700;color:#c9d1d9;'>{df['Year'].min()}–{df['Year'].max()}</span>
        </div>
    </div>
    <div style='display:flex;align-items:center;gap:6px;padding:10px 16px;
        font-size:11px;color:#3fb950;font-weight:500;border-top:1px solid #30363d;'>
        <div style='width:7px;height:7px;border-radius:50%;background:#3fb950;
            animation:pulse 2s infinite;'></div>
        LIVE · Updated just now
    </div>
    <style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}</style>
    """, unsafe_allow_html=True)

    # Forecast controls in sidebar when on forecast page
    if page == 'forecast':
        st.markdown("<hr style='border-color:#30363d;margin:8px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;font-weight:600;padding:0 0 8px;'>Segment</div>", unsafe_allow_html=True)
        segs = ["Technology","Furniture","Office Supplies","West","East"]
        for s in segs:
            is_s = (s == seg_choice)
            style_s = f"display:block;width:100%;padding:7px 12px;font-size:12px;cursor:pointer;border-radius:6px;margin-bottom:3px;border:1px solid {'#7091E6' if is_s else '#30363d'};background:{'#7091E6' if is_s else '#21262d'};color:{'white' if is_s else '#8b949e'};"
            if st.button(s, key=f"seg_{s}", use_container_width=True):
                st.session_state.seg_choice = s
                st.rerun()
        st.markdown("<div style='font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;font-weight:600;padding:10px 0 6px;'>Forecast Horizon</div>", unsafe_allow_html=True)
        hz = st.slider("", 1, 3, horizon, label_visibility='collapsed', key='hz_slider')
        if hz != horizon:
            st.session_state.horizon = hz
            st.rerun()

# ── Load HTML ──────────────────────────────────────────────────────────────
page = st.session_state.page
for tpl_dir in ['templates','templates/templates']:
    tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), tpl_dir, f'{page}.html')
    if os.path.exists(tpl_path): break

with open(tpl_path,'r') as f: html=f.read()

repl = {
    "__MONTHLY__":monthly_json,"__MA3__":ma3_json,"__YR__":yr_json,
    "__CAT__":cat_json,"__REG__":reg_json,
    "__PEAK_DATE__":f'"{str(monthly.iloc[pidx]["Date"])[:10]}"',
    "__PEAK_VAL__":str(round(float(monthly.iloc[pidx]['Sales'])/1e3,2)),
    "__TOTAL_REV__":f'"${float(df["Sales"].sum())/1e6:.2f}M"',
    "__TOTAL_ORD__":f'"{len(df):,}"',
    "__AVG_ORD__":f'"${float(df["Sales"].mean()):,.0f}"',
    "__AVG_SHIP__":f'"{float(df["Ship_Days"].mean()):.1f} days"',
    "__YOY__":f'"{yoy:.1f}"',
    "__YR_MIN__":str(int(df['Year'].min())),"__YR_MAX__":str(int(df['Year'].max())),
    "__ISO_ANOM__":iso_json,"__Z_ANOM__":z_json,
    "__WS__":ws_json,"__RM4__":rm4_json,
    "__ISO_CT__":str(len(iso_an)),"__Z_CT__":str(len(z_an)),
    "__BOTH_CT__":str(both_ct),"__WEEKS_CT__":str(len(ws)),
    "__ISO_TBL__":make_tbl(iso_an),"__Z_TBL__":make_tbl(z_an),
    "__CLUST__":clust_json,
    "__SEG_HVS__":json.dumps(seg_cards['High Volume Stable']),
    "__SEG_GD__":json.dumps(seg_cards['Growing Demand']),
    "__SEG_HV__":json.dumps(seg_cards['High Volatility']),
    "__SEG_DD__":json.dumps(seg_cards['Declining Demand']),
    "__HIST__":hist_json,"__FC__":fc_json,"__FC_TBL__":fc_tbl_json,
    "__SEG_CHOICE__":f'"{seg_choice}"',
    "__MAE__":f'"${mae/1e3:.1f}K"',"__RMSE__":f'"${rmse_val/1e3:.1f}K"',
    "__M1V__":f'"${m1v/1e3:.0f}K"',"__M3V__":f'"${m3v/1e3:.0f}K"',
    "__M1PCT__":f'"+{(m1v-prev)/prev*100:.1f}% vs prior"',
    "__M3PCT__":f'"+{(m3v-prev)/prev*100:.1f}% vs current"',
}
for k,v in repl.items(): html=html.replace(k,v)

# Remove sidebar from HTML (Streamlit sidebar handles it)
# Keep only content div from HTML
import re
# Extract just body content between <body> and </body>
body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL)
if body_match:
    body_content = body_match.group(1)
    # Remove the sidebar div from HTML
    body_content = re.sub(r'<div class="sidebar">.*?</div>\s*\n\s*<div class="content">', '<div class="content" style="padding:0;">', body_content, flags=re.DOTALL)
    # Rebuild minimal HTML
    style_match = re.search(r'<style>(.*?)</style>', html, re.DOTALL)
    style = style_match.group(1) if style_match else ''
    script_tags = re.findall(r'<script[^>]*>.*?</script>', html, re.DOTALL)
    
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*{{font-family:'Inter',sans-serif;box-sizing:border-box;margin:0;padding:0;}}
body{{background:#0d1117;color:#e6edf3;}}
{style}
.content{{padding:20px 24px!important;}}
</style></head>
<body>{body_content}</body></html>"""

components.html(html, height=880, scrolling=True)
