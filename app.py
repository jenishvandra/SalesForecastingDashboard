import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json, os
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="SalesIQ", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
*{margin:0;padding:0;box-sizing:border-box;}
.stApp{background:#0d1117;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebar"]{display:none!important;}
[data-testid="collapsedControl"]{display:none!important;}
.main .block-container{padding:0!important;max-width:100%!important;}
section.main>div{padding:0!important;}
.stButton>button{
    background:#21262d!important;border:1px solid #30363d!important;
    color:#c9d1d9!important;border-radius:6px!important;
    font-size:12px!important;font-family:Inter,sans-serif!important;
}
.stButton>button:hover{background:#3D52A0!important;color:white!important;}
.stRadio>div{display:flex!important;flex-direction:row!important;gap:6px!important;}
.stRadio>div>label{
    background:#21262d!important;border:1px solid #30363d!important;
    border-radius:6px!important;padding:6px 14px!important;
    font-size:12px!important;color:#8b949e!important;
}
.stRadio>div>label:has(input:checked){
    background:#7091E6!important;color:white!important;border-color:#7091E6!important;
}
.stRadio input{display:none!important;}
.stSlider>div>div>div{background:#7091E6!important;}
div[data-testid="stHorizontalBlock"]{gap:6px!important;}
[class*="st-key-nav_hidden_row"]{position:absolute!important;width:1px!important;
    height:1px!important;overflow:hidden!important;padding:0!important;margin:0!important;
    border:0!important;clip:rect(0,0,0,0)!important;}
[class*="st-key-seg_hidden_row"]{position:absolute!important;width:1px!important;
    height:1px!important;overflow:hidden!important;padding:0!important;margin:0!important;
    border:0!important;clip:rect(0,0,0,0)!important;}
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

# ── Compute data ───────────────────────────────────────────────────────────
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

# ── Serialize JSON ─────────────────────────────────────────────────────────
def safe(v): return None if (isinstance(v,float) and np.isnan(v)) else round(float(v),2)

monthly_json = json.dumps([{"x":str(r.Date)[:10],"y":safe(r.Sales/1e3)} for r in monthly.itertuples()])
ma3_json     = json.dumps([{"x":str(monthly.iloc[i]['Date'])[:10],"y":safe(ma3.iloc[i]/1e3)} for i in range(len(ma3))])
yr_json      = json.dumps([{"x":int(k),"y":round(float(v)/1e6,2)} for k,v in yr_rev.items()])
cat_json     = json.dumps([{"x":k,"y":round(float(v)/1e3,0)} for k,v in df.groupby('Category')['Sales'].sum().sort_values().items()])
reg_json     = json.dumps([{"x":k,"y":round(float(v)/1e3,0)} for k,v in df.groupby('Region')['Sales'].sum().sort_values().items()])

cmap_anom = {11:'Holiday season surge',12:'Christmas/year-end peak',
             1:'Post-holiday dip',2:'Post-holiday slowdown',
             3:'Promotional campaign',6:'Regional supply disruption',
             7:'Summer promotions',8:'Back-to-school demand',9:'End-of-quarter surge'}

def make_anom_tbl(an):
    out=[]
    for dt,val in an.items():
        mo=pd.to_datetime(dt).month
        out.append({"date":str(dt)[:10],"rev":f"${float(val):,.0f}",
            "sig":"spike" if float(val)>float(ws.mean()) else "drop",
            "cause":cmap_anom.get(mo,"Unusual fluctuation")})
    return out

iso_json = json.dumps([{"x":str(i)[:10],"y":safe(float(v)/1e3),"type":"spike" if float(v)>float(ws.mean()) else "drop"} for i,v in iso_an.items()])
z_json   = json.dumps([{"x":str(i)[:10],"y":safe(float(v)/1e3),"type":"spike" if float(v)>float(ws.mean()) else "drop"} for i,v in z_an.items()])
ws_json  = json.dumps([{"x":str(i)[:10],"y":safe(float(v)/1e3)} for i,v in ws.items()])
rm4_json = json.dumps([{"x":str(i)[:10],"y":safe(float(v)/1e3)} for i,v in rm4.items()])
clust_json=json.dumps([{"name":r['Sub-Category'],"x":round(float(r['px']),1),"y":round(float(r['py']),1),"seg":r['Segment']} for _,r in feat.iterrows()])

seg_cards={seg:sorted(feat[feat['Segment']==seg]['Sub-Category'].tolist())
           for seg in ['High Volume Stable','Growing Demand','High Volatility','Declining Demand']}

# ── SARIMA forecast (pure statsmodels — no compiled Stan binary, safe on any Python) ──
seg_choice = st.session_state.seg_choice
horizon    = st.session_state.horizon

@st.cache_data
def run_prophet(seg, hz):
    sdf = df[df['Category']==seg] if seg in ["Technology","Furniture","Office Supplies"] \
          else df[df['Region']==seg]
    sm  = sdf.groupby(pd.Grouper(key='Order Date',freq='ME'))['Sales'].sum().reset_index()
    sm.columns=['ds','y']

    model = SARIMAX(sm['y'].values, order=(1,1,1), seasonal_order=(1,1,0,12),
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    fitted_vals = fit.fittedvalues
    fc_res      = fit.get_forecast(steps=hz)
    future_mean = np.asarray(fc_res.predicted_mean)
    conf        = np.asarray(fc_res.conf_int(alpha=0.05))
    future_dates= pd.date_range(start=sm['ds'].iloc[-1], periods=hz+1, freq='ME')[1:]

    hist_df = pd.DataFrame({"ds":sm['ds'], "yhat":fitted_vals,
                             "yhat_lower":fitted_vals, "yhat_upper":fitted_vals})
    fut_df  = pd.DataFrame({"ds":future_dates, "yhat":future_mean,
                             "yhat_lower":conf[:,0], "yhat_upper":conf[:,1]})
    fc = pd.concat([hist_df, fut_df], ignore_index=True)
    return sm, fc

seg_m, fc = run_prophet(seg_choice, horizon)
fo        = fc.tail(horizon)
prev      = float(seg_m['y'].iloc[-1])
tp        = fc['yhat'].iloc[:len(seg_m)].values
mae       = float(np.mean(np.abs(seg_m['y'].values-tp)))
rmse_val  = float(np.sqrt(np.mean((seg_m['y'].values-tp)**2)))
m1v       = float(fo['yhat'].iloc[0])
m3v       = float(fo['yhat'].iloc[-1])

hist_json  = json.dumps([{"x":str(r.ds)[:10],"y":safe(r.y/1e3)} for r in seg_m.itertuples()])
fc_json    = json.dumps([{"x":str(r.ds)[:10],"y":safe(r.yhat/1e3),"lo":safe(r.yhat_lower/1e3),"hi":safe(r.yhat_upper/1e3)} for r in fo.itertuples()])
fc_tbl_json= json.dumps([{"month":pd.to_datetime(str(r.ds)[:10]).strftime('%b %y'),"fc":f"${float(r.yhat)/1e3:.0f}K","lo":f"${float(r.yhat_lower)/1e3:.0f}K","hi":f"${float(r.yhat_upper)/1e3:.0f}K"} for r in fo.itertuples()])

# ── Load & inject HTML ─────────────────────────────────────────────────────
page = st.session_state.page
tpl_path = os.path.join(os.path.dirname(__file__),'templates',f'{page}.html')
with open(tpl_path,'r') as f: html=f.read()

replacements = {
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
    "__ISO_TBL__":json.dumps(make_anom_tbl(iso_an)),
    "__Z_TBL__":json.dumps(make_anom_tbl(z_an)),
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
for k,v in replacements.items(): html=html.replace(k,v)

# ── Render HTML ────────────────────────────────────────────────────────────
components.html(html, height=920, scrolling=True)

# ── Navigation buttons (hidden helpers — sidebar clicks trigger these via JS) ──
with st.container(key="nav_hidden_row"):
    c1,c2,c3,c4 = st.columns(4)
    pages = [('overview','📊 Overview'),('forecast','📈 Forecasting'),
             ('anomaly','🔔 Anomalies'),('segments','🗂 Segments')]
    for col,(pg,lbl) in zip([c1,c2,c3,c4], pages):
        with col:
            if st.button(lbl, key=f'nav_{pg}', use_container_width=True,
                         type='primary' if page==pg else 'secondary'):
                st.session_state.page=pg; st.rerun()

# ── Forecast controls ──────────────────────────────────────────────────────
if page=='forecast':
    with st.container(key="seg_hidden_row"):
        segs=["Technology","Furniture","Office Supplies","West","East"]
        seg_cols = st.columns(len(segs))
        for col,s in zip(seg_cols, segs):
            with col:
                if st.button(s, key=f'segbtn_{s}', use_container_width=True,
                             type='primary' if s==seg_choice else 'secondary'):
                    st.session_state.seg_choice=s; st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    hz = st.slider("Forecast Horizon (months)", 1, 3, horizon)
    if hz!=horizon: st.session_state.horizon=hz; st.rerun()
