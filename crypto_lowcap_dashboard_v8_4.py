import json, math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Crypto Low-Cap TRUE BREAKOUT PRO v8.4", page_icon="₿", layout="wide")
DATA=Path("data/latest_scan.json")

def n(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else np.nan
    except: return np.nan
def pct(v):
    x=n(v); return "N/A" if pd.isna(x) else f"{x:+.1f}%"
def num(v,d=1):
    x=n(v); return "N/A" if pd.isna(x) else f"{x:.{d}f}"
def age(s):
    try:
        return (datetime.now(timezone.utc)-datetime.fromisoformat(str(s).replace("Z","+00:00"))).total_seconds()/60
    except: return np.nan
def load():
    if not DATA.exists(): return {}
    try: return json.loads(DATA.read_text())
    except: return {}
def rows(bg):
    a=bg.get("all") or bg.get("top10") or []
    if not a: return pd.DataFrame()
    d=pd.DataFrame(a)
    return d.rename(columns={
      "coin":"Coin","ticker":"Ticker","technical_score":"Technical score",
      "risk_adjusted_score":"Risk-adjusted score","breakout_state":"Breakout state",
      "breakout_confidence":"Breakout confidence","weighted_rsi":"Weighted RSI",
      "btc_rel_7d_pct":"BTC-rel 7d %","vol_mcap_pct":"Vol/MCap %",
      "downside_beta":"Downside beta","btc_down_day_rel_pct":"BTC-down-day rel %",
      "btc_down_day_outperform_pct":"BTC-down-day outperform %",
      "mcap_m":"MCap $M","price_change_24h_pct":"24h %",
      "daily_breakout":"Daily breakout","weekly_breakout":"Weekly breakout",
      "volume_confirmed":"Volume confirmed","obv_breakout":"OBV breakout",
      "atr_expanding":"ATR expanding","ema_bullish":"EMA bullish",
      "base_quality":"Base quality","supply_score":"Supply score",
      "circ_pct_max":"Circulating % max","fdv_mcap":"FDV/MCap",
      "unlock_risk":"Unlock risk","oi_1d_pct":"OI 1D %",
      "oi_3d_pct":"OI 3D %","oi_7d_pct":"OI 7D %",
      "funding":"Funding","derivatives_state":"Derivatives state",
      "stoch_k":"StochRSI %K","stoch_d":"StochRSI %D","adx":"ADX",
      "plus_di":"+DI","minus_di":"-DI","rsi_quality_pct":"RSI quality %"})
def heat(bg):
    out=[]
    for r in (bg.get("all") or bg.get("top10") or []):
        v=r.get("rsi") or r.get("mtf_rsi") or {}
        if not v: continue
        z={"Coin":r.get("coin",""),"Ticker":str(r.get("ticker","")).upper()}
        for tf in ["1H","4H","1D","1W","1M","3M"]: z[tf]=n(v.get(tf))
        z["Weighted RSI"]=n(r.get("weighted_rsi"))
        vals=[z[t] for t in ["1H","4H","1D","1W","1M","3M"] if pd.notna(z[t])]
        z["RSI coverage"]=f"{len(vals)}/6"; z["Deep oversold"]=sum(x<30 for x in vals)>=2
        z["Bullish alignment"]=sum(x>=60 for x in vals)>=4
        z["Bearish alignment"]=sum(x<=40 for x in vals)>=4
        out.append(z)
    return pd.DataFrame(out)

bg=load()
with st.sidebar:
    st.header("Dashboard controls")
    if st.button("🔄 Refresh background data",use_container_width=True): st.rerun()
    if st.button("🧹 Clear dashboard cache",use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.caption("Read-only presentation layer. Telegram alerts remain owned by the background scanner.")

st.title("₿ Crypto Low-Cap TRUE BREAKOUT PRO v8.4")
st.caption("Market regime → breakout radar → multi-timeframe RSI → coin deep dive")
if not bg:
    st.warning("No background scan is available yet. Run the GitHub Actions scanner first."); st.stop()

meta=bg.get("meta",{}); btc=bg.get("btc_market",{}); df=rows(bg); a=age(meta.get("generated_at"))
fresh="🟢 FRESH" if pd.notna(a) and a<=90 else ("🟡 AGING" if pd.notna(a) and a<=240 else "🔴 STALE")

st.subheader("🌐 General Crypto Market Direction")
c=st.columns(8)
c[0].metric("BTC Direction",str(btc.get("signal","N/A")).replace("🟢 ","").replace("🔴 ","").replace("🟡 ","").replace("🟠 ","").replace("⚪ ",""))
c[1].metric("BTC Price",f"$ {n(meta.get('btc_price_usd')):,.0f}" if pd.notna(n(meta.get('btc_price_usd'))) else "N/A")
c[2].metric("BTC 24h",pct(meta.get("btc_24h"))); c[3].metric("BTC 7d",pct(meta.get("btc_7d")))
c[4].metric("BTC Dominance",f"{num(meta.get('btc_dominance_pct'),2)}%"); c[5].metric("USDT Dominance",f"{num(meta.get('usdt_dominance_pct'),2)}%")
c[6].metric("Fear & Greed",str(meta.get("fear_greed_value","N/A"))); c[7].metric("Data",fresh)
st.caption(f"Engine {meta.get('engine_version','N/A')} | Generated {meta.get('generated_at','N/A')} | Age {num(a,0)} min | Regime {meta.get('regime','N/A')} | BTC dominance {meta.get('btc_dominance_trend','N/A')} | USDT dominance {meta.get('usdt_dominance_trend','N/A')} | TOTAL3/BTC {meta.get('total3_btc_trend','N/A')} | F&G {meta.get('fear_greed_classification','N/A')} | Breadth {num(meta.get('breadth'),1)}%")
r=st.columns(6)
r[0].metric("BTC Weighted RSI",num(btc.get("weighted_rsi"))); r[1].metric("BTC ADX",num(btc.get("adx"))); r[2].metric("BTC +DI",num(btc.get("plus_di"))); r[3].metric("BTC -DI",num(btc.get("minus_di"))); r[4].metric("BTC Stoch K",num(btc.get("stoch_k"))); r[5].metric("BTC EMA","BULLISH" if btc.get("ema_bullish") else "NOT BULLISH")

st.subheader("🚀 Breakout Radar")
if not df.empty:
    states=["🟡 BREAKOUT ATTEMPT","🟢 BREAKOUT ACCEPTED","🟢 RETEST HELD","🟢 TRUE BREAKOUT"]
    radar=df[df.get("Breakout state",pd.Series(index=df.index)).isin(states)].copy()
    if radar.empty:
        mask=pd.Series(False,index=df.index)
        for q in ["Daily breakout","Weekly breakout"]:
            if q in df: mask|=df[q].fillna(False)
        radar=df[mask].copy()
    if "Risk-adjusted score" in radar: radar=radar.sort_values("Risk-adjusted score",ascending=False)
    cols=[x for x in ["Coin","Ticker","Breakout state","Risk-adjusted score","Technical score","Breakout confidence","Weighted RSI","BTC-rel 7d %","Vol/MCap %","Daily breakout","Weekly breakout","Volume confirmed","OBV breakout","ATR expanding","EMA bullish"] if x in radar]
    st.dataframe(radar[cols].round(2),use_container_width=True,hide_index=True)
else: st.info("No coin-level records are available.")

st.subheader("📊 Multi-Timeframe RSI")
h=heat(bg)
if h.empty: st.info("The background file does not contain per-timeframe RSI values.")
else:
    f=st.selectbox("RSI view",["All","Deep oversold","Bullish alignment","Bearish alignment"])
    if f!="All": h=h[h[f]]
    st.dataframe(h.round(1),use_container_width=True,hide_index=True)
    st.caption("RSI: <30 oversold, 30–40 weak, 40–60 neutral, 60–70 strong, >70 overbought. Coverage prevents missing data being mistaken for a signal.")

st.subheader("🎯 Coin Deep Dive")
if not df.empty:
    choices=sorted(df["Ticker"].dropna().astype(str).unique()); default=0
    t=st.selectbox("Select a coin",choices,index=default)
    x=df[df["Ticker"].astype(str).str.upper().eq(t.upper())].head(1)
    if not x.empty:
        z=x.iloc[0]; m=st.columns(6)
        m[0].metric("Technical score",num(z.get("Technical score"))); m[1].metric("Risk-adjusted",num(z.get("Risk-adjusted score"))); m[2].metric("Confidence",num(z.get("Breakout confidence"))); m[3].metric("Weighted RSI",num(z.get("Weighted RSI"))); m[4].metric("BTC-rel 7d",pct(z.get("BTC-rel 7d %"))); m[5].metric("Vol/MCap",f"{num(z.get('Vol/MCap %'))}%")
        l,r=st.columns(2)
        with l:
            st.markdown("**Structure & momentum**")
            q=[x for x in ["Breakout state","Daily breakout","Weekly breakout","Volume confirmed","OBV breakout","ATR expanding","EMA bullish","Base quality","MCap $M","24h %"] if x in z.index]
            st.dataframe(pd.DataFrame({"Metric":q,"Value":[z[x] for x in q]}),hide_index=True,use_container_width=True)
        with r:
            st.markdown("**Risk & derivatives**")
            q=[x for x in ["Downside beta","BTC-down-day rel %","BTC-down-day outperform %","OI 1D %","OI 3D %","OI 7D %","Funding","Derivatives state","Supply score","Circulating % max","FDV/MCap","Unlock risk"] if x in z.index]
            st.dataframe(pd.DataFrame({"Metric":q,"Value":[z[x] for x in q]}),hide_index=True,use_container_width=True)

st.subheader("🧪 Scan Quality & Data Health")
q=meta.get("quality_counts",{}) or {}; c=st.columns(5)
c[0].metric("Candidates",meta.get("candidate_count","N/A")); c[1].metric("Pre-screen",meta.get("pre_screen_count","N/A")); c[2].metric("Ranked",meta.get("ranked_count",len(df))); c[3].metric("RSI 6/6",q.get("rsi_full","N/A")); c[4].metric("RSI 0/6",q.get("rsi_zero","N/A"))
st.caption("v8.4 is a presentation layer over the latest background scan. The background engine remains responsible for collection, scoring, breakout-state transitions and Telegram alerting.")
