import json, os, time, random
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import ccxt

VERSION = "v8.2.5"
CG = "https://api.coingecko.com/api/v3"
ZEN_ID = "horizen"
WEIGHTS = {"1H": .05, "4H": .10, "1D": .20, "1W": .25, "1M": .20, "3M": .20}
CACHE = {}

def cg_get(path, params=None):
    headers = {"accept": "application/json"}
    key = os.getenv("COINGECKO_API_KEY")
    if key: headers["x-cg-demo-api-key"] = key
    last = None
    for i in range(5):
        try:
            r = requests.get(f"{CG}/{path.lstrip('/')}", params=params or {}, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(min(60, 2**i) + random.uniform(.2, 1.0)); continue
            r.raise_for_status(); return r.json()
        except Exception as e:
            last = e; time.sleep(min(30, 2**i) + random.uniform(.2, 1.0))
    raise last

def markets():
    frames=[]
    for page in (1,2):
        frames.append(pd.DataFrame(cg_get("coins/markets", {"vs_currency":"usd","order":"market_cap_desc","per_page":250,"page":page,"sparkline":"false","price_change_percentage":"24h,7d"})))
    try:
        zen=pd.DataFrame(cg_get("coins/markets", {"vs_currency":"usd","ids":ZEN_ID,"sparkline":"false","price_change_percentage":"24h,7d"}))
        if not zen.empty: frames.append(zen)
    except Exception as e:
        print(f"ZEN direct fetch failed: {type(e).__name__}: {e}")
    return pd.concat(frames,ignore_index=True).drop_duplicates("id")

def btc_dominance():
    current=np.nan; d1=np.nan; d7=np.nan
    try:
        g=cg_get("global")
        current=float(g.get("data",{}).get("market_cap_percentage",{}).get("btc",np.nan))
    except Exception as e:
        print(f"BTC dominance current failed: {type(e).__name__}: {e}")
    # CoinGecko global market-cap history includes historical BTC dominance.
    try:
        h=cg_get("global/market_cap_chart",{"vs_currency":"usd","days":"7"})
        pct=h.get("market_cap_percentage",{})
        btc_hist=pct.get("btc") or pct.get("bitcoin") or []
        if btc_hist:
            vals=pd.Series([float(x[1]) for x in btc_hist if len(x)>=2])
            if len(vals):
                d1=float(vals.iloc[-1]-vals.iloc[max(0,len(vals)-24*1)-1]) if len(vals)>24 else float(vals.iloc[-1]-vals.iloc[0])
                d7=float(vals.iloc[-1]-vals.iloc[0])
                current=float(vals.iloc[-1]) if pd.isna(current) else current
    except Exception as e:
        print(f"BTC dominance history failed: {type(e).__name__}: {e}")
    if pd.isna(current): return {"current":np.nan,"change_1d":np.nan,"change_7d":np.nan,"trend":"N/A"}
    if pd.notna(d7) and d7>=0.50: trend="📈 BTC DOMINANCE RISING"
    elif pd.notna(d7) and d7<=-0.50: trend="📉 BTC DOMINANCE FALLING"
    else: trend="➡️ BTC DOMINANCE FLAT"
    return {"current":current,"change_1d":d1,"change_7d":d7,"trend":trend}

def fear_greed():
    """Fetch the latest Crypto Fear & Greed Index from Alternative.me."""
    try:
        r=requests.get("https://api.alternative.me/fng/",params={"limit":2,"format":"json"},timeout=20)
        r.raise_for_status(); j=r.json(); rows=j.get("data",[]) or []
        if not rows: raise ValueError("No Fear & Greed data returned")
        cur=rows[0]; value=float(cur.get("value",np.nan)); classification=str(cur.get("value_classification","N/A"))
        prev=float(rows[1].get("value",np.nan)) if len(rows)>1 else np.nan
        change=value-prev if pd.notna(value) and pd.notna(prev) else np.nan
        return {"value":value,"classification":classification,"change_1d":change,"timestamp":cur.get("timestamp")}
    except Exception as e:
        print(f"Fear & Greed failed: {type(e).__name__}: {e}")
        return {"value":np.nan,"classification":"N/A","change_1d":np.nan,"timestamp":None}

def total3_btc():
    """Estimate TOTAL3/BTC from CoinGecko global market-cap history.
    TOTAL3 = total crypto market cap - BTC market cap - ETH market cap.
    Ratio = TOTAL3 market cap / BTC market cap.
    """
    ratio=np.nan; d1=np.nan; d7=np.nan
    try:
        h=cg_get("global/market_cap_chart",{"vs_currency":"usd","days":"7"})
        total_hist=h.get("market_cap",[]) or h.get("market_caps",[])
        pct=h.get("market_cap_percentage",{})
        btc_pct=pct.get("btc") or pct.get("bitcoin") or []
        eth_pct=pct.get("eth") or pct.get("ethereum") or []
        if total_hist and btc_pct and eth_pct:
            total=pd.DataFrame(total_hist,columns=["ts","total"]).drop_duplicates("ts").sort_values("ts")
            bp=pd.DataFrame(btc_pct,columns=["ts","btc_pct"]).drop_duplicates("ts").sort_values("ts")
            ep=pd.DataFrame(eth_pct,columns=["ts","eth_pct"]).drop_duplicates("ts").sort_values("ts")
            z=total.merge(bp,on="ts",how="inner").merge(ep,on="ts",how="inner")
            if len(z)>=2:
                z["btc_mcap"]=z["total"]*z["btc_pct"]/100.0
                z["total3"]=z["total"]*(1.0-(z["btc_pct"]+z["eth_pct"])/100.0)
                z["ratio"]=z["total3"]/z["btc_mcap"].replace(0,np.nan)
                z=z.replace([np.inf,-np.inf],np.nan).dropna(subset=["ratio"])
                if len(z):
                    ratio=float(z.iloc[-1].ratio)
                    end_ts=pd.to_datetime(z.iloc[-1].ts,unit="ms",utc=True)
                    target1=end_ts-pd.Timedelta(days=1)
                    target7=end_ts-pd.Timedelta(days=7)
                    i1=(pd.to_datetime(z.ts,unit="ms",utc=True)-target1).abs().idxmin()
                    i7=(pd.to_datetime(z.ts,unit="ms",utc=True)-target7).abs().idxmin()
                    r1=float(z.loc[i1,"ratio"]); r7=float(z.loc[i7,"ratio"])
                    d1=(ratio/r1-1)*100 if r1 else np.nan
                    d7=(ratio/r7-1)*100 if r7 else np.nan
    except Exception as e:
        print(f"TOTAL3/BTC failed: {type(e).__name__}: {e}")
    if pd.isna(ratio): return {"ratio":np.nan,"change_1d_pct":np.nan,"change_7d_pct":np.nan,"trend":"N/A"}
    if pd.notna(d7) and d7>=5: trend="📈 TOTAL3/BTC RISING"
    elif pd.notna(d7) and d7<=-5: trend="📉 TOTAL3/BTC FALLING"
    else: trend="➡️ TOTAL3/BTC FLAT"
    return {"ratio":ratio,"change_1d_pct":d1,"change_7d_pct":d7,"trend":trend}

def exchange(name):
    k=f"ex:{name}"
    if k in CACHE: return CACHE[k]
    try:
        ex=getattr(ccxt,name)({"enableRateLimit":True,"timeout":30000})
        ex.load_markets(); CACHE[k]=ex; return ex
    except Exception as e:
        print(f"Exchange unavailable: {name}: {type(e).__name__}: {e}")
        CACHE[k]=None; return None

def symbol(ex,ticker):
    if ex is None: return None
    t=str(ticker).upper()
    for s in (f"{t}/USDT",f"{t}/USDT:USDT",f"{t}/USDC",f"{t}/USD"):
        m=ex.markets.get(s)
        if m and m.get("active",True) and (m.get("spot") or s.endswith(":USDT")): return s
    for s,m in ex.markets.items():
        if str(m.get("base","")).upper()==t and str(m.get("quote","")).upper()=="USDT" and m.get("active",True) and m.get("spot"): return s
    return None

def ohlcv(ex,sym,tf,limit):
    if ex is None or not sym: raise ValueError("exchange or symbol unavailable")
    k=("ohlcv",ex.id,sym,tf,limit)
    if k in CACHE:return CACHE[k].copy()
    d=pd.DataFrame(ex.fetch_ohlcv(sym,tf,limit=limit),columns=["ts","open","high","low","close","volume"])
    d["ts"]=pd.to_datetime(d.ts,unit="ms",utc=True); d=d.drop_duplicates("ts").sort_values("ts")
    if len(d)>1:d=d.iloc[:-1].copy()
    CACHE[k]=d; return d.copy()

def rsi(s,n=14):
    s=pd.Series(s,dtype=float); d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); al=l.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    out=100-(100/(1+ag/al.replace(0,np.nan)))
    out=out.where(~((al==0)&(ag>0)),100).where(~((al==0)&(ag==0)),50)
    return float(out.iloc[-1]) if len(out) and pd.notna(out.iloc[-1]) else np.nan

def mtf_rsi(exchanges,ticker):
    vals={k:np.nan for k in WEIGHTS}; sources={k:"N/A" for k in WEIGHTS}
    if not isinstance(exchanges,(list,tuple)): exchanges=[exchanges]
    exchanges=[e for e in exchanges if e is not None]
    def use(tf_key, tf_candidates, min_len=15):
        for ex in exchanges:
            sym=symbol(ex,ticker)
            if not sym: continue
            for tf in tf_candidates:
                try:
                    d=ohlcv(ex,sym,tf,300 if tf in ("1h","4h") else 1000)
                    if len(d)>=min_len:
                        v=rsi(d.close)
                        if pd.notna(v):
                            vals[tf_key]=v; sources[tf_key]=ex.id.upper(); return True
                except Exception: continue
        return False
    use("1H",("1h",),15); use("4H",("4h",),15)
    if not use("1D",("1d",),15): return vals,sources
    for ex in exchanges:
        sym=symbol(ex,ticker)
        if not sym: continue
        try:
            d=ohlcv(ex,sym,"1d",1000); s=d.set_index("ts").close.astype(float).dropna().sort_index()
            for key,rule in (("1W","W-SUN"),("1M","ME"),("3M","QE")):
                if pd.isna(vals[key]):
                    q=s.resample(rule).last().dropna()
                    if len(q)>=15:
                        v=rsi(q)
                        if pd.notna(v): vals[key]=v; sources[key]=ex.id.upper()
        except Exception: continue
    use("1M",("1M",),15); use("3M",("3M",),15)
    return vals,sources

def weighted(vals):
    u=[(v,WEIGHTS[k]) for k,v in vals.items() if pd.notna(v)]
    return float(np.average([v for v,w in u],weights=[w for v,w in u])) if u else np.nan

def atr(d,n=14):
    pc=d.close.shift(1); tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def adx(d,n=14):
    up=d.high.diff(); dn=-d.low.diff(); plus=up.where((up>dn)&(up>0),0); minus=dn.where((dn>up)&(dn>0),0)
    tr=pd.concat([d.high-d.close.shift(),d.low-d.close.shift(),d.high-d.low],axis=1).abs().max(axis=1)
    av=tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    pdi=100*plus.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/av; mdi=100*minus.ewm(alpha=1/n,adjust=False,min_periods=n).mean()/av
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); a=dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return (float(a.iloc[-1]),float(pdi.iloc[-1]),float(mdi.iloc[-1]),float(a.iloc[-2])) if len(a)>=2 and pd.notna(a.iloc[-1]) else (np.nan,np.nan,np.nan,np.nan)

def stoch_rsi(close):
    s=pd.Series(close,dtype=float); d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); al=l.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rv=100-(100/(1+ag/al.replace(0,np.nan))); rv=rv.where(~((al==0)&(ag>0)),100).where(~((al==0)&(ag==0)),50)
    lo=rv.rolling(14,min_periods=14).min(); hi=rv.rolling(14,min_periods=14).max()
    x=((rv-lo)/(hi-lo).replace(0,np.nan)*100).where((hi-lo)!=0,50); k=x.rolling(3,min_periods=3).mean(); q=k.rolling(3,min_periods=3).mean()
    return (float(k.iloc[-1]),float(q.iloc[-1])) if pd.notna(k.iloc[-1]) and pd.notna(q.iloc[-1]) else (np.nan,np.nan)

def metrics(d,w):
    if len(d)<30 or len(w)<25:return {}
    resistance=float(d.high.iloc[-21:-1].max()); wres=float(w.high.iloc[-21:-1].max()); close=float(d.close.iloc[-1]); avgvol=float(d.volume.iloc[-21:-1].mean()); vr=float(d.volume.iloc[-1])/avgvol if avgvol else np.nan
    a=atr(d); av=float(a.iloc[-1]); ap=float(a.iloc[-2]); obv=(np.sign(d.close.diff()).fillna(0)*d.volume).cumsum()
    e20=d.close.ewm(span=20,adjust=False).mean().iloc[-1]; e50=d.close.ewm(span=50,adjust=False).mean().iloc[-1]; e200=d.close.ewm(span=200,adjust=False).mean().iloc[-1] if len(d)>=200 else np.nan
    k,sd=stoch_rsi(d.close); ad,pdi,mdi,adprev=adx(d); candle=(close-d.low.iloc[-1])/(d.high.iloc[-1]-d.low.iloc[-1]) if d.high.iloc[-1]!=d.low.iloc[-1] else 0
    out={"daily_breakout":close>=resistance*1.005,"weekly_breakout":close>=wres*1.0025,"volume_ratio":vr,"volume_confirmed":vr>=1.5,"close_near_high":candle>=.98,
         "atr_distance":(close-resistance)/av if av else np.nan,"atr_expanding":av>=1.10*ap,"obv_breakout":obv.iloc[-1]>obv.iloc[-21:-1].max(),"ema_bullish":e20>e50,
         "ema200_bullish":bool(pd.notna(e200) and close>e200),"stoch_k":k,"stoch_d":sd,"adx":ad,"pdi":pdi,"mdi":mdi,"adx_prev":adprev,
         "breakout_failed":bool(close<resistance and any(d.close.iloc[-6:-1]>=resistance*1.005)),"breakout_accepted":bool((d.close.iloc[-3:]>resistance).all()),
         "breakout_distance_pct":(close-resistance)/resistance*100}
    out["retest_held"]=bool(out["breakout_accepted"] and d.low.iloc[-10:].min()<=resistance*1.01 and close>resistance)
    return out

def downside(ex,ticker):
    try:
        s=symbol(ex,ticker); b=symbol(ex,"BTC")
        if not s or not b:return np.nan,np.nan,np.nan
        a=ohlcv(ex,s,"1d",90)[["ts","close"]].rename(columns={"close":"coin"}); q=ohlcv(ex,b,"1d",90)[["ts","close"]].rename(columns={"close":"btc"})
        z=a.merge(q,on="ts").set_index("ts").pct_change().dropna(); z=z[z.btc<0]
        if len(z)<5:return np.nan,np.nan,np.nan
        beta=float(z.coin.cov(z.btc)/z.btc.var()) if z.btc.var() else np.nan
        return beta,float((z.coin-z.btc).mean()*100),float((z.coin>z.btc).mean()*100)
    except Exception:return np.nan,np.nan,np.nan

def technical_score(m,wrsi,rel,downhit,rsi_quality=100):
    s=0
    s+=12 if m.get("daily_breakout") else 0; s+=8 if m.get("weekly_breakout") else 0; s+=8 if m.get("volume_confirmed") else 0; s+=5 if m.get("close_near_high") else 0
    s+=8 if m.get("obv_breakout") else 0; s+=5 if m.get("atr_expanding") else 0; s+=6 if m.get("ema_bullish") else 0; s+=3 if m.get("ema200_bullish") else 0
    s+=8 if pd.notna(m.get("atr_distance")) and m["atr_distance"]>=.25 else 0; s+=7 if m.get("retest_held") else (4 if m.get("breakout_accepted") else 0)
    s+=5 if pd.notna(m.get("adx")) and m["adx"]>=25 else 0; s+=4 if pd.notna(m.get("pdi")) and pd.notna(m.get("mdi")) and m["pdi"]-m["mdi"]>=5 else 0
    s+=5 if pd.notna(m.get("stoch_k")) and pd.notna(m.get("stoch_d")) and m["stoch_k"]>=50 and m["stoch_k"]>m["stoch_d"] else 0
    s+=5 if pd.notna(wrsi) and wrsi<75 else 0; s+=5 if rel>=5 else 0; s+=4 if pd.notna(downhit) and downhit>=50 else 0
    raw=float(np.clip(s/96*100,0,100)); quality_factor=0.70+0.30*float(np.clip(rsi_quality,0,100))/100.0
    return float(np.clip(raw*quality_factor,0,100))

def true_breakout(m,vr,wrsi,rel):
    return bool(m.get("daily_breakout") and m.get("weekly_breakout") and m.get("volume_confirmed") and m.get("close_near_high") and pd.notna(m.get("atr_distance")) and m["atr_distance"]>=.25 and m.get("atr_expanding") and m.get("obv_breakout") and rel>=5 and vr>=10 and pd.notna(wrsi) and wrsi<75 and pd.notna(m.get("stoch_k")) and pd.notna(m.get("stoch_d")) and m["stoch_k"]>=50 and m["stoch_k"]>m["stoch_d"] and pd.notna(m.get("adx")) and m["adx"]>=25 and m["adx"]>m.get("adx_prev",-np.inf) and pd.notna(m.get("pdi")) and pd.notna(m.get("mdi")) and m["pdi"]-m["mdi"]>=5 and m.get("ema_bullish"))

def telegram(msg):
    token=os.getenv("TELEGRAM_BOT_TOKEN",""); chat=os.getenv("TELEGRAM_CHAT_ID","")
    if not token or not chat:return False,"not configured"
    try:
        j=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":chat,"text":msg[:4096]},timeout=20).json()
        return bool(j.get("ok")),j.get("description","")
    except Exception as e:return False,str(e)

def btc_direction(btc24,btc7,wrsi,m,dom):
    score=0
    score += 2 if btc24>2 else (-2 if btc24<-2 else 0)
    score += 2 if btc7>5 else (-2 if btc7<-5 else 0)
    score += 2 if pd.notna(wrsi) and wrsi>=60 else (-2 if pd.notna(wrsi) and wrsi<=40 else 0)
    score += 2 if m.get("ema_bullish") else (-2 if m and not m.get("ema_bullish") else 0)
    score += 1 if pd.notna(m.get("adx")) and m["adx"]>=25 and m.get("pdi",0)>m.get("mdi",0) else (-1 if pd.notna(m.get("adx")) and m["adx"]>=25 and m.get("mdi",0)>m.get("pdi",0) else 0)
    # Dominance is reported as context; it does not override BTC price trend.
    if dom.get("change_7d",np.nan) >= 0.50: score += 1
    elif dom.get("change_7d",np.nan) <= -0.50: score -= 1
    if score>=5:return "🟢 BTC BULLISH / RISK-ON"
    if score<=-5:return "🔴 BTC BEARISH / RISK-OFF"
    if score>=2:return "🟡 BTC LEANING BULLISH"
    if score<=-2:return "🟠 BTC LEANING BEARISH"
    return "⚪ BTC NEUTRAL / TRANSITION"

def main():
    df=markets(); df["mcap_m"]=df.market_cap/1e6; df["vol_m"]=df.total_volume/1e6; df["vr"]=df.total_volume/df.market_cap*100
    btc=df[df.id=="bitcoin"].iloc[0]; btc7=float(btc.price_change_percentage_7d or btc.price_change_percentage_7d_in_currency or 0); btc24=float(btc.price_change_percentage_24h or 0)
    dom=btc_dominance()
    t3=total3_btc()
    fg=fear_greed()
    stable={"tether","usd-coin","dai","usds","true-usd","usdd"}; c=df[df.mcap_m.between(20,500)&(df.vol_m>=2)&(df.vr>=5)&~df.id.isin(stable)].copy()
    zen=df[df.id==ZEN_ID]
    if not zen.empty:c=pd.concat([c,zen]).drop_duplicates("id")
    c["btc_rel_7d"]=c.price_change_percentage_7d_in_currency-btc7
    c["pre"]=np.clip(c.vr/25*20,0,20)+np.clip((c.price_change_percentage_24h+10)*.8,0,20)+np.clip((c.btc_rel_7d+10)*.4,0,20)+np.clip(c.vr/10,0,20)
    pre=c.sort_values("pre",ascending=False).head(60).copy()
    if not zen.empty:pre=pd.concat([pre,zen]).drop_duplicates("id")

    ex=None; primary_exchange="N/A"
    for name in ("okx","kraken","bybit"):
        candidate=exchange(name)
        if candidate is not None: ex=candidate; primary_exchange=name; break
    exchange_pool=[exchange(name) for name in ("okx","kraken","bybit")]; exchange_pool=[e for e in exchange_pool if e is not None]

    btc_vals,btc_rsi_sources=mtf_rsi(exchange_pool,"BTC"); btc_wrsi=weighted(btc_vals); btc_valid=[k for k,v in btc_vals.items() if pd.notna(v)]
    btc_mm={}
    for name in ("okx","kraken","bybit"):
        try:
            e=exchange(name); s=symbol(e,"BTC")
            if s:
                btc_mm=metrics(ohlcv(e,s,"1d",1000),ohlcv(e,s,"1w",100)); break
        except Exception: continue
    btc_signal=btc_direction(btc24,btc7,btc_wrsi,btc_mm,dom)
    btc_rsi_source="/".join(dict.fromkeys(btc_rsi_sources[k] for k in WEIGHTS if btc_rsi_sources[k]!="N/A")) or "N/A"

    rows=[]; rejected_zero_rsi=[]
    for _,x in pre.iterrows():
        t=str(x.symbol).upper(); vals,rsi_sources=mtf_rsi(exchange_pool,t); wr=weighted(vals); valid_tfs=[k for k,v in vals.items() if pd.notna(v)]; rsi_quality=len(valid_tfs)/len(WEIGHTS)*100; rsi_source="/".join(dict.fromkeys(rsi_sources[k] for k in WEIGHTS if rsi_sources[k]!="N/A")) or "N/A"; mm={}; source=primary_exchange.upper() if primary_exchange!="N/A" else "N/A"
        if len(valid_tfs)==0:
            rejected_zero_rsi.append({"ticker":t,"coin":x["name"],"reason":"No valid RSI timeframe data"}); continue
        for name in ("okx","kraken","bybit"):
            try:
                e=exchange(name); s=symbol(e,t)
                if s: mm=metrics(ohlcv(e,s,"1d",100),ohlcv(e,s,"1w",80)); source=name.upper(); break
            except Exception: continue
        beta,downrel,downhit=downside(ex,t); tech=technical_score(mm,wr,float(x.btc_rel_7d),downhit,rsi_quality); tb=true_breakout(mm,float(x.vr),wr,float(x.btc_rel_7d))
        if rsi_quality < 50:sig="⚠️ DATA QUALITY WARNING"
        elif mm.get("breakout_failed"):sig="🚨 BREAKOUT FAILED"
        elif tb and wr<80:sig="🚀 TRUE BREAKOUT"
        elif mm.get("retest_held"):sig="🚀 RETEST HELD"
        elif mm.get("breakout_accepted"):sig="🟢 BREAKOUT ACCEPTED"
        elif mm.get("daily_breakout"):sig="🟡 BREAKOUT ATTEMPT"
        elif tech>=70:sig="🟢 STRONG SETUP"
        else:sig="⚪ BASE / PRE-BREAKOUT"
        rows.append({"coin":x["name"],"ticker":t,"id":x["id"],"market_cap_m":float(x.mcap_m),"volume_m":float(x.vol_m),"vol_mcap_pct":float(x.vr),"change_24h_pct":float(x.price_change_percentage_24h),"change_7d_pct":float(x.price_change_percentage_7d_in_currency),"btc_rel_7d_pct":float(x.btc_rel_7d),
                     "rsi_1h":vals["1H"],"rsi_4h":vals["4H"],"rsi_1d":vals["1D"],"rsi_1w":vals["1W"],"rsi_1m":vals["1M"],"rsi_3m":vals["3M"],"weighted_rsi":wr,"rsi_valid_timeframes":len(valid_tfs),"rsi_quality_pct":rsi_quality,"rsi_source":rsi_source,"downside_beta":beta,"btc_down_day_rel_pct":downrel,"btc_down_day_outperform_pct":downhit,"technical_score":tech,"true_breakout":tb,"breakout_state":sig,"breakout_exchange":source,"breakout_pct":mm.get("breakout_distance_pct"),"breakout_volume_ratio":mm.get("volume_ratio"),"adx":mm.get("adx"),"plus_di":mm.get("pdi"),"minus_di":mm.get("mdi"),"stoch_k":mm.get("stoch_k"),"stoch_d":mm.get("stoch_d")})
    out=pd.DataFrame(rows).sort_values(["true_breakout","technical_score"],ascending=[False,False])
    breadth=float((c.price_change_percentage_24h>0).mean()*100)
    regime="RISK-ON" if btc24>2 and btc7>3 and breadth>=55 else ("RISK-OFF" if btc24<-3 and btc7<-5 and breadth<35 else "MIXED / TRANSITION")
    now=datetime.now(timezone.utc).isoformat(); clean=out.replace({np.nan:None})
    quality_counts={"pre_screen":len(pre),"ranked":len(out),"zero_rsi_rejected":len(rejected_zero_rsi),"rsi_6of6":int((out.rsi_valid_timeframes==6).sum()) if not out.empty else 0,"rsi_5plus":int((out.rsi_valid_timeframes>=5).sum()) if not out.empty else 0,"rsi_3plus":int((out.rsi_valid_timeframes>=3).sum()) if not out.empty else 0,"data_quality_warning":int((out.rsi_quality_pct<50).sum()) if not out.empty else 0}
    btc_payload={"signal":btc_signal,"price_change_24h":btc24,"price_change_7d":btc7,"weighted_rsi":btc_wrsi,"rsi_valid_timeframes":len(btc_valid),"rsi_quality_pct":len(btc_valid)/6*100,"rsi_source":btc_rsi_source,"adx":btc_mm.get("adx"),"plus_di":btc_mm.get("pdi"),"minus_di":btc_mm.get("mdi"),"stoch_k":btc_mm.get("stoch_k"),"stoch_d":btc_mm.get("stoch_d"),"ema_bullish":btc_mm.get("ema_bullish"),"daily_breakout":btc_mm.get("daily_breakout"),"weekly_breakout":btc_mm.get("weekly_breakout"),"technical_score":technical_score(btc_mm,btc_wrsi,0,np.nan,len(btc_valid)/6*100),"dominance_pct":dom["current"],"dominance_change_1d":dom["change_1d"],"dominance_change_7d":dom["change_7d"],"dominance_trend":dom["trend"],"total3_btc_ratio":t3["ratio"],"total3_btc_change_1d_pct":t3["change_1d_pct"],"total3_btc_change_7d_pct":t3["change_7d_pct"],"total3_btc_trend":t3["trend"],"fear_greed_value":fg["value"],"fear_greed_classification":fg["classification"],"fear_greed_change_1d":fg["change_1d"]}
    payload={"meta":{"engine_version":VERSION,"generated_at":now,"regime":regime,"btc_24h":btc24,"btc_7d":btc7,"breadth":breadth,"btc_dominance_pct":dom["current"],"btc_dominance_change_1d":dom["change_1d"],"btc_dominance_change_7d":dom["change_7d"],"btc_dominance_trend":dom["trend"],"candidate_count":len(c),"pre_screen_count":len(pre),"ranked_count":len(out),"exchange":primary_exchange,"data_quality":quality_counts},"btc_market":btc_payload,"top10":clean.head(10).to_dict("records"),"all":clean.to_dict("records")}
    os.makedirs("data",exist_ok=True); open("data/latest_scan.json","w",encoding="utf-8").write(json.dumps(payload,indent=2,default=str))

    statefile="data/telegram_alert_state.json"; state=json.load(open(statefile,encoding="utf-8")) if os.path.exists(statefile) else {"keys":[]}; keys=set(state.get("keys",[])); sent=0
    btc_key=f"{now[:10]}|{VERSION}|BTC|{btc_signal}|{dom['trend']}|{t3['trend']}"
    if btc_key not in keys:
        missing_btc=[tf for tf in WEIGHTS if pd.isna(btc_vals.get(tf))]
        fmt=lambda x: "N/A" if pd.isna(x) else f"{x:.2f}"
        btc_msg=(f"🌐 GENERAL CRYPTO MARKET DIRECTION\n\n{btc_signal}\n\nBTC 24h: {btc24:.2f}%\nBTC 7d: {btc7:.2f}%\n"
                 f"BTC Weighted RSI: {fmt(btc_wrsi)}\nBTC RSI coverage: {len(btc_valid)}/6 ({len(btc_valid)/6*100:.0f}%)\n"
                 f"BTC ADX: {fmt(btc_mm.get('adx',np.nan))}\n+DI / -DI: {fmt(btc_mm.get('pdi',np.nan))} / {fmt(btc_mm.get('mdi',np.nan))}\n"
                 f"StochRSI K/D: {fmt(btc_mm.get('stoch_k',np.nan))} / {fmt(btc_mm.get('stoch_d',np.nan))}\nEMA20>EMA50: {btc_mm.get('ema_bullish')}\n"
                 f"\n₿ BTC DOMINANCE: {fmt(dom['current'])}%\n{dom['trend']}\n"
                 f"Dominance 1D: {fmt(dom['change_1d'])} percentage points\nDominance 7D: {fmt(dom['change_7d'])} percentage points\n"
                 f"\n📊 TOTAL3/BTC: {fmt(t3['ratio'])}\n{t3['trend']}\nTOTAL3/BTC 1D: {fmt(t3['change_1d_pct'])}%\nTOTAL3/BTC 7D: {fmt(t3['change_7d_pct'])}%\n"
                 f"\n😨 FEAR & GREED: {fmt(fg['value'])} — {fg['classification']}\nFear & Greed 1D change: {fmt(fg['change_1d'])}\nSource: Alternative.me\n"
                 f"\nALTCOIN BREADTH: {breadth:.1f}%\nOVERALL REGIME: {regime}\nMissing RSI: {', '.join(missing_btc) if missing_btc else 'None'}")
        ok,err=telegram(btc_msg)
        if ok: keys.add(btc_key); sent+=1
        else: print("Telegram BTC:",err)

    for r in out.to_dict("records"):
        sig=r["breakout_state"]; score=float(r.get("technical_score") or 0); key=f"{now[:10]}|{VERSION}|{r['ticker']}|{sig}"
        if ("BREAKOUT" in sig or "RETEST" in sig) and (score>=70 or r["ticker"]=="ZEN") and key not in keys:
            missing=[tf for tf in WEIGHTS if pd.isna(r.get("rsi_"+tf.lower()))]
            msg=(f"{sig}\n\n{r['coin']} ({r['ticker']})\nTechnical score: {score:.1f}\nWeighted RSI: {r['weighted_rsi'] if r['weighted_rsi'] is not None else 'N/A'}\n"
                 f"BTC market direction: {btc_signal}\nBTC 24h/7d: {btc24:.2f}% / {btc7:.2f}%\nBTC dominance: {dom['current']:.2f}% ({dom['trend']})\n"
                 f"BTC dominance 1D/7D: {dom['change_1d']:.2f}pp / {dom['change_7d']:.2f}pp\nTOTAL3/BTC: {t3['ratio']:.6f} ({t3['trend']})\nTOTAL3/BTC 1D/7D: {t3['change_1d_pct']:.2f}% / {t3['change_7d_pct']:.2f}%\nBTC-relative 7D: {r['btc_rel_7d_pct']:.2f}%\nVol/MCap: {r['vol_mcap_pct']:.2f}%\nADX: {r['adx'] if r['adx'] is not None else 'N/A'}\n"
                 f"\nDATA_QUALITY\nRSI coverage: {r['rsi_valid_timeframes']}/6 ({r['rsi_quality_pct']:.0f}%)\nRSI source: {r['rsi_source']}\nMissing RSI: {', '.join(missing) if missing else 'None'}\n\nRegime: {regime}")
            ok,err=telegram(msg)
            if ok:keys.add(key); sent+=1
            else: print("Telegram:",err)
    state["keys"]=list(keys)[-1000:]; open(statefile,"w",encoding="utf-8").write(json.dumps(state,indent=2))
    print(f"{VERSION} scan complete | candidates={len(c)} pre_screen={len(pre)} primary_exchange={primary_exchange} regime={regime} btc={btc_signal} btc_dom={dom['current']}% {dom['trend']} total3btc={t3['ratio']} {t3['trend']} fng={fg['value']} {fg['classification']} telegram_sent={sent}")

if __name__=="__main__": main()
