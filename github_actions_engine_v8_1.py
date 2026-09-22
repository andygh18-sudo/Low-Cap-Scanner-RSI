import json, os, time, random
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import ccxt

VERSION = "v8.1"
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
    return pd.concat(frames,ignore_index=True).drop_duplicates("id")

def exchange(name):
    k=f"ex:{name}"
    if k in CACHE:
        return CACHE[k]
    try:
        ex=getattr(ccxt,name)({"enableRateLimit":True,"timeout":30000})
        ex.load_markets()
        CACHE[k]=ex
        return ex
    except Exception as e:
        print(f"Exchange unavailable: {name}: {type(e).__name__}: {e}")
        CACHE[k]=None
        return None

def symbol(ex,ticker):
    if ex is None:
        return None
    t=str(ticker).upper()
    for s in (f"{t}/USDT",f"{t}/USDT:USDT",f"{t}/USDC",f"{t}/USD"):
        m=ex.markets.get(s)
        if m and m.get("active",True) and (m.get("spot") or s.endswith(":USDT")): return s
    for s,m in ex.markets.items():
        if str(m.get("base","")).upper()==t and str(m.get("quote","")).upper()=="USDT" and m.get("active",True) and m.get("spot"): return s
    return None

def ohlcv(ex,sym,tf,limit):
    if ex is None or not sym:
        raise ValueError("exchange or symbol unavailable")
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

def mtf_rsi(ex,ticker):
    vals={k:np.nan for k in WEIGHTS}; sym=symbol(ex,ticker)
    if not sym:return vals
    for tf in ("1h","4h"):
        try: vals[tf.upper()]=rsi(ohlcv(ex,sym,tf,300).close)
        except Exception: pass
    try:
        d=ohlcv(ex,sym,"1d",1000); s=d.set_index("ts").close.astype(float).dropna().sort_index()
        vals["1D"]=rsi(s)
        for key,rule in (("1W","W-SUN"),("1M","ME"),("3M","QE")):
            q=s.resample(rule).last().dropna(); vals[key]=rsi(q) if len(q)>=15 else np.nan
    except Exception: pass
    return vals

def weighted(vals):
    u=[(v,WEIGHTS[k]) for k,v in vals.items() if pd.notna(v)]
    return float(np.average([v for v,w in u],weights=[w for v,w in u])) if u else np.nan

def atr(d,n=14):
    pc=d.close.shift(1); tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def adx(d,n=14):
    up=d.high.diff(); dn=-d.low.diff(); plus=up.where((up>dn)&(up>0),0); minus=dn.where((dn>up)&(dn>0),0)
    tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1)
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

def technical_score(m,wrsi,rel,downhit):
    s=0
    s+=12 if m.get("daily_breakout") else 0; s+=8 if m.get("weekly_breakout") else 0; s+=8 if m.get("volume_confirmed") else 0; s+=5 if m.get("close_near_high") else 0
    s+=8 if m.get("obv_breakout") else 0; s+=5 if m.get("atr_expanding") else 0; s+=6 if m.get("ema_bullish") else 0; s+=3 if m.get("ema200_bullish") else 0
    s+=8 if pd.notna(m.get("atr_distance")) and m["atr_distance"]>=.25 else 0; s+=7 if m.get("retest_held") else (4 if m.get("breakout_accepted") else 0)
    s+=5 if pd.notna(m.get("adx")) and m["adx"]>=25 else 0; s+=4 if pd.notna(m.get("pdi")) and pd.notna(m.get("mdi")) and m["pdi"]-m["mdi"]>=5 else 0
    s+=5 if pd.notna(m.get("stoch_k")) and pd.notna(m.get("stoch_d")) and m["stoch_k"]>=50 and m["stoch_k"]>m["stoch_d"] else 0
    s+=5 if pd.notna(wrsi) and wrsi<75 else 0; s+=5 if rel>=5 else 0; s+=4 if pd.notna(downhit) and downhit>=50 else 0
    return float(np.clip(s/96*100,0,100))

def true_breakout(m,vr,wrsi,rel):
    return bool(m.get("daily_breakout") and m.get("weekly_breakout") and m.get("volume_confirmed") and m.get("close_near_high") and pd.notna(m.get("atr_distance")) and m["atr_distance"]>=.25 and m.get("atr_expanding") and m.get("obv_breakout") and rel>=5 and vr>=10 and pd.notna(wrsi) and wrsi<75 and pd.notna(m.get("stoch_k")) and pd.notna(m.get("stoch_d")) and m["stoch_k"]>=50 and m["stoch_k"]>m["stoch_d"] and pd.notna(m.get("adx")) and m["adx"]>=25 and m["adx"]>m.get("adx_prev",-np.inf) and pd.notna(m.get("pdi")) and pd.notna(m.get("mdi")) and m["pdi"]-m["mdi"]>=5 and m.get("ema_bullish"))

def telegram(msg):
    token=os.getenv("TELEGRAM_BOT_TOKEN",""); chat=os.getenv("TELEGRAM_CHAT_ID","")
    if not token or not chat:return False,"not configured"
    try:
        j=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":chat,"text":msg[:4096]},timeout=20).json()
        return bool(j.get("ok")),j.get("description","")
    except Exception as e:return False,str(e)

def main():
    df=markets(); df["mcap_m"]=df.market_cap/1e6; df["vol_m"]=df.total_volume/1e6; df["vr"]=df.total_volume/df.market_cap*100
    btc=df[df.id=="bitcoin"].iloc[0]; btc7=float(btc.price_change_percentage_7d_in_currency or 0); btc24=float(btc.price_change_percentage_24h or 0)
    stable={"tether","usd-coin","dai","usds","true-usd","usdd"}; c=df[df.mcap_m.between(20,500)&(df.vol_m>=2)&(df.vr>=5)&~df.id.isin(stable)].copy()
    zen=df[df.id==ZEN_ID]
    if not zen.empty:c=pd.concat([c,zen]).drop_duplicates("id")
    c["btc_rel_7d"]=c.price_change_percentage_7d_in_currency-btc7
    c["pre"]=np.clip(c.vr/25*20,0,20)+np.clip((c.price_change_percentage_24h+10)*.8,0,20)+np.clip((c.btc_rel_7d+10)*.4,0,20)+np.clip(c.vr/10,0,20)
    pre=c.sort_values("pre",ascending=False).head(60).copy()
    if not zen.empty:pre=pd.concat([pre,zen]).drop_duplicates("id")
    # Bybit can return HTTP 403 from GitHub-hosted runner regions.
    # Use the first reachable exchange instead of making the whole scan fail.
    ex=None
    primary_exchange="N/A"
    for name in ("okx","kraken","bybit"):
        candidate=exchange(name)
        if candidate is not None:
            ex=candidate
            primary_exchange=name
            break
    rows=[]
    for _,x in pre.iterrows():
        t=str(x.symbol).upper(); vals=mtf_rsi(ex,t) if symbol(ex,t) else {k:np.nan for k in WEIGHTS}; wr=weighted(vals); mm={}; source=primary_exchange.upper() if primary_exchange!="N/A" else "N/A"
        for name in ("okx","kraken","bybit"):
            try:
                e=exchange(name); s=symbol(e,t)
                if s: mm=metrics(ohlcv(e,s,"1d",100),ohlcv(e,s,"1w",80)); source=name.upper(); break
            except Exception: continue
        beta,downrel,downhit=downside(ex,t); tech=technical_score(mm,wr,float(x.btc_rel_7d),downhit); tb=true_breakout(mm,float(x.vr),wr,float(x.btc_rel_7d))
        if mm.get("breakout_failed"):sig="🚨 BREAKOUT FAILED"
        elif tb and wr<80:sig="🚀 TRUE BREAKOUT"
        elif mm.get("retest_held"):sig="🚀 RETEST HELD"
        elif mm.get("breakout_accepted"):sig="🟢 BREAKOUT ACCEPTED"
        elif mm.get("daily_breakout"):sig="🟡 BREAKOUT ATTEMPT"
        elif tech>=70:sig="🟢 STRONG SETUP"
        else:sig="⚪ BASE / PRE-BREAKOUT"
        rows.append({"coin":x["name"],"ticker":t,"id":x["id"],"market_cap_m":float(x.mcap_m),"volume_m":float(x.vol_m),"vol_mcap_pct":float(x.vr),"change_24h_pct":float(x.price_change_percentage_24h),"change_7d_pct":float(x.price_change_percentage_7d_in_currency),"btc_rel_7d_pct":float(x.btc_rel_7d),
                     "rsi_1h":vals["1H"],"rsi_4h":vals["4H"],"rsi_1d":vals["1D"],"rsi_1w":vals["1W"],"rsi_1m":vals["1M"],"rsi_3m":vals["3M"],"weighted_rsi":wr,"downside_beta":beta,"btc_down_day_rel_pct":downrel,"btc_down_day_outperform_pct":downhit,"technical_score":tech,"true_breakout":tb,"breakout_state":sig,"breakout_exchange":source,"breakout_pct":mm.get("breakout_distance_pct"),"breakout_volume_ratio":mm.get("volume_ratio"),"adx":mm.get("adx"),"plus_di":mm.get("pdi"),"minus_di":mm.get("mdi"),"stoch_k":mm.get("stoch_k"),"stoch_d":mm.get("stoch_d")})
    out=pd.DataFrame(rows).sort_values(["true_breakout","technical_score"],ascending=[False,False])
    breadth=float((c.price_change_percentage_24h>0).mean()*100); regime="RISK-ON" if btc24>2 and btc7>3 and breadth>=55 else ("RISK-OFF" if btc24<-3 and btc7<-5 and breadth<35 else "MIXED / TRANSITION")
    now=datetime.now(timezone.utc).isoformat(); clean=out.replace({np.nan:None})
    payload={"meta":{"engine_version":VERSION,"generated_at":now,"regime":regime,"btc_24h":btc24,"btc_7d":btc7,"breadth":breadth,"candidate_count":len(c),"pre_screen_count":len(pre),"exchange":primary_exchange},"top10":clean.head(10).to_dict("records"),"all":clean.to_dict("records")}
    os.makedirs("data",exist_ok=True); open("data/latest_scan.json","w",encoding="utf-8").write(json.dumps(payload,indent=2,default=str))
    statefile="data/telegram_alert_state.json"; state=json.load(open(statefile,encoding="utf-8")) if os.path.exists(statefile) else {"keys":[]}; keys=set(state.get("keys",[])); sent=0
    for r in out.to_dict("records"):
        sig=r["breakout_state"]; score=float(r.get("technical_score") or 0); key=f"{now[:10]}|{r['ticker']}|{sig}"
        if ("BREAKOUT" in sig or "RETEST" in sig) and (score>=70 or r["ticker"]=="ZEN") and key not in keys:
            msg=f"{sig}\n\n{r['coin']} ({r['ticker']})\nTechnical score: {score:.1f}\nWeighted RSI: {r['weighted_rsi'] if r['weighted_rsi'] is not None else 'N/A'}\nBTC-relative 7D: {r['btc_rel_7d_pct']:.2f}%\nVol/MCap: {r['vol_mcap_pct']:.2f}%\nADX: {r['adx'] if r['adx'] is not None else 'N/A'}\nRegime: {regime} | BTC 24h: {btc24:.2f}% | BTC 7d: {btc7:.2f}%"
            ok,err=telegram(msg)
            if ok:keys.add(key); sent+=1
            else:print("Telegram:",err)
    state["keys"]=list(keys)[-1000:]; open(statefile,"w",encoding="utf-8").write(json.dumps(state,indent=2))
    print(f"{VERSION} scan complete | candidates={len(c)} pre_screen={len(pre)} primary_exchange={primary_exchange} regime={regime} telegram_sent={sent}")

if __name__=="__main__": main()
