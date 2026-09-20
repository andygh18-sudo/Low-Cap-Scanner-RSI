import json, os, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

CG='https://api.coingecko.com/api/v3'
ZEN_ID='horizen'
STABLES={'tether','usd-coin','dai','usds','true-usd','usdd'}

session=requests.Session()
session.headers.update({'User-Agent':'crypto-lowcap-scanner/2.0'})

def get_json(url, params=None, retries=4):
    last=None
    for i in range(retries):
        try:
            r=session.get(url,params=params,timeout=30)
            if r.status_code==429:
                time.sleep(3*(i+1)); continue
            r.raise_for_status(); return r.json()
        except Exception as e:
            last=e; time.sleep(2*(i+1))
    raise last

def markets():
    return pd.DataFrame(get_json(f'{CG}/coins/markets',{
        'vs_currency':'usd','order':'market_cap_desc','per_page':250,'page':1,
        'sparkline':'false','price_change_percentage':'24h,7d,30d'}))

def chart(cid, days=90):
    return get_json(f'{CG}/coins/{cid}/market_chart',{'vs_currency':'usd','days':days})

def price_series(data):
    p=data.get('prices',[])
    if not p: return pd.Series(dtype=float)
    idx=pd.to_datetime([x[0] for x in p],unit='ms',utc=True)
    return pd.Series([x[1] for x in p],index=idx,dtype=float).sort_index()

def rsi(s,n=14):
    s=pd.Series(s,dtype=float).dropna()
    if len(s)<n+1:return np.nan
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    al=l.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    rs=ag/al.replace(0,np.nan)
    v=100-(100/(1+rs))
    return float(v.iloc[-1]) if pd.notna(v.iloc[-1]) else np.nan

def multi_rsi(s):
    daily=rsi(s)
    weekly=rsi(s.resample('W').last().dropna()) if len(s)>=90 else np.nan
    monthly=rsi(s.resample('ME').last().dropna()) if len(s)>=210 else np.nan
    quarterly=rsi(s.resample('QE').last().dropna()) if len(s)>=365 else np.nan
    return {'1D':daily,'1W':weekly,'1M':monthly,'3M':quarterly}

def resilience(coin_s, btc_s):
    a=pd.DataFrame({'c':coin_s,'b':btc_s}).dropna()
    if len(a)<30:return {'corr':np.nan,'beta':np.nan,'down_beta':np.nan,'down_rel':np.nan,'res_score':0.0}
    r=a.pct_change().dropna(); br=r.b
    cov=np.cov(r.c,br,ddof=1)[0,1]; var=np.var(br,ddof=1)
    beta=cov/var if var>0 else np.nan
    down=r[br<0]
    if len(down)>=10:
        dv=np.var(down.b,ddof=1); db=np.cov(down.c,down.b,ddof=1)[0,1]/dv if dv>0 else np.nan
        down_rel=float((down.c-down.b).mean()*100)
    else: db=np.nan; down_rel=np.nan
    corr=float(r.c.corr(br))
    # 15-point resilience score: lower downside beta + better BTC-down-day relative return.
    beta_pts=7.5 if pd.isna(db) else float(np.clip((1.0-db)*7.5,0,7.5))
    rel_pts=7.5 if pd.isna(down_rel) else float(np.clip((down_rel+3.0)/6.0*7.5,0,7.5))
    return {'corr':corr,'beta':beta,'down_beta':db,'down_rel':down_rel,'res_score':beta_pts+rel_pts}

def scan():
    df=markets(); df['mcap_m']=df.market_cap/1e6; df['vol_m']=df.total_volume/1e6
    df['vr']=np.where(df.market_cap>0,df.total_volume/df.market_cap*100,0)
    btc=df[df.id.eq('bitcoin')].iloc[0]
    btc24=float(btc.price_change_percentage_24h); btc7=float(btc.price_change_percentage_7d_in_currency)
    c=df[df.mcap_m.between(20,500)&(df.vol_m>=2)&(df.vr>=5)&~df.id.isin(STABLES)].copy()
    zen=df[df.id.eq(ZEN_ID)]
    if not zen.empty:c=pd.concat([c,zen],ignore_index=True).drop_duplicates('id')
    c['btc_rel_7d']=c.price_change_percentage_7d_in_currency-btc7
    breadth=float((c.price_change_percentage_24h>0).mean()*100) if len(c) else 0
    if btc24>2 and btc7>3 and breadth>=55: regime='RISK-ON'; regime_penalty=0
    elif btc24<-3 and btc7<-5 and breadth<35: regime='RISK-OFF'; regime_penalty=15
    else: regime='MIXED / TRANSITION'; regime_penalty=7.5

    # Preselect the most liquid/active names to keep scheduled runs reliable.
    c['pre_score']=np.clip(c.vr/25*30,0,30)+np.clip((c.price_change_percentage_24h+10)*1.0,0,20)+np.clip((c.btc_rel_7d+10)*0.5,0,10)
    c=c.sort_values('pre_score',ascending=False).head(24).copy()
    if ZEN_ID not in set(c.id):
        z=df[df.id.eq(ZEN_ID)]
        if not z.empty:c=pd.concat([c,z],ignore_index=True).drop_duplicates('id')

    btc_s=price_series(chart('bitcoin',90)); rows=[]
    for _,x in c.iterrows():
        try:
            s=price_series(chart(x.id,90)); time.sleep(0.8)
            mr=multi_rsi(s); rr=resilience(s,btc_s)
            vals=[v for v in mr.values() if pd.notna(v)]
            weights={'1D':.20,'1W':.25,'1M':.25,'3M':.30}
            usable=[(mr[k],weights[k]) for k in weights if pd.notna(mr[k])]
            wrsi=float(np.average([v for v,w in usable],weights=[w for v,w in usable])) if usable else 50
            # 20 RSI points: constructive zone 45-70, with a mild penalty for extremes.
            rsi_pts=float(np.clip(20-abs(wrsi-57)*0.45,0,20))
            trend_pts=float(np.clip((float(x.price_change_percentage_24h)+5),0,10))
            rel_pts=float(np.clip((float(x.btc_rel_7d)+5)*0.5,0,10))
            vol_pts=float(np.clip(float(x.vr)/2,0,15))
            liq_pts=float(np.clip(float(x.vol_m)/10,0,5))
            base_pts=25.0
            raw=rsi_pts+trend_pts+rel_pts+vol_pts+liq_pts+rr['res_score']+base_pts
            score=float(np.clip(raw-regime_penalty,0,100))
            status='🟢 BREAKOUT CONFIRMATION' if score>=75 else ('🟢 MOMENTUM CONFIRMED' if score>=60 else ('🟡 WATCH / PULLBACK' if score>=45 else '🔴 WEAK'))
            if wrsi>=80 and score>=60: status='🟠 EXTENDED — WAIT FOR RESET'
            elif float(x.btc_rel_7d)>10 and float(x.vr)>=20 and wrsi<70: status='🚀 RELATIVE-STRENGTH BREAKOUT'
            rows.append({
                'coin':x['name'],'ticker':str(x['symbol']).upper(),'id':x.id,
                'market_cap_m':float(x.mcap_m),'volume_m':float(x.vol_m),'vol_mcap_pct':float(x.vr),
                'change_24h_pct':float(x.price_change_percentage_24h),'change_7d_pct':float(x.price_change_percentage_7d_in_currency),
                'btc_rel_7d_pct':float(x.btc_rel_7d),'rsi_1d':mr['1D'],'rsi_1w':mr['1W'],'rsi_1m':mr['1M'],'rsi_3m':mr['3M'],
                'weighted_rsi':wrsi,'downside_beta':rr['down_beta'],'btc_down_day_rel_pct':rr['down_rel'],
                'resilience_score':rr['res_score'],'score':score,'signal':status
            })
        except Exception as e:
            rows.append({'coin':x['name'],'ticker':str(x['symbol']).upper(),'id':x.id,'score':0,'signal':'⚪ DATA ERROR','error':str(e)})
    out=pd.DataFrame(rows).sort_values('score',ascending=False)
    return out, {'generated_at':datetime.now(timezone.utc).isoformat(),'regime':regime,'regime_penalty':regime_penalty,'btc_24h':btc24,'btc_7d':btc7,'breadth':breadth}

def main():
    out,meta=scan(); os.makedirs('data',exist_ok=True)
    payload={'meta':meta,'top10':out.head(10).replace({np.nan:None}).to_dict(orient='records'),'all':out.replace({np.nan:None}).to_dict(orient='records')}
    with open('data/latest_scan.json','w') as f: json.dump(payload,f,indent=2)
    out.to_csv('data/latest_scan.csv',index=False)
    print(json.dumps(meta,indent=2)); print(out.head(10).to_string(index=False))

if __name__=='__main__':main()
