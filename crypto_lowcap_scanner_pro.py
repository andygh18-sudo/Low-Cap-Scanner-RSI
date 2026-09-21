import streamlit as st

st.set_page_config(page_title="Crypto Low-Cap TRUE BREAKOUT PRO v6", page_icon="₿", layout="wide")

# Dashboard refresh controls
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "refresh_minutes" not in st.session_state:
    st.session_state.refresh_minutes = 15

with st.sidebar:
    st.subheader("🔄 Dashboard Refresh")
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.session_state.auto_refresh = st.checkbox(
        "Auto Refresh", value=st.session_state.auto_refresh,
        help="Refresh the dashboard while this browser session is active."
    )
    st.session_state.refresh_minutes = st.selectbox(
        "Refresh every", [5, 10, 15, 30, 60],
        index=[5, 10, 15, 30, 60].index(st.session_state.refresh_minutes),
        format_func=lambda x: f"{x} minutes"
    )

run_every = f"{st.session_state.refresh_minutes}m" if st.session_state.auto_refresh else None

@st.fragment(run_every=run_every, key="dashboard_refresh")
def _refresh_status():
    from datetime import datetime
    st.caption(f"Last dashboard refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

_refresh_status()
import pandas as pd
import numpy as np
import requests
import time
import random
import ccxt
from datetime import datetime, timezone
from rsi_engine import multi_timeframe_rsi, weighted_rsi


CG = "https://api.coingecko.com/api/v3"
ZEN_ID = "horizen"
DEFAULT_EXCHANGE = "bybit"
COINGECKO_API_KEY = __import__("os").getenv("COINGECKO_API_KEY", "")
CG_HEADERS = {"accept": "application/json"}
if COINGECKO_API_KEY:
    CG_HEADERS["x-cg-demo-api-key"] = COINGECKO_API_KEY

CG_RETRIES = 5
CG_BACKOFF_BASE = 2.0

def _cg_get(path, params=None, timeout=30):
    """CoinGecko GET with 429-aware exponential backoff and jitter."""
    url = f"{CG}/{path.lstrip('/')}"
    last_exc = None
    for attempt in range(CG_RETRIES):
        try:
            r = requests.get(url, params=params or {}, headers=CG_HEADERS, timeout=timeout)
            if r.status_code != 429:
                r.raise_for_status()
                return r
            retry_after = r.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else CG_BACKOFF_BASE * (2 ** attempt)
            except (TypeError, ValueError):
                wait = CG_BACKOFF_BASE * (2 ** attempt)
            wait = min(wait, 60.0) + random.uniform(0.25, 1.25)
            time.sleep(wait)
            last_exc = requests.HTTPError(f"429 Too Many Requests for {url}; retrying after {wait:.1f}s")
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < CG_RETRIES - 1:
                time.sleep(min(CG_BACKOFF_BASE * (2 ** attempt), 30.0) + random.uniform(0.25, 1.25))
    raise last_exc or requests.HTTPError(f"CoinGecko request failed: {url}")

@st.cache_data(ttl=600, show_spinner=False)
def cg_markets():
    """CoinGecko Top 500 universe with 429 protection.

    If page 2 is temporarily rate-limited, page 1 is retained so the
    scanner can still run; ZEN is explicitly re-added downstream.
    """
    frames=[]
    warning=None
    for page in (1, 2):
        p={"vs_currency":"usd","order":"market_cap_desc","per_page":250,"page":page,
           "sparkline":"false","price_change_percentage":"7d,30d"}
        try:
            data=_cg_get("coins/markets", params=p, timeout=30).json()
            if data:
                frames.append(pd.DataFrame(data))
        except Exception as exc:
            warning=f"CoinGecko page {page} unavailable after retries: {exc}"
            if page == 1 and not frames:
                raise
            break
    if not frames:
        return pd.DataFrame()
    out=pd.concat(frames,ignore_index=True).drop_duplicates("id")
    if warning:
        out.attrs["coingecko_warning"] = warning
    return out

@st.cache_data(ttl=600, show_spinner=False)
def cg_chart(cid, days=365):
    return _cg_get(f"coins/{cid}/market_chart",
                   params={"vs_currency":"usd","days":days},timeout=30).json()


def cg_daily_ohlc_from_market_chart(coin_id, days=90):
    """Build completed UTC daily OHLC from CoinGecko market-chart prices.

    CoinGecko's market_chart endpoint provides price points rather than
    candle OHLC, so each completed UTC day is reconstructed from the
    intraday prices. This is used only as an ADX/DMI fallback; exchange
    candles remain preferred for structure/breakout calculations.
    """
    payload=cg_chart(coin_id, days)
    prices=payload.get("prices",[])
    if not prices:
        return pd.DataFrame()
    q=pd.DataFrame(prices,columns=["ts","price"])
    q["ts"]=pd.to_datetime(q["ts"],unit="ms",utc=True)
    q["price"]=pd.to_numeric(q["price"],errors="coerce")
    q=q.dropna().set_index("ts").sort_index()
    d=pd.DataFrame({
        "open":q["price"].resample("1D").first(),
        "high":q["price"].resample("1D").max(),
        "low":q["price"].resample("1D").min(),
        "close":q["price"].resample("1D").last(),
    }).dropna().reset_index(drop=True)
    if len(d)>1:
        d=d.iloc[:-1].copy()
    return d


def atr_series(high, low, close, n=14):
    h=pd.Series(high,dtype=float); l=pd.Series(low,dtype=float); c=pd.Series(close,dtype=float)
    prev=c.shift(1)
    tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()


def obv_series(close, volume):
    c=pd.Series(close,dtype=float).reset_index(drop=True); v=pd.Series(volume,dtype=float).reset_index(drop=True)
    delta=c.diff()
    direction=np.sign(delta).fillna(0)
    return (direction*v.fillna(0)).cumsum()


def ema_series(values, n):
    return pd.Series(values,dtype=float).ewm(span=n,adjust=False,min_periods=n).mean()


def macd_metrics(close, fast=12, slow=26, signal=9):
    c=pd.Series(close,dtype=float)
    ef=c.ewm(span=fast,adjust=False,min_periods=fast).mean()
    es=c.ewm(span=slow,adjust=False,min_periods=slow).mean()
    macd=ef-es
    sig=macd.ewm(span=signal,adjust=False,min_periods=signal).mean()
    hist=macd-sig
    accel=bool(len(hist.dropna())>=3 and hist.iloc[-1]>hist.iloc[-2]>hist.iloc[-3])
    return (float(macd.iloc[-1]) if pd.notna(macd.iloc[-1]) else np.nan,
            float(sig.iloc[-1]) if pd.notna(sig.iloc[-1]) else np.nan,
            float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else np.nan, accel)


def bollinger_metrics(close, n=20, mult=2.0):
    c=pd.Series(close,dtype=float)
    mid=c.rolling(n,min_periods=n).mean(); std=c.rolling(n,min_periods=n).std(ddof=0)
    upper=mid+mult*std; lower=mid-mult*std
    width=(upper-lower)/mid.replace(0,np.nan)
    if len(width.dropna())<6:
        return np.nan,np.nan,False
    current=float(width.iloc[-1]); avg20=float(width.iloc[-21:-1].mean()) if len(width)>=21 else float(width.iloc[:-1].mean())
    expanding=pd.notna(current) and pd.notna(avg20) and current>=avg20*1.10
    return current,avg20,bool(expanding)


def cmf_series(high, low, close, volume, n=20):
    h=pd.Series(high,dtype=float); l=pd.Series(low,dtype=float); c=pd.Series(close,dtype=float); v=pd.Series(volume,dtype=float)
    rng=(h-l).replace(0,np.nan)
    mfm=((c-l)-(h-c))/rng
    mfv=mfm*v
    return mfv.rolling(n,min_periods=n).sum()/v.rolling(n,min_periods=n).sum().replace(0,np.nan)


def mfi_series(high, low, close, volume, n=14):
    h=pd.Series(high,dtype=float); l=pd.Series(low,dtype=float); c=pd.Series(close,dtype=float); v=pd.Series(volume,dtype=float)
    tp=(h+l+c)/3.0; flow=tp*v; delta=tp.diff()
    pos=flow.where(delta>0,0.0).rolling(n,min_periods=n).sum()
    neg=flow.where(delta<0,0.0).abs().rolling(n,min_periods=n).sum()
    ratio=pos/neg.replace(0,np.nan)
    out=100-(100/(1+ratio))
    out=out.where(~((neg==0)&(pos>0)),100.0)
    out=out.where(~((neg==0)&(pos==0)),50.0)
    return out


def acceptance_metrics(d, resistance, max_wait=10, retest_tol=0.01):
    """Return breakout acceptance, retest-held and distance %.

    Acceptance requires three completed closes above the prior resistance.
    Retest-held requires a recent pullback near resistance followed by a
    close back above it. This intentionally avoids intrabar-only confirmation.
    """
    if d is None or len(d)<3 or not pd.notna(resistance) or resistance<=0:
        return False,False,np.nan
    closes=pd.to_numeric(d["close"],errors="coerce").reset_index(drop=True)
    highs=pd.to_numeric(d["high"],errors="coerce").reset_index(drop=True)
    lows=pd.to_numeric(d["low"],errors="coerce").reset_index(drop=True)
    distance=(float(closes.iloc[-1])-float(resistance))/float(resistance)*100
    accepted=bool((closes.iloc[-3:]>resistance).all())
    retest=False
    if accepted:
        start=max(0,len(d)-max_wait)
        recent_low=lows.iloc[start:].min()
        near=recent_low<=resistance*(1+retest_tol)
        post=closes.iloc[start:]
        retest=bool(near and (post>resistance).sum()>=2 and closes.iloc[-1]>resistance)
    return accepted,retest,float(distance)


def _find_contract_symbol(ex, ticker):
    t=str(ticker).upper()
    preferred=[f"{t}/USDT:USDT",f"{t}/USDT"]
    for sym in preferred:
        m=ex.markets.get(sym)
        if m and m.get("active",True) and (m.get("swap") or m.get("contract")):
            return sym
    for sym,m in ex.markets.items():
        if (str(m.get("base","")).upper()==t and str(m.get("quote","")).upper()=="USDT"
            and m.get("active",True) and (m.get("swap") or m.get("contract"))):
            return sym
    return None


def derivatives_context(ex, ticker):
    """Optional current derivatives context. Missing derivatives never block a signal."""
    out={"oi":np.nan,"funding":np.nan,"oi_available":False,"funding_available":False}
    try:
        symbol=_find_contract_symbol(ex,ticker)
        if not symbol:
            return out
        if getattr(ex,"has",{}).get("fetchOpenInterest"):
            oi=ex.fetch_open_interest(symbol)
            val=oi.get("openInterestValue",oi.get("openInterestAmount",np.nan))
            if pd.notna(val): out["oi"]=float(val); out["oi_available"]=True
        if getattr(ex,"has",{}).get("fetchFundingRate"):
            fr=ex.fetch_funding_rate(symbol)
            val=fr.get("fundingRate",np.nan)
            if pd.notna(val): out["funding"]=float(val); out["funding_available"]=True
    except Exception:
        pass
    return out

def rsi_series(x, n=14):
    """Return full Wilder-style RSI series."""
    s = pd.Series(x, dtype=float)
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = l.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(~((al == 0) & (ag > 0)), 100.0)
    out = out.where(~((al == 0) & (ag == 0)), 50.0)
    return out

def stoch_rsi(values, rsi_len=14, stoch_len=14, k_len=3, d_len=3):
    """Return StochRSI %K, %D on a 0-100 scale."""
    rs = rsi_series(values, rsi_len)
    low = rs.rolling(stoch_len, min_periods=stoch_len).min()
    high = rs.rolling(stoch_len, min_periods=stoch_len).max()
    denom = high - low
    # A flat RSI window makes the textbook StochRSI denominator zero.
    # Treat that case as neutral (50) instead of propagating NaN into %K/%D.
    stoch = ((rs - low) / denom.replace(0, np.nan)) * 100.0
    stoch = stoch.where(denom != 0, 50.0)
    k = stoch.rolling(k_len, min_periods=k_len).mean()
    d = k.rolling(d_len, min_periods=d_len).mean()
    if k.dropna().empty or d.dropna().empty:
        return np.nan, np.nan
    return float(k.iloc[-1]), float(d.iloc[-1])

def _daily_closes_from_prices(prices):
    """Convert CoinGecko timestamp/price pairs to completed UTC daily closes."""
    if not prices:
        return pd.Series(dtype=float)
    q = pd.DataFrame(prices, columns=["ts", "price"])
    q["ts"] = pd.to_datetime(q["ts"], unit="ms", utc=True)
    q["price"] = pd.to_numeric(q["price"], errors="coerce")
    q = q.dropna(subset=["ts", "price"]).set_index("ts").sort_index()
    daily = q["price"].resample("1D").last().dropna()
    # Never use the currently forming UTC day.
    if len(daily) > 1:
        daily = daily.iloc[:-1]
    return daily


def _exchange_daily_closes(ticker, exchange_names=("bybit", "okx", "kraken")):
    """Fallback daily closes when CoinGecko data is unavailable/rate-limited."""
    for name in exchange_names:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            symbol = find_market_symbol(ex, ticker)
            if not symbol:
                continue
            candles = ex.fetch_ohlcv(symbol, timeframe="1d", limit=120)
            if not candles or len(candles) < 40:
                continue
            q = pd.DataFrame(candles, columns=["ts","open","high","low","close","volume"])
            q["ts"] = pd.to_datetime(q["ts"], unit="ms", utc=True)
            q = q.sort_values("ts")
            # Exchange OHLCV normally includes the current candle; exclude it.
            if len(q) > 1:
                q = q.iloc[:-1]
            return q["close"].astype(float).dropna(), name
        except Exception:
            continue
    return pd.Series(dtype=float), None


@st.cache_data(ttl=300)
def daily_stochrsi_with_source(coin_id, ticker=None):
    """Daily StochRSI returning %K, %D and the actual data source used."""
    for days in (90, 365):
        try:
            payload = cg_chart(coin_id, days)
            closes = _daily_closes_from_prices(payload.get("prices", []))
            if len(closes) >= 40:
                k, dval = stoch_rsi(closes)
                if pd.notna(k) and pd.notna(dval):
                    return k, dval, "CoinGecko"
        except Exception:
            pass
    if ticker:
        closes, source = _exchange_daily_closes(ticker)
        if len(closes) >= 40:
            k, dval = stoch_rsi(closes)
            if pd.notna(k) and pd.notna(dval):
                return k, dval, str(source).upper()
    return np.nan, np.nan, "Unavailable"


def daily_stochrsi(coin_id, ticker=None):
    """Compatibility wrapper returning only %K and %D."""
    k, dval, _source = daily_stochrsi_with_source(coin_id, ticker)
    return k, dval


def find_market_symbol(ex, ticker):
    """Return the first usable USDT spot symbol for a ticker."""
    t = str(ticker).upper()
    candidates = [f"{t}/USDT", f"{t}/USDT:USDT"]
    for sym in candidates:
        if sym in ex.markets:
            market = ex.markets[sym]
            if market.get("active", True) and market.get("spot", True):
                return sym
    # Fallback: match by base/quote in case the exchange uses a different unified symbol.
    for sym, market in ex.markets.items():
        if str(market.get("base", "")).upper() == t and str(market.get("quote", "")).upper() == "USDT" and market.get("spot", True):
            return sym
    return None


def adx_dmi(high, low, close, n=14):
    """Robust Wilder ADX/DMI with explicit initialization.

    Requires 2*n+1 completed candles for the first ADX and returns the
    latest ADX, +DI, -DI and prior ADX. Extra history is preferred because
    ADX is recursive and benefits from a warm-up period.
    """
    h=pd.to_numeric(pd.Series(high),errors="coerce").reset_index(drop=True)
    l=pd.to_numeric(pd.Series(low),errors="coerce").reset_index(drop=True)
    c=pd.to_numeric(pd.Series(close),errors="coerce").reset_index(drop=True)
    z=pd.concat([h,l,c],axis=1).dropna().reset_index(drop=True)
    if len(z)<(2*n+2):
        return np.nan,np.nan,np.nan,np.nan
    h,l,c=z.iloc[:,0],z.iloc[:,1],z.iloc[:,2]
    up=h.diff(); down=-l.diff()
    plus_dm=pd.Series(np.where((up>down)&(up>0),up,0.0),index=z.index)
    minus_dm=pd.Series(np.where((down>up)&(down>0),down,0.0),index=z.index)
    tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)

    # Wilder RMA initialized from the first n observations rather than
    # pandas EMA warm-up. This prevents false NaN values in sparse histories.
    def wilder(x, period):
        x=pd.Series(x,dtype=float).reset_index(drop=True)
        out=pd.Series(np.nan,index=x.index,dtype=float)
        if len(x)<period:return out
        out.iloc[period-1]=x.iloc[:period].sum()
        for i in range(period,len(x)):
            out.iloc[i]=out.iloc[i-1]-(out.iloc[i-1]/period)+x.iloc[i]
        return out

    tr_r=wilder(tr,n)
    pdm_r=wilder(plus_dm,n)
    mdm_r=wilder(minus_dm,n)
    plus_di=100*pdm_r/tr_r.replace(0,np.nan)
    minus_di=100*mdm_r/tr_r.replace(0,np.nan)
    di_sum=plus_di+minus_di
    dx=100*(plus_di-minus_di).abs()/di_sum.replace(0,np.nan)

    # First ADX is the arithmetic mean of the first n valid DX values,
    # followed by Wilder recursive smoothing.
    adx=pd.Series(np.nan,index=dx.index,dtype=float)
    valid_dx=dx.dropna()
    if len(valid_dx)<n+1:
        return np.nan,np.nan,np.nan,np.nan
    first_pos=valid_dx.index[n-1]
    adx.iloc[first_pos]=valid_dx.iloc[:n].mean()
    for i in range(first_pos+1,len(dx)):
        if pd.notna(dx.iloc[i]) and pd.notna(adx.iloc[i-1]):
            adx.iloc[i]=(adx.iloc[i-1]*(n-1)+dx.iloc[i])/n

    valid=adx.dropna()
    if len(valid)<2:return np.nan,np.nan,np.nan,np.nan
    last=valid.index[-1]; prev=valid.index[-2]
    vals=(adx.loc[last],plus_di.loc[last],minus_di.loc[last],adx.loc[prev])
    if not all(pd.notna(v) for v in vals):return np.nan,np.nan,np.nan,np.nan
    return tuple(float(v) for v in vals)

@st.cache_data(ttl=300)
def _valid_daily_ohlc_for_adx(d):
    """Validate completed daily OHLC data before ADX/DMI calculation."""
    if d is None or d.empty or not {"high","low","close"}.issubset(d.columns): return False
    z=d[["high","low","close"]].apply(pd.to_numeric,errors="coerce").dropna()
    if len(z)<35: return False
    return bool((z["high"]>=z["low"]).all() and (z>0).all().all())

def _exchange_daily_ohlc(exchange_name,ticker,limit=250):
    """Fetch completed daily OHLCV from an exchange as an ADX fallback."""
    try:
        ex=getattr(ccxt,exchange_name)({"enableRateLimit":True}); ex.load_markets()
        symbol=find_market_symbol(ex,ticker)
        if not symbol:return pd.DataFrame()
        rows=ex.fetch_ohlcv(symbol,timeframe="1d",limit=limit)
        if not rows:return pd.DataFrame()
        d=pd.DataFrame(rows,columns=["ts","open","high","low","close","volume"])
        d=d.iloc[:-1].copy() if len(d)>1 else d
        return d.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def daily_adx_with_source(coin_id, ticker=None):
    """Robust Daily ADX-14/DMI returning values plus actual source used."""
    candidates=[]
    try:
        d=cg_daily_ohlc_from_market_chart(coin_id,90)
        if _valid_daily_ohlc_for_adx(d): candidates.append(("CoinGecko",d))
    except Exception:
        pass

    t=str(ticker or ("BTC" if str(coin_id).lower()=="bitcoin" else "")).upper().strip()
    if t:
        for ex_name in ("bybit","okx","kraken"):
            d=_exchange_daily_ohlc(ex_name,t,250)
            if _valid_daily_ohlc_for_adx(d): candidates.append((ex_name.upper(),d))

    for source,d in candidates:
        try:
            vals=adx_dmi(d["high"],d["low"],d["close"],14)
            if all(pd.notna(v) for v in vals):
                return (*vals, source)
        except Exception:
            continue
    return np.nan,np.nan,np.nan,np.nan,"Unavailable"


def daily_adx(coin_id,ticker=None):
    """Compatibility wrapper returning ADX/DMI values without source."""
    av,pdi,mdi,ap,_source=daily_adx_with_source(coin_id,ticker)
    return av,pdi,mdi,ap


def true_breakout_metrics(exchange_name,ticker):
    """TRUE BREAKOUT PRO metrics: structure, volatility, volume flow, trend, momentum and acceptance."""
    out={"daily_resistance":np.nan,"weekly_resistance":np.nan,"daily_breakout":False,"weekly_breakout":False,"volume_confirmed":False,"close_near_high":False,"atr14":np.nan,"atr20_avg":np.nan,"atr_expanding":False,"atr_breakout_distance":np.nan,"atr_distance_confirmed":False,"obv_breakout":False,"ema20":np.nan,"ema50":np.nan,"ema200":np.nan,"ema_bullish":False,"ema200_bullish":False,"bb_width":np.nan,"bb_expanding":False,"cmf20":np.nan,"cmf_bullish":False,"mfi14":np.nan,"mfi_bullish":False,"macd_hist":np.nan,"macd_accelerating":False,"breakout_accepted":False,"breakout_retest_held":False,"breakout_distance_pct":np.nan,"rvol5":np.nan,"rvol_accelerating":False,"oi":np.nan,"funding":np.nan,"oi_available":False,"funding_available":False,"true_breakout_data":False,"breakout_exchange":None,"breakout_failed":False}
    try:
        ex=getattr(ccxt,exchange_name)({"enableRateLimit":True}); ex.load_markets(); symbol=find_market_symbol(ex,ticker)
        if not symbol:return out
        rows=ex.fetch_ohlcv(symbol,timeframe='1d',limit=260)
        if len(rows)<60:return out
        d=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume']); d=d.iloc[:-1].copy()
        cur=d.iloc[-1]; prior=d.iloc[-21:-1]; res=float(prior.high.max()); avgv=float(prior.volume.mean()); vol5=float(d.volume.iloc[-6:-1].mean()) if len(d)>=6 else np.nan; rvol5=float(cur.volume/vol5) if pd.notna(vol5) and vol5>0 else np.nan; rng=float(cur.high-cur.low); loc=((float(cur.close)-float(cur.low))/rng) if rng>0 else 0
        atr=atr_series(d.high,d.low,d.close,14); ac=float(atr.iloc[-1]); a20=atr.iloc[-21:-1].dropna(); aavg=float(a20.mean()) if len(a20) else np.nan; adist=(float(cur.close)-res)/ac if ac>0 else np.nan
        obv=obv_series(d.close,d.volume); op=obv.iloc[-21:-1].dropna()
        e20=float(ema_series(d.close,20).iloc[-1]); e50=float(ema_series(d.close,50).iloc[-1]); e200s=ema_series(d.close,200); e200=float(e200s.iloc[-1]) if pd.notna(e200s.iloc[-1]) else np.nan
        bbw,bba,bbe=bollinger_metrics(d.close); cmf=cmf_series(d.high,d.low,d.close,d.volume); mfi=mfi_series(d.high,d.low,d.close,d.volume); mac,ms,mh,ma=macd_metrics(d.close); acc,ret,dist=acceptance_metrics(d,res); der=derivatives_context(ex,ticker)
        wbreak=False; wres=np.nan
        try:
            wr=ex.fetch_ohlcv(symbol,timeframe='1w',limit=30); w=pd.DataFrame(wr,columns=['ts','open','high','low','close','volume']); w=w.iloc[:-1]
            if len(w)>=21:wres=float(w.iloc[-21:-1].high.max()); wbreak=float(w.iloc[-1].close)>=wres*1.0025
        except Exception:pass
        out.update({'daily_resistance':res,'weekly_resistance':wres,'daily_breakout':float(cur.close)>=res*1.005,'weekly_breakout':bool(wbreak),'volume_confirmed':avgv>0 and float(cur.volume)>=avgv*1.5,'close_near_high':loc>=.75,'atr14':ac,'atr20_avg':aavg,'atr_expanding':pd.notna(aavg) and ac>=aavg*1.10,'atr_breakout_distance':adist,'atr_distance_confirmed':pd.notna(adist) and adist>=.25,'obv_breakout':pd.notna(obv.iloc[-1]) and len(op)>0 and obv.iloc[-1]>op.max(),'ema20':e20,'ema50':e50,'ema200':e200,'ema_bullish':float(cur.close)>e20>e50,'ema200_bullish':pd.notna(e200) and float(cur.close)>e200,'bb_width':bbw,'bb_expanding':bbe,'cmf20':float(cmf.iloc[-1]) if pd.notna(cmf.iloc[-1]) else np.nan,'cmf_bullish':pd.notna(cmf.iloc[-1]) and cmf.iloc[-1]>0,'mfi14':float(mfi.iloc[-1]) if pd.notna(mfi.iloc[-1]) else np.nan,'mfi_bullish':pd.notna(mfi.iloc[-1]) and 50<=mfi.iloc[-1]<85,'macd_hist':mh,'macd_accelerating':bool(ma and pd.notna(mac) and pd.notna(ms) and mac>ms and mh>0),'breakout_accepted':acc,'breakout_retest_held':ret,'breakout_distance_pct':dist,'rvol5':rvol5,'rvol_accelerating':pd.notna(rvol5) and rvol5>=1.5 and float(cur.volume)>=avgv*1.5,'oi':der['oi'],'funding':der['funding'],'oi_available':der['oi_available'],'funding_available':der['funding_available'],'true_breakout_data':True,'breakout_exchange':exchange_name})
        # Explicit post-breakout failure state. A close back below the
        # prior resistance after acceptance is treated as a failed breakout.
        recent_closes=d.close.iloc[-6:]
        prior_breakout=bool((recent_closes.iloc[:-1] >= res*1.005).any()) if len(recent_closes)>=2 else False
        out["breakout_failed"]=bool(prior_breakout and float(recent_closes.iloc[-1]) < res)
        return out
    except Exception:return out

def breakout_confidence(m,btc_rel,vr,rs,sk,sd,av,pdi,mdi,ap):
    # 100-point confirmation score. Core structure/participation/trend/momentum
    # dominate; derivatives are context rather than a gate.
    score=0.0
    for pts,key in [(12,'daily_breakout'),(8,'weekly_breakout'),(5,'close_near_high'),
                    (8,'volume_confirmed'),(8,'obv_breakout'),(8,'atr_distance_confirmed'),
                    (5,'atr_expanding'),(6,'ema_bullish'),(3,'ema200_bullish'),
                    (5,'cmf_bullish'),(4,'mfi_bullish'),(5,'macd_accelerating'),
                    (5,'bb_expanding'),(4,'breakout_accepted'),(7,'breakout_retest_held'),
                    (4,'rvol_accelerating')]:
        score += pts if m.get(key) else 0
    if pd.notna(vr) and vr>=10: score+=3
    if pd.notna(av) and av>=25 and pd.notna(pdi) and pd.notna(mdi) and pdi-mdi>=5: score+=4
    if pd.notna(ap) and pd.notna(av) and av>ap: score+=3
    if pd.notna(sk) and pd.notna(sd) and sk>sd: score+=2
    if pd.notna(sk) and sk>=50: score+=1
    if pd.notna(rs) and 45<=rs<75: score+=2
    if pd.notna(btc_rel) and btc_rel>=5: score+=3
    if pd.notna(btc_rel) and btc_rel>=10: score+=2
    if pd.notna(m.get("btc_down_day_rel_pct")) and m.get("btc_down_day_rel_pct")>0: score+=2
    return float(min(100,score))

def is_true_breakout(metrics,btc_rel_7d,volume_mcap,weighted_rsi,stoch_k,stoch_d,adx_value,plus_di,minus_di,adx_prev):
    core=all([bool(metrics.get('daily_breakout')),bool(metrics.get('weekly_breakout')),bool(metrics.get('volume_confirmed')),bool(metrics.get('close_near_high')),bool(metrics.get('atr_distance_confirmed')),bool(metrics.get('atr_expanding')),bool(metrics.get('obv_breakout')),pd.notna(btc_rel_7d) and btc_rel_7d>=5,pd.notna(volume_mcap) and volume_mcap>=10,pd.notna(weighted_rsi) and weighted_rsi<75,pd.notna(stoch_k) and pd.notna(stoch_d) and stoch_k>stoch_d,pd.notna(stoch_k) and stoch_k>=50,pd.notna(adx_value) and adx_value>=25,pd.notna(adx_prev) and adx_value>adx_prev,pd.notna(plus_di) and pd.notna(minus_di) and (plus_di-minus_di)>=5])
    if pd.notna(metrics.get('ema20')) and pd.notna(metrics.get('ema50')): core=core and bool(metrics.get('ema_bullish'))
    return bool(core)

def signal(row):
    if row["score"]>=75:return "🟢 BREAKOUT CONFIRMATION"
    if row["score"]>=60:return "🟢 MOMENTUM CONFIRMED"
    if row["score"]>=45:return "🟡 WATCH / PULLBACK"
    return "🔴 WEAK"


@st.cache_data(ttl=300)
def downside_resilience(exchange_name,ticker):
    """Compare a coin with BTC specifically on BTC-negative daily sessions."""
    try:
        ex=getattr(ccxt,exchange_name)({"enableRateLimit":True}); ex.load_markets()
        sym=find_market_symbol(ex,ticker); btc_sym=find_market_symbol(ex,"BTC")
        if not sym or not btc_sym:return np.nan,np.nan,np.nan
        a=ex.fetch_ohlcv(sym,timeframe="1d",limit=60); b=ex.fetch_ohlcv(btc_sym,timeframe="1d",limit=60)
        if len(a)<35 or len(b)<35:return np.nan,np.nan,np.nan
        ad=pd.DataFrame(a,columns=["ts","open","high","low","close","volume"]); bd=pd.DataFrame(b,columns=["ts","open","high","low","close","volume"])
        ad=ad.iloc[:-1].copy(); bd=bd.iloc[:-1].copy()
        ar=ad.close.pct_change(); br=bd.close.pct_change()
        q=pd.DataFrame({"coin":ar.values,"btc":br.values}).dropna(); q=q[q.btc<0]
        if len(q)<5:return np.nan,np.nan,np.nan
        beta=float(q["coin"].cov(q["btc"])/q["btc"].var()) if q["btc"].var()>0 else np.nan
        rel=float((q["coin"]-q["btc"]).mean()*100)
        hit=float((q["coin"]>q["btc"]).mean()*100)
        return beta,rel,hit
    except Exception:
        return np.nan,np.nan,np.nan


@st.cache_data(ttl=300)
def btc_market_context(exchange_name):
    """Build BTC benchmark context using the same breakout/momentum tools as the scanner."""
    try:
        btc_row = cg_markets()
        btc_row = btc_row[btc_row["id"] == "bitcoin"]
        if btc_row.empty:
            return None
        b = btc_row.iloc[0]
        vals = multi_timeframe_rsi("BTC")
        usable = [(v, {"1H": .05, "4H": .10, "1D": .20, "1W": .25, "1M": .20, "3M": .20}[k])
                  for k, v in vals.items() if pd.notna(v)]
        wrsi = float(np.average([v for v, w in usable], weights=[w for v, w in usable])) if usable else np.nan
        sk, sd, stoch_source = daily_stochrsi_with_source("bitcoin", "BTC")
        av, pdi, mdi, ap, adx_source = daily_adx_with_source("bitcoin", "BTC")
        bm = true_breakout_metrics(exchange_name, "BTC")
        # BTC is the benchmark, so BTC-relative strength is intentionally not used here.
        bullish_trend = (pd.notna(av) and av >= 25 and pd.notna(pdi) and pd.notna(mdi) and pdi > mdi)
        trend_strengthening = bullish_trend and pd.notna(ap) and av > ap
        if bm.get("daily_breakout") and bm.get("weekly_breakout") and bm.get("volume_confirmed") and bm.get("close_near_high") and bullish_trend and trend_strengthening and pd.notna(wrsi) and wrsi < 75 and pd.notna(sk) and pd.notna(sd) and sk >= 50 and sk > sd:
            status = "🚀 BTC STRUCTURAL BREAKOUT"
        elif bullish_trend and trend_strengthening:
            status = "🟢 BTC STRENGTHENING BULLISH TREND"
        elif pd.notna(av) and av >= 25 and pd.notna(mdi) and pd.notna(pdi) and mdi > pdi:
            status = "🔴 BTC STRONG BEARISH TREND"
        else:
            status = "🟡 BTC MIXED / DEVELOPING"
        return {
            "price": float(b["current_price"]),
            "24h": float(b["price_change_percentage_24h"]),
            "7d": float(b.get("price_change_percentage_7d_in_currency", np.nan)),
            "mcap_m": float(b["market_cap"]) / 1e6,
            "volume_m": float(b["total_volume"]) / 1e6,
            "weighted_rsi": wrsi,
            "stoch_k": sk,
            "stoch_d": sd,
            "adx": av,
            "plus_di": pdi,
            "minus_di": mdi,
            "adx_rising": bool(pd.notna(av) and pd.notna(ap) and av > ap),
            "adx_source": adx_source,
            "stoch_source": stoch_source,
            "daily_breakout": bool(bm.get("daily_breakout")),
            "weekly_breakout": bool(bm.get("weekly_breakout")),
            "volume_confirmed": bool(bm.get("volume_confirmed")),
            "close_near_high": bool(bm.get("close_near_high")),
            "status": status,
        }
    except Exception:
        return None

st.title("₿ Crypto Low-Cap TRUE BREAKOUT PRO Scanner")
st.caption("Background scanner + dashboard: volume surge + BTC-relative strength + RSI + downside resilience")

# Show the latest GitHub Actions background result if available.
try:
    if __import__("os").path.exists("data/latest_scan.json"):
        with open("data/latest_scan.json", "r") as f:
            bg=__import__("json").load(f)
        bm=bg.get("meta", {})
        st.info(f"Background scan: {bm.get('generated_at','not run yet')} | Regime: {bm.get('regime','N/A')} | BTC 24h: {bm.get('btc_24h','N/A')}% | BTC 7d: {bm.get('btc_7d','N/A')}%")
        if bg.get("top10"):
            st.subheader("⭐ Latest Background Top 10")
            bgdf=pd.DataFrame(bg["top10"])
            cols=[c for c in ["coin","ticker","score","signal","weighted_rsi","btc_rel_7d_pct","vol_mcap_pct","downside_beta","btc_down_day_rel_pct"] if c in bgdf.columns]
            st.dataframe(bgdf[cols].round(2) if not bgdf.empty else bgdf,use_container_width=True,hide_index=True)
except Exception:
    pass

with st.sidebar:
    st.header("Filters")
    mincap=st.number_input("Min market cap ($M)", min_value=20.0, max_value=1000.0, value=20.0, step=10.0, format="%.1f")
    maxcap=st.number_input("Max market cap ($M)", min_value=50.0, max_value=2000.0, value=500.0, step=25.0, format="%.1f")
    minvol=st.number_input("Min 24h volume ($M)", min_value=0.5, max_value=500.0, value=2.0, step=0.5, format="%.1f")
    minvr=st.number_input("Min volume / market cap (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0, format="%.1f")
    exchange=st.selectbox("Exchange candles",["bybit","okx","kraken"],index=0)
    st.caption("Exchange candle access is public; no trading/API key is required.")
    run=st.button("🚀 Run full scanner",type="primary")

    st.divider()
    st.caption("BTC is always analysed separately as the benchmark and is not subject to the low-cap market-cap filter.")



# BTC benchmark panel
try:
    btc_ctx = btc_market_context(exchange)
    if btc_ctx:
        st.subheader("₿ Bitcoin Market Benchmark")
        bc1,bc2,bc3,bc4,bc5,bc6 = st.columns(6)
        bc1.metric("BTC Price", f"${btc_ctx['price']:,.0f}")
        bc2.metric("BTC 24h", f"{btc_ctx['24h']:.2f}%")
        bc3.metric("BTC 7d", f"{btc_ctx['7d']:.2f}%")
        bc4.metric("Weighted RSI", f"{btc_ctx['weighted_rsi']:.1f}" if pd.notna(btc_ctx['weighted_rsi']) else "N/A")
        bc5.metric("ADX-14", f"{btc_ctx['adx']:.1f}" if pd.notna(btc_ctx['adx']) else "N/A")
        bc6.metric("BTC Signal", btc_ctx['status'])
        btc_table = pd.DataFrame([{
            "Daily breakout": btc_ctx["daily_breakout"],
            "Weekly breakout": btc_ctx["weekly_breakout"],
            "Volume confirmed": btc_ctx["volume_confirmed"],
            "Close near high": btc_ctx["close_near_high"],
            "StochRSI %K": btc_ctx["stoch_k"],
            "StochRSI %D": btc_ctx["stoch_d"],
            "+DI": btc_ctx["plus_di"],
            "-DI": btc_ctx["minus_di"],
            "ADX rising": btc_ctx["adx_rising"],
            "Data Source": f"StochRSI: {btc_ctx.get('stoch_source','Unavailable')} | ADX/DMI: {btc_ctx.get('adx_source','Unavailable')}",
        }])
        st.dataframe(btc_table.round(1), use_container_width=True, hide_index=True)
        st.caption("BTC is treated as the benchmark: BTC-relative strength and the low-cap volume/MCap threshold are not applied to BTC itself.")
except Exception as e:
    st.warning(f"BTC benchmark unavailable: {e}")

if "run" not in st.session_state: st.session_state.run=True

if run or st.session_state.run:
    try:
        with st.spinner("Scanning CoinGecko Top 500…"):
            df=cg_markets()
            if getattr(df, "attrs", {}).get("coingecko_warning"):
                st.warning(df.attrs["coingecko_warning"] + " — continuing with the available market universe.")
        df["mcap_m"]=df.market_cap/1e6
        df["vol_m"]=df.total_volume/1e6
        df["vr"]=df.total_volume/df.market_cap*100
        df["btc_rel_7d"]=np.nan

        btc=df[df.id=="bitcoin"]
        btc7=float(btc["price_change_percentage_7d_in_currency"].iloc[0]) if not btc.empty else 0
        btc24=float(btc["price_change_percentage_24h"].iloc[0]) if not btc.empty else 0
        candidates=df[df.mcap_m.between(mincap,maxcap)&(df.vol_m>=minvol)&(df.vr>=minvr)].copy()
        stable={"tether","usd-coin","dai","usds","true-usd","usdd"}
        candidates=candidates[~candidates.id.isin(stable)].copy()

        # Always retain ZEN.
        zen=df[df.id==ZEN_ID]
        if not zen.empty:
            candidates=pd.concat([candidates,zen],ignore_index=True).drop_duplicates("id")

        candidates["btc_rel_7d"]=candidates["price_change_percentage_7d_in_currency"]-btc7
        candidates["vol_score"]=np.clip(candidates.vr/25*20,0,20)
        candidates["mom_score"]=np.clip((candidates["price_change_percentage_24h"]+10)*0.8,0,20)
        candidates["rel_score"]=np.clip((candidates.btc_rel_7d+10)*0.4,0,20)
        candidates["liq_score"]=np.clip(candidates.vr/10,0,20)
        candidates["base_score"]=candidates.vol_score+candidates.mom_score+candidates.rel_score+candidates.liq_score

        top15=candidates.sort_values("base_score",ascending=False).head(15).copy()
        zen_top=candidates[candidates.id==ZEN_ID]
        top=pd.concat([top15,zen_top],ignore_index=True).drop_duplicates("id").head(16).copy()
        st.subheader("Market scan")
        a,b,c,d=st.columns(4)
        a.metric("Candidates",len(candidates))
        b.metric("BTC 24h",f"{btc24:.2f}%")
        c.metric("BTC 7d",f"{btc7:.2f}%")
        c.metric("Exchange",exchange.upper())
        d.metric("Updated",datetime.now(timezone.utc).strftime("%H:%M UTC"))

        view=top[["name","symbol","mcap_m","vol_m","vr","price_change_percentage_24h",
                  "price_change_percentage_7d_in_currency","btc_rel_7d","base_score"]].copy()
        view.columns=["Coin","Ticker","MCap $M","Vol $M","Vol/MCap %","24h %","7d %","BTC-rel 7d %","Base score"]
        st.dataframe(view.round(2),use_container_width=True,hide_index=True)

        st.subheader("Multi-timeframe RSI heatmap")
        rows=[]
        weights={"1H":.05,"4H":.10,"1D":.20,"1W":.25,"1M":.20,"3M":.20}
        for _,x in top.head(10).iterrows():
            vals = multi_timeframe_rsi(str(x["symbol"]).upper())
            usable=[(v,weights[k]) for k,v in vals.items() if pd.notna(v)]
            score=np.average([v for v,w in usable],weights=[w for v,w in usable]) if usable else np.nan
            deep=sum(pd.notna(v) and v<30 for v in vals.values())>=2
            bull=sum(pd.notna(v) and v>=60 for v in vals.values())>=4
            bear=sum(pd.notna(v) and v<=40 for v in vals.values())>=4
            over=sum(pd.notna(v) and v>70 for v in vals.values())>=3
            row={"Coin":x["name"],"Ticker":x["symbol"].upper(),**vals,
                 "Weighted RSI":score,"Deep oversold":deep,"Bullish alignment":bull,
                 "Bearish alignment":bear,"Overbought":over}
            rows.append(row)
        heat=pd.DataFrame(rows)
        display_heat = heat.copy()
        for col in ["1H","4H","1D","1W","1M","3M"]:
            if col in display_heat.columns:
                display_heat[col] = display_heat[col].apply(lambda v: "N/A" if pd.isna(v) else round(float(v), 1))
        st.dataframe(display_heat,use_container_width=True,hide_index=True)
        st.caption("N/A = insufficient exchange history for a valid RSI-14, not a zero value.")
        st.subheader("Daily StochRSI")
        stoch_rows = []
        for _, x in top.head(10).iterrows():
            k, d, stoch_source = daily_stochrsi_with_source(str(x["id"]), str(x["symbol"]).upper())
            if pd.isna(k) or pd.isna(d):
                stoch_signal = "N/A"
            elif k < 20 and k > d:
                stoch_signal = "🟢 OVERSOLD BULLISH CROSS"
            elif k > 80 and k < d:
                stoch_signal = "🔴 OVERBOUGHT BEARISH CROSS"
            elif k > d:
                stoch_signal = "🟢 BULLISH"
            else:
                stoch_signal = "🔴 BEARISH"
            stoch_rows.append({
                "Coin": x["name"], "Ticker": str(x["symbol"]).upper(),
                "StochRSI %K": k, "StochRSI %D": d, "Data Source": stoch_source, "Signal": stoch_signal
            })
        stochdf = pd.DataFrame(stoch_rows)
        if not stochdf.empty:
            stoch_display = stochdf.copy()
            for col in ["StochRSI %K", "StochRSI %D"]:
                stoch_display[col] = stoch_display[col].apply(lambda v: "N/A" if pd.isna(v) else round(float(v), 1))
            st.dataframe(stoch_display, use_container_width=True, hide_index=True)
        st.caption("Daily StochRSI uses RSI-14, StochRSI-14, %K 3, %D 3 on completed daily candles. CoinGecko is used first, with Bybit → OKX → Kraken fallback when CoinGecko history is unavailable or rate-limited. <20 is oversold; >80 is overbought.")

        st.subheader("Daily ADX / DMI")
        adx_rows = []
        for _, x in top.head(10).iterrows():
            av, pdi, mdi, ap, adx_source = daily_adx_with_source(str(x["id"]), str(x["symbol"]).upper())
            if pd.isna(av):
                adx_signal = "N/A"
            elif av >= 25 and pdi > mdi and av > ap:
                adx_signal = "🟢 STRONG + RISING BULLISH TREND"
            elif av >= 25 and pdi > mdi:
                adx_signal = "🟢 STRONG BULLISH TREND"
            elif av >= 25 and mdi > pdi:
                adx_signal = "🔴 STRONG BEARISH TREND"
            elif av >= 20:
                adx_signal = "🟡 TREND DEVELOPING"
            else:
                adx_signal = "⚪ WEAK / RANGE"
            adx_rows.append({"Coin": x["name"], "Ticker": str(x["symbol"]).upper(),
                             "ADX-14": av, "+DI": pdi, "-DI": mdi,
                             "ADX rising": (bool(av > ap) if pd.notna(av) and pd.notna(ap) else False),
                             "Data Source": adx_source,
                             "Signal": adx_signal})
        adxdf = pd.DataFrame(adx_rows)
        if not adxdf.empty:
            adx_display = adxdf.copy()
            for col in ["ADX-14", "+DI", "-DI"]:
                adx_display[col] = adx_display[col].apply(lambda v: "N/A" if pd.isna(v) else round(float(v), 1))
            st.dataframe(adx_display, use_container_width=True, hide_index=True)
        st.caption("ADX-14 measures trend strength; +DI/-DI provide direction. ADX >25 is a common strong-trend reference, while rising ADX indicates strengthening trend.")

        st.subheader("Crypto regime filter")
        # Simple, transparent regime proxy from BTC 24h/7d and market breadth.
        breadth = float((candidates["price_change_percentage_24h"] > 0).mean() * 100) if len(candidates) else 0
        if btc24 > 2 and btc7 > 3 and breadth >= 55:
            regime, regime_adj = "RISK-ON", 1.00
        elif btc24 < -3 and btc7 < -5 and breadth < 35:
            regime, regime_adj = "RISK-OFF", 0.65
        else:
            regime, regime_adj = "MIXED / TRANSITION", 0.85
        r1,r2,r3=st.columns(3)
        r1.metric("Regime",regime)
        r2.metric("Low-cap breadth",f"{breadth:.0f}%")
        r3.metric("Regime multiplier",f"{regime_adj:.2f}")

        st.subheader("0–100 unified scanner score")
        final=[]
        for _,x in top.iterrows():
            rr=heat[heat.Ticker.eq(str(x.symbol).upper())]
            rs=float(rr["Weighted RSI"].iloc[0]) if not rr.empty and pd.notna(rr["Weighted RSI"].iloc[0]) else 50

            # RSI: reward constructive 45-70, avoid chasing extreme overbought.
            rsi_component=max(0,20-abs(rs-57)*0.45)

            # Trend / breakout component from 24h and 7d momentum.
            trend_component=min(10,max(0,(float(x["price_change_percentage_24h"])+5)*1.0))
            rel_component=min(10,max(0,(float(x["btc_rel_7d"])+5)*0.5))

            # Volume confirmation and liquidity.
            volume_component=min(15,max(0,float(x["vr"])/2))
            liquidity_component=min(5,max(0,float(x["vol_m"])/10))

            # ADX/DMI: 10 points. Reward a strong, rising bullish trend;
            # do not reward high ADX by itself because ADX has no direction.
            adx_value, plus_di, minus_di, adx_prev, adx_source = daily_adx_with_source(str(x["id"]), str(x["symbol"]).upper())
            if pd.isna(adx_value):
                adx_component = 0.0
            else:
                strength = np.clip((float(adx_value) - 15.0) / 20.0, 0.0, 1.0)
                direction = 1.0 if pd.notna(plus_di) and pd.notna(minus_di) and float(plus_di) > float(minus_di) else 0.0
                rising = 1.0 if pd.notna(adx_prev) and float(adx_value) > float(adx_prev) else 0.5
                adx_component = 10.0 * strength * direction * rising

            # Components total 100 before regime adjustment.
            raw = rsi_component + trend_component + rel_component + volume_component + liquidity_component + adx_component + 30
            score = min(100,max(0,raw*regime_adj))

            # TRUE BREAKOUT PRO confirmation and extension control.
            bm=true_breakout_metrics(exchange,str(x["symbol"]).upper()); down_beta,down_rel,down_hit=downside_resilience(exchange,str(x["symbol"]).upper()); bm["downside_beta"]=down_beta; bm["btc_down_day_rel_pct"]=down_rel; bm["btc_down_day_outperform_pct"]=down_hit; stoch_k,stoch_d,stoch_source=daily_stochrsi_with_source(str(x["id"]), str(x["symbol"]).upper())
            confidence=breakout_confidence(bm,x["btc_rel_7d"],x["vr"],rs,stoch_k,stoch_d,adx_value,plus_di,minus_di,adx_prev)
            true_break=is_true_breakout(bm,x["btc_rel_7d"],x["vr"],rs,stoch_k,stoch_d,adx_value,plus_di,minus_di,adx_prev)
            extended=bool((pd.notna(rs) and rs>=80) or (pd.notna(bm.get("breakout_distance_pct")) and bm["breakout_distance_pct"]>2))
            if bm.get("breakout_failed"): status="🚨 BREAKOUT FAILED"
            elif true_break and confidence>=90 and not extended: status="🚀 HIGH-CONVICTION TRUE BREAKOUT"
            elif true_break and extended: status="🚀 TRUE BREAKOUT — EXTENDED / WAIT FOR RETEST"
            elif true_break: status="🟢 TRUE BREAKOUT"
            elif confidence>=80: status="🟢 BREAKOUT CONFIRMED"
            elif confidence>=70: status="🟡 STRONG BREAKOUT WATCH"
            elif rs>=80 and score>=60: status="🟠 EXTENDED — WAIT FOR RESET"
            elif float(x["btc_rel_7d"])>10 and float(x["vr"])>=20 and rs<70: status="💪 RELATIVE-STRENGTH LEADER"
            elif score>=75: status="🟢 STRONG SETUP"
            elif score>=60: status="🟢 MOMENTUM CONFIRMED"
            elif score>=45: status="🟡 WATCH / PULLBACK"
            else: status="🔴 WEAK"

            final.append([x["name"],x["symbol"].upper(),round(score,1),status,round(confidence,1),round(rs,1),round(x["btc_rel_7d"],1),round(x["vr"],1),round(x["price_change_percentage_24h"],1),bm.get("daily_breakout",False),bm.get("weekly_breakout",False),bm.get("volume_confirmed",False),bm.get("close_near_high",False),bm.get("atr14",np.nan),bm.get("atr_breakout_distance",np.nan),bm.get("atr_expanding",False),bm.get("obv_breakout",False),bm.get("ema20",np.nan),bm.get("ema50",np.nan),bm.get("ema200",np.nan),bm.get("ema_bullish",False),bm.get("ema200_bullish",False),bm.get("bb_width",np.nan),bm.get("bb_expanding",False),bm.get("cmf20",np.nan),bm.get("cmf_bullish",False),bm.get("mfi14",np.nan),bm.get("mfi_bullish",False),bm.get("macd_hist",np.nan),bm.get("macd_accelerating",False),bm.get("breakout_accepted",False),bm.get("breakout_retest_held",False),bm.get("breakout_distance_pct",np.nan),bm.get("rvol5",np.nan),bm.get("rvol_accelerating",False),bm.get("breakout_failed",False),bm.get("downside_beta",np.nan),bm.get("btc_down_day_rel_pct",np.nan),bm.get("btc_down_day_outperform_pct",np.nan),bm.get("oi",np.nan),bm.get("funding",np.nan),stoch_k,stoch_d,adx_value,plus_di,minus_di,(adx_value>adx_prev if pd.notna(adx_value) and pd.notna(adx_prev) else False),f"StochRSI: {stoch_source} | ADX/DMI: {adx_source}"])

        finaldf=pd.DataFrame(final,columns=["Coin","Ticker","Score","Signal","Breakout confidence","Weighted RSI","BTC-rel 7d %","Vol/MCap %","24h %","Daily breakout","Weekly breakout","Volume confirmed","Close near high","ATR-14","Breakout / ATR","ATR expanding","OBV breakout","EMA20","EMA50","EMA200","EMA bullish","Above EMA200","BB width","BB expanding","CMF20","CMF bullish","MFI14","MFI bullish","MACD histogram","MACD accelerating","Breakout accepted","Retest held","Breakout distance %","RVOL 5D","RVOL accelerating","Breakout failed","Downside beta","BTC-down-day rel %","BTC-down-day outperform %","Open interest","Funding rate","StochRSI %K","StochRSI %D","ADX-14","+DI","-DI","ADX rising","Data Source"])
        finaldf=finaldf.sort_values("Score",ascending=False)
        st.dataframe(finaldf,use_container_width=True,hide_index=True)
        st.caption("TRUE BREAKOUT PRO: structure + volume/OBV + ATR + EMA + volatility + money flow + MACD + acceptance/retest + optional derivatives.")

        st.subheader("Scanner rules")
        st.markdown("""
        **TRUE BREAKOUT CORE:** 20-day + 20-week breakout, volume ≥1.5× prior 20-day average, close in top 25%, breakout ≥0.25 ATR-14, ATR expanding ≥10%, OBV new 20-day high, BTC-relative 7D ≥+5pp, volume/MCap ≥10%, weighted RSI <75%, bullish StochRSI, ADX-14 ≥25 and rising, +DI−DI ≥5, and EMA20 > EMA50 when available.

**FAILURE CONTROL:** a post-breakout close back below resistance is labelled **🚨 BREAKOUT FAILED**. Retest-held confirmation is tracked separately rather than treating every initial breakout as fully confirmed.

        **CONFIDENCE 0–100:** adds EMA200, Bollinger expansion, CMF, MFI, MACD acceleration, breakout acceptance/retest, stronger BTC-relative strength, optional open interest and funding.

        **EXTENSION CONTROL:** confirmed breakouts >~2% above resistance or weighted RSI ≥80 are labelled extended / wait for retest.

        **Important:** this is a ranking/filtering engine, not a prediction or buy/sell system. Derivatives are optional and never block a signal when unavailable.
        """)

        st.info("The scanner is designed to surface candidates for further analysis. It does not execute trades and does not provide financial advice.")
    except Exception as e:
        st.error(f"Scanner error: {e}")
