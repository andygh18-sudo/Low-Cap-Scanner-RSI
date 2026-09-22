import streamlit as st

st.set_page_config(page_title="Crypto Low-Cap TRUE BREAKOUT PRO v8", page_icon="₿", layout="wide")

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



# ------------------------- V8 DATA-CACHE ARCHITECTURE -------------------------
# Reuse one CCXT exchange instance per scan. CCXT explicitly recommends
# reusing exchange instances so the built-in rate limiter can work correctly.
_EXCHANGE_CACHE = {}
_OHLC_CACHE = {}
_ANALYSIS_CACHE = {}
_DERIV_CACHE = {}


def get_exchange(exchange_name):
    name = str(exchange_name).lower()
    if name not in _EXCHANGE_CACHE:
        ex = getattr(ccxt, name)({"enableRateLimit": True})
        ex.load_markets()
        _EXCHANGE_CACHE[name] = ex
    return _EXCHANGE_CACHE[name]


def _completed_ohlcv(ex, symbol, timeframe="1d", limit=260):
    """Fetch completed candles once per exchange/symbol/timeframe per process."""
    key = (ex.id, symbol, timeframe, int(limit))
    if key in _OHLC_CACHE:
        return _OHLC_CACHE[key].copy()
    try:
        rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not rows:
            d = pd.DataFrame(columns=["ts","open","high","low","close","volume"])
        else:
            d = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
            d = d.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            if len(d) > 1:
                d = d.iloc[:-1].copy()  # completed candles only
        _OHLC_CACHE[key] = d.copy()
        return d
    except Exception:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])


def _cache_key_coin(exchange_name, ticker):
    return (str(exchange_name).lower(), str(ticker).upper())


def _normalise_mtf_rsi(raw):
    """Normalize rsi_engine output to the scanner's six canonical timeframe keys."""
    if not isinstance(raw, dict):
        return {}
    aliases = {
        "1h":"1H", "1H":"1H", "60m":"1H", "60min":"1H",
        "4h":"4H", "4H":"4H", "240m":"4H", "240min":"4H",
        "1d":"1D", "1D":"1D", "daily":"1D",
        "1w":"1W", "1W":"1W", "weekly":"1W",
        "1m":"1M", "1M":"1M", "monthly":"1M",
        "3m":"3M", "3M":"3M", "quarterly":"3M",
    }
    out = {}
    for k, v in raw.items():
        ck = aliases.get(str(k).strip(), aliases.get(str(k).strip().lower()))
        if ck is None:
            continue
        try:
            fv = float(v)
            if np.isfinite(fv) and 0 <= fv <= 100:
                out[ck] = fv
        except (TypeError, ValueError):
            continue
    return out


def _rsi_from_close(close, period=14):
    """Return the latest RSI-14 from a close series, or NaN if insufficient data."""
    try:
        s = pd.Series(close, dtype=float).dropna().reset_index(drop=True)
        if len(s) < period + 1:
            return np.nan
        return rsi(s, period)
    except Exception:
        return np.nan


def _exchange_mtf_rsi_fallback(ex, symbol):
    """Build MTF RSI from exchange candles when available."""
    out = {}
    try:
        for tf, limit in (("1h", 120), ("4h", 120)):
            d = _completed_ohlcv(ex, symbol, tf, limit)
            if len(d) >= 30:
                out["1H" if tf == "1h" else "4H"] = _rsi_from_close(d["close"])

        d = _completed_ohlcv(ex, symbol, "1d", 420)
        if len(d) >= 30:
            out["1D"] = _rsi_from_close(d["close"])

        w = _completed_ohlcv(ex, symbol, "1w", 150)
        if len(w) >= 30:
            out["1W"] = _rsi_from_close(w["close"])
            ww = w.copy()
            ww["dt"] = pd.to_datetime(ww["ts"], unit="ms", utc=True)
            ww = ww.set_index("dt")
            monthly = ww["close"].resample("ME").last().dropna()
            quarterly = ww["close"].resample("QE").last().dropna()
            if len(monthly) >= 15:
                out["1M"] = _rsi_from_close(monthly)
            if len(quarterly) >= 15:
                out["3M"] = _rsi_from_close(quarterly)
    except Exception:
        pass
    return {k: float(v) for k, v in out.items() if pd.notna(v) and np.isfinite(float(v))}


def _coingecko_mtf_rsi_fallback(coin_id):
    """Guaranteed MTF RSI fallback from CoinGecko price history.

    This path is deliberately independent of exchange symbol availability.
    It guarantees that the weighted RSI can still be calculated when a coin is
    not listed on the selected candle exchange. Intraday RSI is omitted when
    CoinGecko does not provide suitable intraday history; the weighted score
    automatically re-normalizes the remaining valid timeframe weights.
    """
    out = {}
    try:
        # Three years gives enough daily history to construct daily/weekly/monthly/quarterly RSI.
        payload = cg_chart(coin_id, 1095)
        prices = payload.get("prices", []) if isinstance(payload, dict) else []
        if not prices:
            return out
        q = pd.DataFrame(prices, columns=["ts", "price"])
        q["dt"] = pd.to_datetime(q["ts"], unit="ms", utc=True)
        q["price"] = pd.to_numeric(q["price"], errors="coerce")
        q = q.dropna(subset=["dt", "price"]).sort_values("dt").set_index("dt")
        if q.empty:
            return out

        daily = q["price"].resample("1D").last().dropna()
        if len(daily) > 1:
            daily = daily.iloc[:-1]
        if len(daily) >= 30:
            out["1D"] = _rsi_from_close(daily)

        weekly = daily.resample("W-SUN").last().dropna()
        if len(weekly) >= 30:
            out["1W"] = _rsi_from_close(weekly)

        monthly = daily.resample("ME").last().dropna()
        if len(monthly) >= 15:
            out["1M"] = _rsi_from_close(monthly)

        quarterly = daily.resample("QE").last().dropna()
        if len(quarterly) >= 15:
            out["3M"] = _rsi_from_close(quarterly)
    except Exception:
        pass
    return {k: float(v) for k, v in out.items() if pd.notna(v) and np.isfinite(float(v))}


def _mtf_rsi_cached(ticker, ex=None, symbol=None, coin_id=None):
    """Return normalized MTF RSI with exchange-independent CoinGecko fallback.

    CoinGecko requires the CoinGecko coin ID (e.g. ``horizen``), not the ticker
    (e.g. ``ZEN``).  The coin ID is therefore carried explicitly so the fallback
    cannot silently query an invalid endpoint.
    """
    key = str(ticker).upper()
    cg_id = str(coin_id).strip() if coin_id else key
    cache_key = ("rsi", key, cg_id, getattr(ex, "id", None), symbol)
    if cache_key in _ANALYSIS_CACHE:
        return dict(_ANALYSIS_CACHE[cache_key])

    vals = {}
    try:
        vals = _normalise_mtf_rsi(multi_timeframe_rsi(key))
    except Exception:
        vals = {}

    # Fill missing timeframes from exchange candles first.
    if ex is not None and symbol:
        missing = {"1H", "4H", "1D", "1W", "1M", "3M"} - set(vals)
        if missing:
            fallback = _exchange_mtf_rsi_fallback(ex, symbol)
            for tf, value in fallback.items():
                vals.setdefault(tf, value)

    # Critical fallback: exchange listings are not universal. If the selected
    # exchange has no symbol for a candidate, use CoinGecko price history so
    # Weighted RSI is still available from valid longer timeframes.
    missing = {"1D", "1W", "1M", "3M"} - set(vals)
    if missing:
        fallback = _coingecko_mtf_rsi_fallback(cg_id)
        for tf, value in fallback.items():
            vals.setdefault(tf, value)

    _ANALYSIS_CACHE[cache_key] = dict(vals)
    return dict(vals)


def weighted_rsi_from_values(vals):
    """Calculate the weighted RSI without ever returning the literal None."""
    weights={"1H":.05,"4H":.10,"1D":.20,"1W":.25,"1M":.20,"3M":.20}
    if not isinstance(vals, dict):
        return np.nan
    usable=[]
    for k, w in weights.items():
        try:
            v=float(vals.get(k, np.nan))
            if np.isfinite(v) and 0 <= v <= 100:
                usable.append((v,w))
        except (TypeError, ValueError):
            continue
    if not usable:
        return np.nan
    # Re-normalize weights when one or more very-long timeframes are unavailable.
    return float(np.average([v for v,w in usable], weights=[w for v,w in usable]))


def supply_quality(row):
    """Transparent supply/FDV quality using CoinGecko market fields.

    Unlock schedules are not exposed by the /coins/markets response, so v8
    does not invent an unlock-risk number. It reports the measurable dilution
    proxies and marks unlock data unavailable unless an external source is added.
    """
    circ=pd.to_numeric(pd.Series([row.get("circulating_supply",np.nan)]),errors="coerce").iloc[0]
    total=pd.to_numeric(pd.Series([row.get("total_supply",np.nan)]),errors="coerce").iloc[0]
    maxs=pd.to_numeric(pd.Series([row.get("max_supply",np.nan)]),errors="coerce").iloc[0]
    mcap=pd.to_numeric(pd.Series([row.get("market_cap",np.nan)]),errors="coerce").iloc[0]
    fdv=pd.to_numeric(pd.Series([row.get("fully_diluted_valuation",np.nan)]),errors="coerce").iloc[0]
    circ_total=(circ/total*100) if pd.notna(circ) and pd.notna(total) and total>0 else np.nan
    circ_max=(circ/maxs*100) if pd.notna(circ) and pd.notna(maxs) and maxs>0 else np.nan
    fdv_mcap=(fdv/mcap) if pd.notna(fdv) and pd.notna(mcap) and mcap>0 else np.nan
    if pd.notna(circ_max):
        score=float(np.clip(circ_max,0,100))
    elif pd.notna(circ_total):
        score=float(np.clip(circ_total,0,100))
    else:
        score=np.nan
    if pd.notna(fdv_mcap) and fdv_mcap>1:
        score=max(0,score - min(30,(fdv_mcap-1)*15)) if pd.notna(score) else np.nan
    return {"circ_pct_total":circ_total,"circ_pct_max":circ_max,"fdv_mcap":fdv_mcap,"supply_score":score,"unlock_risk":"N/A"}


def base_quality_metrics(d):
    """Measure consolidation/base quality before the breakout."""
    if d is None or len(d)<65:
        return {"base_days":np.nan,"base_range_pct":np.nan,"base_atr_pct":np.nan,"base_bb_width":np.nan,"base_quality":np.nan}
    # Exclude the current breakout candle and measure the preceding 30 sessions.
    x=d.iloc[-61:-1].copy() if len(d)>=61 else d.iloc[:-1].copy()
    if len(x)<30:
        return {"base_days":np.nan,"base_range_pct":np.nan,"base_atr_pct":np.nan,"base_bb_width":np.nan,"base_quality":np.nan}
    base30=x.iloc[-30:]
    mid=float(base30.close.median()) if base30.close.notna().any() else np.nan
    rng=((float(base30.high.max())-float(base30.low.min()))/mid*100) if pd.notna(mid) and mid>0 else np.nan
    atr=atr_series(x.high,x.low,x.close,14)
    atr_pct=(float(atr.iloc[-1])/float(base30.close.iloc[-1])*100) if pd.notna(atr.iloc[-1]) and float(base30.close.iloc[-1])>0 else np.nan
    bbw,_,_=bollinger_metrics(base30.close)
    # Smaller range/ATR and a non-expanding base are preferable.
    q=0.0
    if pd.notna(rng): q += float(np.clip((25-rng)/25,0,1))*40
    if pd.notna(atr_pct): q += float(np.clip((8-atr_pct)/8,0,1))*30
    if pd.notna(bbw): q += float(np.clip((0.25-bbw)/0.25,0,1))*30
    return {"base_days":30,"base_range_pct":rng,"base_atr_pct":atr_pct,"base_bb_width":bbw,"base_quality":q}


def derivatives_history(ex,ticker):
    """Current + recent OI/funding context. History is optional per exchange."""
    key=(ex.id,str(ticker).upper())
    if key in _DERIV_CACHE:
        return dict(_DERIV_CACHE[key])
    out={"oi":np.nan,"funding":np.nan,"oi_1d_pct":np.nan,"oi_3d_pct":np.nan,"oi_7d_pct":np.nan,
         "funding_avg_1d":np.nan,"funding_avg_3d":np.nan,"funding_trend":np.nan,"derivatives_state":"N/A",
         "oi_available":False,"funding_available":False}
    try:
        symbol=_find_contract_symbol(ex,ticker)
        if not symbol:
            _DERIV_CACHE[key]=out; return dict(out)
        if getattr(ex,"has",{}).get("fetchOpenInterest"):
            oi=ex.fetch_open_interest(symbol)
            val=oi.get("openInterestValue",oi.get("openInterestAmount",np.nan))
            if pd.notna(val): out["oi"]=float(val); out["oi_available"]=True
        if getattr(ex,"has",{}).get("fetchOpenInterestHistory"):
            hist=ex.fetch_open_interest_history(symbol,timeframe="1d",limit=10)
            vals=[]
            for h in hist or []:
                v=h.get("openInterestValue",h.get("openInterestAmount",np.nan))
                ts=h.get("timestamp")
                if pd.notna(v): vals.append((ts,float(v)))
            vals=sorted(vals,key=lambda z:z[0] if z[0] is not None else 0)
            if vals:
                cur=vals[-1][1]
                def ch(n):
                    if len(vals)>n and vals[-1-n][1]!=0:return (cur/vals[-1-n][1]-1)*100
                    return np.nan
                out["oi_1d_pct"]=ch(1); out["oi_3d_pct"]=ch(3); out["oi_7d_pct"]=ch(7)
        if getattr(ex,"has",{}).get("fetchFundingRate"):
            fr=ex.fetch_funding_rate(symbol); val=fr.get("fundingRate",np.nan)
            if pd.notna(val): out["funding"]=float(val); out["funding_available"]=True
        if getattr(ex,"has",{}).get("fetchFundingRateHistory"):
            hist=ex.fetch_funding_rate_history(symbol,limit=30)
            vals=[float(h.get("fundingRate")) for h in (hist or []) if pd.notna(h.get("fundingRate"))]
            if vals:
                out["funding_avg_1d"]=float(np.mean(vals[-3:]))
                out["funding_avg_3d"]=float(np.mean(vals[-9:]))
                out["funding_trend"]=float(vals[-1]-vals[0]) if len(vals)>1 else 0.0
        oi_ch=out["oi_3d_pct"]
        fr=out["funding"]
        if pd.notna(oi_ch) and pd.notna(fr):
            if oi_ch>5 and fr>0.0005: state="CROWDED LONGS"
            elif oi_ch>5 and fr<0: state="SHORT-SQUEEZE FUEL"
            elif oi_ch< -5 and fr<0: state="SHORTS COVERING / DELEVERAGING"
            elif oi_ch>0: state="HEALTHY OI BUILD"
            else: state="NEUTRAL"
            out["derivatives_state"]=state
    except Exception:
        pass
    _DERIV_CACHE[key]=out
    return dict(out)


def analysis_bundle(exchange_name, coin_row):
    """Compute all reusable technical evidence once per coin."""
    ticker=str(coin_row["symbol"]).upper(); cid=str(coin_row["id"]); key=(str(exchange_name).lower(),cid,ticker)
    if key in _ANALYSIS_CACHE and isinstance(_ANALYSIS_CACHE[key],dict) and "daily" in _ANALYSIS_CACHE[key]:
        return _ANALYSIS_CACHE[key]
    out={"ticker":ticker,"id":cid,"daily":pd.DataFrame(),"weekly":pd.DataFrame(),"metrics":{},"rsi":{},"weighted_rsi":np.nan,
         "stoch_k":np.nan,"stoch_d":np.nan,"stoch_source":"Unavailable","adx":np.nan,"pdi":np.nan,"mdi":np.nan,"adx_prev":np.nan,"adx_source":"Unavailable",
         "down_beta":np.nan,"down_rel":np.nan,"down_hit":np.nan,"supply":supply_quality(coin_row),"base":{},"deriv":{}}
    try:
        # RSI is deliberately calculated BEFORE requiring an exchange symbol.
        # Many low-cap assets are not listed on the selected candle exchange,
        # but CoinGecko still has price history for them.
        try:
            ex=get_exchange(exchange_name)
        except Exception:
            ex=None
        sym=find_market_symbol(ex,ticker) if ex is not None else None

        vals=_mtf_rsi_cached(ticker, ex, sym, cid)
        out["rsi"]=vals
        out["weighted_rsi"]=weighted_rsi_from_values(vals)

        # Exchange-dependent technical calculations require a valid symbol.
        if not sym:
            _ANALYSIS_CACHE[key]=out; return out
        d=_completed_ohlcv(ex,sym,"1d",260); w=_completed_ohlcv(ex,sym,"1w",30)
        out["daily"]=d; out["weekly"]=w
        # Technical metrics from the same daily candle set.
        if len(d)>=60:
            cur=d.iloc[-1]; prior=d.iloc[-21:-1]; res=float(prior.high.max())
            bm=true_breakout_metrics_cached(ex,sym,d,w,res)
            out["metrics"]=bm
            out["base"]=base_quality_metrics(d)
            # BTC-down-day resilience uses timestamp-aligned daily returns.
            out["down_beta"],out["down_rel"],out["down_hit"]=downside_resilience_cached(ex,sym)
        # RSI was already calculated above from the shared RSI path.
        # Prefer exchange daily calculations for the remaining indicators;
        # CoinGecko remains the RSI fallback.
        if len(d)>=50:
            sk,sd=stoch_rsi(d.close)
            out["stoch_k"],out["stoch_d"],out["stoch_source"]=sk,sd,ex.id.upper()
            av,pdi,mdi,ap=adx_dmi(d.high,d.low,d.close,14)
            if all(pd.notna(v) for v in (av,pdi,mdi,ap)):
                out["adx"],out["pdi"],out["mdi"],out["adx_prev"],out["adx_source"]=av,pdi,mdi,ap,ex.id.upper()
        if pd.isna(out["stoch_k"]):
            sk,sd,src=daily_stochrsi_with_source(cid,ticker); out["stoch_k"],out["stoch_d"],out["stoch_source"]=sk,sd,src
        if pd.isna(out["adx"]):
            av,pdi,mdi,ap,src=daily_adx_with_source(cid,ticker); out["adx"],out["pdi"],out["mdi"],out["adx_prev"],out["adx_source"]=av,pdi,mdi,ap,src
        out["deriv"]=derivatives_history(ex,ticker)
    except Exception:
        pass
    _ANALYSIS_CACHE[key]=out
    return out


def true_breakout_metrics_cached(ex,sym,d,w,res=None):
    """V8 breakout calculation using already-fetched candles and exchange instance."""
    out={"daily_resistance":np.nan,"weekly_resistance":np.nan,"daily_breakout":False,"weekly_breakout":False,"volume_confirmed":False,"close_near_high":False,"atr14":np.nan,"atr20_avg":np.nan,"atr_expanding":False,"atr_breakout_distance":np.nan,"atr_distance_confirmed":False,"obv_breakout":False,"ema20":np.nan,"ema50":np.nan,"ema200":np.nan,"ema_bullish":False,"ema200_bullish":False,"bb_width":np.nan,"bb_expanding":False,"cmf20":np.nan,"cmf_bullish":False,"mfi14":np.nan,"mfi_bullish":False,"macd_hist":np.nan,"macd_accelerating":False,"breakout_accepted":False,"breakout_retest_held":False,"breakout_distance_pct":np.nan,"rvol5":np.nan,"rvol_accelerating":False,"oi":np.nan,"funding":np.nan,"oi_available":False,"funding_available":False,"true_breakout_data":False,"breakout_exchange":ex.id,"breakout_failed":False}
    if len(d)<60:return out
    try:
        cur=d.iloc[-1]; prior=d.iloc[-21:-1]; res=float(res if res is not None else prior.high.max()); avgv=float(prior.volume.mean()); vol5=float(d.volume.iloc[-6:-1].mean()) if len(d)>=6 else np.nan
        rvol5=float(cur.volume/vol5) if pd.notna(vol5) and vol5>0 else np.nan; rng=float(cur.high-cur.low); loc=((float(cur.close)-float(cur.low))/rng) if rng>0 else 0
        atr=atr_series(d.high,d.low,d.close,14); ac=float(atr.iloc[-1]); a20=atr.iloc[-21:-1].dropna(); aavg=float(a20.mean()) if len(a20) else np.nan; adist=(float(cur.close)-res)/ac if ac>0 else np.nan
        obv=obv_series(d.close,d.volume); op=obv.iloc[-21:-1].dropna()
        e20s=ema_series(d.close,20); e50s=ema_series(d.close,50); e200s=ema_series(d.close,200); e20=float(e20s.iloc[-1]); e50=float(e50s.iloc[-1]); e200=float(e200s.iloc[-1]) if pd.notna(e200s.iloc[-1]) else np.nan
        bbw,bba,bbe=bollinger_metrics(d.close); cmf=cmf_series(d.high,d.low,d.close,d.volume); mfi=mfi_series(d.high,d.low,d.close,d.volume); mac,ms,mh,ma=macd_metrics(d.close); acc,ret,dist=acceptance_metrics(d,res)
        wres=np.nan; wbreak=False
        if len(w)>=21:
            wres=float(w.iloc[-21:-1].high.max()); wbreak=float(w.iloc[-1].close)>=wres*1.0025
        recent_closes=d.close.iloc[-6:]; prior_breakout=bool((recent_closes.iloc[:-1]>=res*1.005).any()) if len(recent_closes)>=2 else False
        failed=bool(prior_breakout and float(recent_closes.iloc[-1])<res)
        out.update({"daily_resistance":res,"weekly_resistance":wres,"daily_breakout":float(cur.close)>=res*1.005,"weekly_breakout":wbreak,"volume_confirmed":avgv>0 and float(cur.volume)>=avgv*1.5,"close_near_high":loc>=.75,"atr14":ac,"atr20_avg":aavg,"atr_expanding":pd.notna(aavg) and ac>=aavg*1.10,"atr_breakout_distance":adist,"atr_distance_confirmed":pd.notna(adist) and adist>=.25,"obv_breakout":pd.notna(obv.iloc[-1]) and len(op)>0 and obv.iloc[-1]>op.max(),"ema20":e20,"ema50":e50,"ema200":e200,"ema_bullish":float(cur.close)>e20>e50,"ema200_bullish":pd.notna(e200) and float(cur.close)>e200,"bb_width":bbw,"bb_expanding":bbe,"cmf20":float(cmf.iloc[-1]) if pd.notna(cmf.iloc[-1]) else np.nan,"cmf_bullish":pd.notna(cmf.iloc[-1]) and cmf.iloc[-1]>0,"mfi14":float(mfi.iloc[-1]) if pd.notna(mfi.iloc[-1]) else np.nan,"mfi_bullish":pd.notna(mfi.iloc[-1]) and 50<=mfi.iloc[-1]<85,"macd_hist":mh,"macd_accelerating":bool(ma and pd.notna(mac) and pd.notna(ms) and mac>ms and mh>0),"breakout_accepted":acc,"breakout_retest_held":ret,"breakout_distance_pct":dist,"rvol5":rvol5,"rvol_accelerating":pd.notna(rvol5) and rvol5>=1.5 and float(cur.volume)>=avgv*1.5,"breakout_failed":failed,"true_breakout_data":True})
        return out
    except Exception:return out


def downside_resilience_cached(ex,sym):
    """Timestamp-aligned downside resilience using the same exchange instance."""
    try:
        btc_sym=find_market_symbol(ex,"BTC")
        if not btc_sym:return np.nan,np.nan,np.nan
        a=_completed_ohlcv(ex,sym,"1d",90); b=_completed_ohlcv(ex,btc_sym,"1d",90)
        if len(a)<35 or len(b)<35:return np.nan,np.nan,np.nan
        ar=a.set_index("ts").close.pct_change(); br=b.set_index("ts").close.pct_change()
        q=pd.concat([ar.rename("coin"),br.rename("btc")],axis=1,join="inner").dropna(); q=q[q.btc<0]
        if len(q)<5:return np.nan,np.nan,np.nan
        beta=float(q.coin.cov(q.btc)/q.btc.var()) if q.btc.var()>0 else np.nan
        rel=float((q.coin-q.btc).mean()*100); hit=float((q.coin>q.btc).mean()*100)
        return beta,rel,hit
    except Exception:return np.nan,np.nan,np.nan


def breakout_state(m):
    """State machine for breakout lifecycle."""
    if m.get("breakout_failed"): return "🚨 BREAKOUT FAILED"
    if m.get("daily_breakout") and m.get("weekly_breakout"):
        if m.get("breakout_retest_held"): return "🚀 RETEST HELD"
        if m.get("breakout_accepted"): return "🟢 BREAKOUT ACCEPTED"
        return "🟡 BREAKOUT ATTEMPT"
    return "⚪ BASE / PRE-BREAKOUT"


def technical_score_v8(m,rs,btc_rel,base,supply,down_rel,down_hit):
    """Clean 0-100 technical score. Regime is deliberately separate."""
    parts=[]
    def add(v,w): parts.append(float(np.clip(v,0,1))*w)
    add(1 if m.get("daily_breakout") else 0,15)
    add(1 if m.get("weekly_breakout") else 0,8)
    add(1 if m.get("volume_confirmed") else 0,10)
    add(1 if m.get("obv_breakout") else 0,8)
    add(1 if m.get("close_near_high") else 0,5)
    add(1 if m.get("atr_distance_confirmed") else 0,6)
    add(1 if m.get("atr_expanding") else 0,5)
    add(1 if m.get("ema_bullish") else 0,7)
    add(1 if m.get("ema200_bullish") else 0,4)
    add(1 if m.get("cmf_bullish") else 0,4)
    add(1 if m.get("mfi_bullish") else 0,3)
    add(1 if m.get("macd_accelerating") else 0,5)
    add(1 if m.get("bb_expanding") else 0,4)
    add(1 if m.get("breakout_accepted") else 0,4)
    add(1 if m.get("breakout_retest_held") else 0,6)
    if pd.notna(rs): add(1 if 45<=rs<75 else (0.5 if 35<=rs<85 else 0),4)
    if pd.notna(btc_rel): add(np.clip((btc_rel+5)/15,0,1),4)
    if pd.notna(down_rel): add(np.clip((down_rel+5)/10,0,1),2)
    if pd.notna(down_hit): add(np.clip(down_hit/100,0,1),2)
    if pd.notna(base.get("base_quality")): add(base["base_quality"]/100,2)
    return float(np.clip(sum(parts),0,100))


def confidence_v8(m,rs,sk,sd,av,pdi,mdi,ap,btc_rel):
    return breakout_confidence(m,btc_rel,m.get("vr",np.nan),rs,sk,sd,av,pdi,mdi,ap)

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

st.title("₿ Crypto Low-Cap TRUE BREAKOUT PRO v8 Scanner")
st.caption("v8: cached multi-layer market analysis + 50–100 candidate pre-screen + structural breakout state machine")

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

# ------------------------------ V8 DASHBOARD ------------------------------
with st.sidebar:
    st.header("Filters")
    mincap=st.number_input("Min market cap ($M)", min_value=20.0, max_value=1000.0, value=20.0, step=10.0, format="%.1f", key="v8_mincap")
    maxcap=st.number_input("Max market cap ($M)", min_value=50.0, max_value=2000.0, value=500.0, step=25.0, format="%.1f", key="v8_maxcap")
    minvol=st.number_input("Min 24h volume ($M)", min_value=0.5, max_value=500.0, value=2.0, step=0.5, format="%.1f", key="v8_minvol")
    minvr=st.number_input("Min volume / market cap (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0, format="%.1f", key="v8_minvr")
    preselect=st.slider("Technical pre-screen size", min_value=50, max_value=100, value=60, step=10, key="v8_preselect")
    exchange=st.selectbox("Exchange candles",["bybit","okx","kraken"],index=0, key="v8_exchange")
    run=st.button("🚀 Run full scanner",type="primary", key="v8_run")
    st.caption("v8 analyses a broad pre-screen before applying the expensive structural breakout engine. ZEN is always retained.")

st.title("₿ Crypto Low-Cap TRUE BREAKOUT PRO v8")
st.caption("Cached technical engine + 50–100 pre-screen + breakout lifecycle + supply quality + downside resilience")

if "run" not in st.session_state: st.session_state.run=True
if run or st.session_state.run:
    try:
        with st.spinner("Loading market universe with CoinGecko 429 protection…"):
            df=cg_markets()
        if df.empty:
            st.error("No market-universe data returned.")
            st.stop()
        warning=getattr(df,"attrs",{}).get("coingecko_warning")
        if warning: st.warning(str(warning)+" — continuing with available universe.")
        df["mcap_m"]=pd.to_numeric(df.market_cap,errors="coerce")/1e6
        df["vol_m"]=pd.to_numeric(df.total_volume,errors="coerce")/1e6
        df["vr"]=pd.to_numeric(df.total_volume,errors="coerce")/pd.to_numeric(df.market_cap,errors="coerce")*100
        stable={"tether","usd-coin","dai","usds","true-usd","usdd"}
        btc=df[df.id.eq("bitcoin")]
        btc7=float(btc["price_change_percentage_7d_in_currency"].iloc[0]) if not btc.empty and pd.notna(btc["price_change_percentage_7d_in_currency"].iloc[0]) else 0.0
        btc24=float(btc["price_change_percentage_24h"].iloc[0]) if not btc.empty and pd.notna(btc["price_change_percentage_24h"].iloc[0]) else 0.0
        candidates=df[df.mcap_m.between(mincap,maxcap)&(df.vol_m>=minvol)&(df.vr>=minvr)&~df.id.isin(stable)].copy()
        zen=df[df.id.eq(ZEN_ID)]
        if not zen.empty: candidates=pd.concat([candidates,zen],ignore_index=True).drop_duplicates("id")
        candidates["btc_rel_7d"]=pd.to_numeric(candidates["price_change_percentage_7d_in_currency"],errors="coerce")-btc7
        candidates["vol_score"]=np.clip(candidates.vr/25*20,0,20)
        candidates["mom_score"]=np.clip((candidates["price_change_percentage_24h"]+10)*0.8,0,20)
        candidates["rel_score"]=np.clip((candidates.btc_rel_7d+10)*0.4,0,20)
        candidates["liq_score"]=np.clip(candidates.vr/10,0,20)
        candidates["base_score"]=candidates.vol_score+candidates.mom_score+candidates.rel_score+candidates.liq_score
        # Broad pre-screen: 50–100 candidates, not 15. ZEN is always retained.
        presel=candidates.sort_values("base_score",ascending=False).head(int(preselect)).copy()
        if not zen.empty:
            presel=pd.concat([presel,zen],ignore_index=True).drop_duplicates("id")
        # Limit the expensive derivatives history to the pre-screen itself; all other metrics are cached.
        st.subheader("Market universe")
        c1,c2,c3,c4,c5=st.columns(5)
        c1.metric("Filtered candidates",len(candidates)); c2.metric("Pre-screen",len(presel)); c3.metric("BTC 24h",f"{btc24:.2f}%"); c4.metric("BTC 7d",f"{btc7:.2f}%"); c5.metric("Exchange",exchange.upper())
        view=presel[["name","symbol","mcap_m","vol_m","vr","price_change_percentage_24h","price_change_percentage_7d_in_currency","btc_rel_7d","base_score"]].copy()
        view.columns=["Coin","Ticker","MCap $M","Vol $M","Vol/MCap %","24h %","7d %","BTC-rel 7d %","Pre-screen score"]
        st.dataframe(view.round(2),use_container_width=True,hide_index=True)

        # One analysis bundle per pre-screen coin. This is the core v8 cache architecture.
        bundles={}
        progress=st.progress(0,text="Building cached technical bundles…")
        total=max(1,len(presel))
        for i,(_,x) in enumerate(presel.iterrows(),1):
            bundles[str(x["id"])] = analysis_bundle(exchange,x)
            progress.progress(i/total,text=f"Analysing {i}/{total}: {str(x['symbol']).upper()}")
        progress.empty()

        # Build RSI heatmap from the same cached values used by final scoring.
        rows=[]
        for _,x in presel.iterrows():
            b=bundles[str(x.id)]; vals=b["rsi"]; row={"Coin":x["name"],"Ticker":str(x["symbol"]).upper(),**vals,"Weighted RSI":b["weighted_rsi"]}
            row["Deep oversold"]=sum(pd.notna(v) and v<30 for v in vals.values())>=2
            row["Bullish alignment"]=sum(pd.notna(v) and v>=60 for v in vals.values())>=4
            row["Bearish alignment"]=sum(pd.notna(v) and v<=40 for v in vals.values())>=4
            row["Overbought"]=sum(pd.notna(v) and v>70 for v in vals.values())>=3
            rows.append(row)
        heat=pd.DataFrame(rows)
        # Never expose Python None/NaN as the Weighted RSI value. If a coin has
        # at least one valid timeframe, recompute directly from the displayed
        # RSI dictionary; otherwise show N/A explicitly.
        if not heat.empty:
            heat["Weighted RSI"] = heat.apply(
                lambda r: weighted_rsi_from_values({tf: r.get(tf, np.nan) for tf in ["1H","4H","1D","1W","1M","3M"]}),
                axis=1
            )
        st.subheader("Multi-timeframe RSI heatmap — cached")
        st.dataframe(heat.round(1),use_container_width=True,hide_index=True)

        # Simple regime context remains separate from technical score.
        breadth=float((candidates["price_change_percentage_24h"]>0).mean()*100) if len(candidates) else 0
        if btc24>2 and btc7>3 and breadth>=55: regime="RISK-ON"; regime_score=85
        elif btc24<-3 and btc7<-5 and breadth<35: regime="RISK-OFF"; regime_score=25
        else: regime="MIXED / TRANSITION"; regime_score=55
        r1,r2,r3=st.columns(3); r1.metric("Regime",regime); r2.metric("Low-cap breadth",f"{breadth:.0f}%"); r3.metric("Regime score",regime_score)

        final=[]
        for _,x in presel.iterrows():
            b=bundles[str(x.id)]; m=dict(b["metrics"]); m["vr"]=float(x["vr"]); rs=b["weighted_rsi"]; sk=b["stoch_k"]; sd=b["stoch_d"]; av=b["adx"]; pdi=b["pdi"]; mdi=b["mdi"]; ap=b["adx_prev"]; btc_rel=float(x["btc_rel_7d"]); base=b["base"]; supply=b["supply"]; der=b["deriv"]
            m.update({"btc_down_day_rel_pct":b["down_rel"]})
            tech=technical_score_v8(m,rs,btc_rel,base,supply,b["down_rel"],b["down_hit"])
            confidence=confidence_v8(m,rs,sk,sd,av,pdi,mdi,ap,btc_rel)
            true_break=is_true_breakout(m,btc_rel,float(x["vr"]),rs,sk,sd,av,pdi,mdi,ap)
            state=breakout_state(m)
            # Risk-adjusted score is separate from technical score.
            risk_adj=tech*(0.70+0.003*regime_score)
            if pd.notna(supply.get("supply_score")): risk_adj += (supply["supply_score"]-50)*0.05
            if pd.notna(b["down_hit"]): risk_adj += (b["down_hit"]-50)*0.04
            if pd.notna(b["down_rel"]): risk_adj += np.clip(b["down_rel"]*0.3,-5,5)
            risk_adj=float(np.clip(risk_adj,0,100))
            extended=bool((pd.notna(rs) and rs>=80) or (pd.notna(m.get("breakout_distance_pct")) and m["breakout_distance_pct"]>2))
            if m.get("breakout_failed"): status="🚨 BREAKOUT FAILED"
            elif true_break and confidence>=90 and not extended: status="🚀 HIGH-CONVICTION TRUE BREAKOUT"
            elif true_break and extended: status="🚀 TRUE BREAKOUT — EXTENDED / WAIT FOR RETEST"
            elif true_break: status="🟢 TRUE BREAKOUT"
            elif confidence>=80: status="🟢 BREAKOUT CONFIRMATION"
            elif state=="🟡 BREAKOUT ATTEMPT": status="🟡 BREAKOUT WATCH"
            elif risk_adj>=75: status="🟢 STRONG SETUP"
            elif risk_adj>=60: status="🟢 MOMENTUM CONFIRMED"
            elif risk_adj>=45: status="🟡 WATCH / PULLBACK"
            else: status="🔴 WEAK"
            final.append({
                "Coin":x["name"],"Ticker":str(x["symbol"]).upper(),"Technical score":round(tech,1),"Risk-adjusted score":round(risk_adj,1),"Signal":status,"Breakout state":state,"Breakout confidence":round(confidence,1),"Weighted RSI":round(rs,1) if pd.notna(rs) else np.nan,"BTC-rel 7d %":round(btc_rel,1),"Vol/MCap %":round(float(x["vr"]),1),"MCap $M":round(float(x["mcap_m"]),1),"24h %":round(float(x["price_change_percentage_24h"]),1),
                "Daily breakout":m.get("daily_breakout",False),"Weekly breakout":m.get("weekly_breakout",False),"Volume confirmed":m.get("volume_confirmed",False),"OBV breakout":m.get("obv_breakout",False),"ATR expanding":m.get("atr_expanding",False),"EMA bullish":m.get("ema_bullish",False),"Breakout accepted":m.get("breakout_accepted",False),"Retest held":m.get("breakout_retest_held",False),"Breakout distance %":m.get("breakout_distance_pct",np.nan),"RVOL 5D":m.get("rvol5",np.nan),
                "Base quality":base.get("base_quality",np.nan),"Base range %":base.get("base_range_pct",np.nan),"Supply score":supply.get("supply_score",np.nan),"Circulating % max":supply.get("circ_pct_max",np.nan),"FDV/MCap":supply.get("fdv_mcap",np.nan),"Unlock risk":supply.get("unlock_risk","N/A"),
                "Downside beta":b["down_beta"],"BTC-down-day rel %":b["down_rel"],"BTC-down-day outperform %":b["down_hit"],"OI 1D %":der.get("oi_1d_pct",np.nan),"OI 3D %":der.get("oi_3d_pct",np.nan),"OI 7D %":der.get("oi_7d_pct",np.nan),"Funding":der.get("funding",np.nan),"Funding 3D avg":der.get("funding_avg_3d",np.nan),"Derivatives state":der.get("derivatives_state","N/A"),
                "StochRSI %K":sk,"StochRSI %D":sd,"ADX":av,"+DI":pdi,"-DI":mdi,"ADX rising":bool(pd.notna(av) and pd.notna(ap) and av>ap),"Data sources":f"RSI: cached engine | Stoch: {b['stoch_source']} | ADX: {b['adx_source']} | OHLCV: {exchange.upper()}"
            })
        finaldf=pd.DataFrame(final).sort_values("Risk-adjusted score",ascending=False)
        st.subheader("TRUE BREAKOUT PRO v8 — final ranking")
        st.dataframe(finaldf.round(2),use_container_width=True,hide_index=True)

        # Focus table: strongest structural candidates only.
        focus=finaldf[(finaldf["Signal"].str.contains("BREAKOUT|SETUP",regex=True)) | finaldf["Ticker"].eq("ZEN")].head(15)
        st.subheader("⭐ Breakout focus")
        st.dataframe(focus.round(2),use_container_width=True,hide_index=True)

        st.subheader("v8 architecture / rules")
        st.markdown("""
**1. Data cache:** one CCXT exchange instance and one OHLCV dataset per exchange/symbol/timeframe are reused across all indicators.

**2. Broad pre-screen:** 50–100 candidates are analysed before the expensive structural breakout engine; ZEN is always retained.

**3. Clean scoring:** Technical Quality is 0–100; Regime Score and Risk-Adjusted Score are separate.

**4. TRUE BREAKOUT:** retains the v7 structural gate — daily/weekly breakout, volume, candle quality, ATR distance/expansion, OBV, BTC-relative strength, RSI, StochRSI, ADX/DMI and EMA alignment.

**5. Breakout lifecycle:** BASE → BREAKOUT ATTEMPT → BREAKOUT ACCEPTED → RETEST HELD, with explicit BREAKOUT FAILED state.

**6. Derivatives:** current OI/funding plus recent OI/funding history when the exchange supports the CCXT unified methods; derivatives are context, not an automatic buy/sell gate.

**7. Base quality:** measures pre-breakout consolidation range, ATR and Bollinger compression.

**8. Supply quality:** circulating/max supply and FDV/MCap are included. Unlock schedules are explicitly shown as N/A because they are not available from the current CoinGecko market-universe response; v8 does not invent unlock data.

**9. Downside resilience:** BTC-down-day beta, average relative performance and outperform rate are timestamp-aligned.

**10. Data reuse:** RSI, StochRSI, ADX, OHLCV, derivatives and downside-resilience results are reused rather than recomputed in separate dashboard sections.
        """)
        st.caption("v8 is a research/ranking engine. It does not execute trades or guarantee outcomes.")
    except Exception as e:
        st.error(f"Scanner error: {e}")
