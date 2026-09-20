import streamlit as st

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
import ccxt
from datetime import datetime, timezone
from rsi_engine import multi_timeframe_rsi, weighted_rsi

st.set_page_config(page_title="Crypto Low-Cap Pro Scanner", page_icon="₿", layout="wide")

CG = "https://api.coingecko.com/api/v3"
ZEN_ID = "horizen"
DEFAULT_EXCHANGE = "bybit"
COINGECKO_API_KEY = __import__("os").getenv("COINGECKO_API_KEY", "")
CG_HEADERS = {"accept": "application/json"}
if COINGECKO_API_KEY:
    CG_HEADERS["x-cg-demo-api-key"] = COINGECKO_API_KEY

@st.cache_data(ttl=300)
def cg_markets():
    p={"vs_currency":"usd","order":"market_cap_desc","per_page":250,"page":1,
       "sparkline":"false","price_change_percentage":"7d,30d"}
    r=requests.get(f"{CG}/coins/markets",params=p,timeout=30); r.raise_for_status()
    return pd.DataFrame(r.json())

@st.cache_data(ttl=300)
def cg_chart(cid, days=365):
    r=requests.get(f"{CG}/coins/{cid}/market_chart",
                   params={"vs_currency":"usd","days":days},timeout=30)
    r.raise_for_status(); return r.json()

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
    denom = (high - low).replace(0, np.nan)
    stoch = ((rs - low) / denom) * 100.0
    k = stoch.rolling(k_len, min_periods=k_len).mean()
    d = k.rolling(d_len, min_periods=d_len).mean()
    if k.dropna().empty or d.dropna().empty:
        return np.nan, np.nan
    return float(k.iloc[-1]), float(d.iloc[-1])

@st.cache_data(ttl=300)
def cg_daily_ohlc_from_market_chart(coin_id, days=90):
    """Build completed daily OHLC candles from CoinGecko market-chart prices.

    CoinGecko Demo OHLC becomes 4-day candles for ranges above 30 days,
    which is insufficient for a reliable ADX-14 after daily resampling.
    Market-chart data is hourly for 2-90 day ranges, so we aggregate the
    hourly prices into daily open/high/low/close candles.
    """
    url=f"{CG}/coins/{coin_id}/market_chart"
    r=requests.get(url, params={"vs_currency":"usd","days":str(days)},
                   headers=CG_HEADERS, timeout=30)
    r.raise_for_status()
    prices=r.json().get("prices",[])
    if not prices:
        return pd.DataFrame()
    d=pd.DataFrame(prices, columns=["ts","price"])
    d["ts"]=pd.to_datetime(d["ts"], unit="ms", utc=True)
    d=d.set_index("ts").sort_index()
    daily=d["price"].resample("1D").agg(["first","max","min","last"]).dropna()
    daily.columns=["open","high","low","close"]
    # Do not use the currently forming UTC day.
    if len(daily)>1:
        daily=daily.iloc[:-1].copy()
    return daily

@st.cache_data(ttl=300)
def cg_ohlc_daily(coin_id, days=30):
    """Return completed daily OHLC using CoinGecko market-chart prices.

    This avoids the Demo OHLC endpoint's coarse 4-day candles for >30 days.
    """
    return cg_daily_ohlc_from_market_chart(coin_id, max(int(days), 60))

def coin_id_for_row(row):
    return str(row.get("id", ""))

def rsi(x, n=14):
    s=pd.Series(x,dtype=float).dropna()
    if len(s)<n+1:return np.nan
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    al=l.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    rs=ag/al.replace(0,np.nan)
    z=100-(100/(1+rs))
    z=z.where(~((al==0)&(ag>0)),100.0)
    return float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else np.nan


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
    """Return Wilder-style ADX, +DI and -DI plus prior ADX for slope."""
    h = pd.Series(high, dtype=float).reset_index(drop=True)
    l = pd.Series(low, dtype=float).reset_index(drop=True)
    c = pd.Series(close, dtype=float).reset_index(drop=True)
    if len(c) < (2 * n + 2):
        return np.nan, np.nan, np.nan, np.nan

    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0))
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0))
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Wilder smoothing via recursive RMA (equivalent to alpha=1/n EMA after initialization).
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    p_dm = plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    m_dm = minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    plus_di = 100 * p_dm / atr.replace(0, np.nan)
    minus_di = 100 * m_dm / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    valid = adx.dropna()
    if len(valid) < 2:
        return np.nan, np.nan, np.nan, np.nan
    return (float(adx.iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1]),
            float(adx.iloc[-2]))

@st.cache_data(ttl=300)
def daily_adx(coin_id):
    """Daily ADX-14/DMI calculated entirely from CoinGecko data with a history buffer."""
    try:
        # ADX-14 needs at least 30 completed daily candles in adx_dmi().
        # Request a much larger CoinGecko OHLC window so that removing the
        # currently forming day still leaves enough observations.
        d=cg_ohlc_daily(coin_id, 90)
        if len(d) < 35:
            return np.nan, np.nan, np.nan, np.nan
        return adx_dmi(d["high"], d["low"], d["close"], 14)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


@st.cache_data(ttl=300)
def true_breakout_metrics(exchange_name, ticker):
    """
    Detect a structural TRUE BREAKOUT using public OHLCV candles.

    Daily confirmation:
      - close >= previous 20-day high * 1.005
      - current volume >= 1.5x previous 20-day average volume
      - close is in the top 25% of the candle range

    Weekly confirmation:
      - weekly close >= previous 20-week high * 1.0025

    Market confirmation is added by the caller:
      - BTC-relative 7D >= +5 percentage points
      - volume/market-cap >= 10%
      - weighted RSI < 75

    Returns metrics even when the final TRUE BREAKOUT condition is false.
    """
    out = {
        "daily_resistance": np.nan, "weekly_resistance": np.nan,
        "daily_breakout": False, "weekly_breakout": False,
        "volume_confirmed": False, "close_near_high": False,
        "true_breakout_data": False, "breakout_exchange": None,
    }
    try:
        ex = getattr(ccxt, exchange_name)({"enableRateLimit": True})
        ex.load_markets()
        symbol = find_market_symbol(ex, ticker)
        if not symbol:
            return out

        daily = ex.fetch_ohlcv(symbol, timeframe="1d", limit=80)
        if len(daily) < 25:
            return out

        d = pd.DataFrame(daily, columns=["ts","open","high","low","close","volume"])
        # Ignore the currently forming candle.
        d = d.iloc[:-1].copy() if len(d) > 1 else d
        if len(d) < 25:
            return out

        current = d.iloc[-1]
        prior20 = d.iloc[-21:-1]
        resistance = float(prior20["high"].max())
        avg_volume = float(prior20["volume"].mean())
        candle_range = float(current["high"] - current["low"])
        close_location = ((float(current["close"]) - float(current["low"])) / candle_range) if candle_range > 0 else 0

        daily_breakout = float(current["close"]) >= resistance * 1.005
        volume_confirmed = avg_volume > 0 and float(current["volume"]) >= avg_volume * 1.5
        close_near_high = close_location >= 0.75

        out.update({
            "daily_resistance": resistance,
            "daily_breakout": bool(daily_breakout),
            "volume_confirmed": bool(volume_confirmed),
            "close_near_high": bool(close_near_high),
            "breakout_exchange": exchange_name,
        })

        # Weekly structural confirmation. 80 daily candles can yield ~11 weeks,
        # so fetch weekly directly when the exchange supports it.
        try:
            weekly = ex.fetch_ohlcv(symbol, timeframe="1w", limit=30)
            if len(weekly) >= 21:
                w = pd.DataFrame(weekly, columns=["ts","open","high","low","close","volume"])
                w = w.iloc[:-1].copy() if len(w) > 1 else w
                if len(w) >= 21:
                    wc = w.iloc[-1]
                    wprior = w.iloc[-21:-1]
                    wres = float(wprior["high"].max())
                    wbreak = float(wc["close"]) >= wres * 1.0025
                    out["weekly_resistance"] = wres
                    out["weekly_breakout"] = bool(wbreak)
        except Exception:
            pass

        return out
    except Exception:
        return out


def is_true_breakout(metrics, btc_rel_7d, volume_mcap, weighted_rsi, stoch_k, stoch_d, adx_value, plus_di, minus_di, adx_prev):
    """Final TRUE BREAKOUT gate."""
    return all([
        bool(metrics.get("daily_breakout")),
        bool(metrics.get("weekly_breakout")),
        bool(metrics.get("volume_confirmed")),
        bool(metrics.get("close_near_high")),
        pd.notna(btc_rel_7d) and float(btc_rel_7d) >= 5.0,
        pd.notna(volume_mcap) and float(volume_mcap) >= 10.0,
        pd.notna(weighted_rsi) and float(weighted_rsi) < 75.0,
        pd.notna(stoch_k) and pd.notna(stoch_d) and float(stoch_k) > float(stoch_d),
        pd.notna(stoch_k) and float(stoch_k) >= 50.0,
        pd.notna(adx_value) and float(adx_value) >= 25.0,
        pd.notna(adx_prev) and float(adx_value) > float(adx_prev),
        pd.notna(plus_di) and pd.notna(minus_di) and float(plus_di) > float(minus_di),
    ])

def signal(row):
    if row["score"]>=75:return "🟢 BREAKOUT CONFIRMATION"
    if row["score"]>=60:return "🟢 MOMENTUM CONFIRMED"
    if row["score"]>=45:return "🟡 WATCH / PULLBACK"
    return "🔴 WEAK"


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
        sk, sd = daily_stochrsi("bitcoin")
        av, pdi, mdi, ap = daily_adx("bitcoin")
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
            "daily_breakout": bool(bm.get("daily_breakout")),
            "weekly_breakout": bool(bm.get("weekly_breakout")),
            "volume_confirmed": bool(bm.get("volume_confirmed")),
            "close_near_high": bool(bm.get("close_near_high")),
            "status": status,
        }
    except Exception:
        return None

st.title("₿ Crypto Low-Cap Pro Scanner")
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
            "Data Source": "CoinGecko",
        }])
        st.dataframe(btc_table.round(1), use_container_width=True, hide_index=True)
        st.caption("BTC is treated as the benchmark: BTC-relative strength and the low-cap volume/MCap threshold are not applied to BTC itself.")
except Exception as e:
    st.warning(f"BTC benchmark unavailable: {e}")

if "run" not in st.session_state: st.session_state.run=True

if run or st.session_state.run:
    try:
        with st.spinner("Scanning CoinGecko Top 250…"):
            df=cg_markets()
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

        top=candidates.sort_values("base_score",ascending=False).head(15).copy()
        st.subheader("Market scan")
        a,b,c,d=st.columns(4)
        a.metric("Candidates",len(candidates))
        b.metric("BTC 24h",f"{btc24:.2f}%")
        b.metric("BTC 7d",f"{btc7:.2f}%")
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
            k, d = daily_stochrsi(str(x["id"]))
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
                "StochRSI %K": k, "StochRSI %D": d, "Data Source": "CoinGecko", "Signal": stoch_signal
            })
        stochdf = pd.DataFrame(stoch_rows)
        if not stochdf.empty:
            stoch_display = stochdf.copy()
            for col in ["StochRSI %K", "StochRSI %D"]:
                stoch_display[col] = stoch_display[col].apply(lambda v: "N/A" if pd.isna(v) else round(float(v), 1))
            st.dataframe(stoch_display, use_container_width=True, hide_index=True)
        st.caption("StochRSI uses RSI-14, StochRSI-14, %K 3, %D 3. <20 is oversold; >80 is overbought.")

        st.subheader("Daily ADX / DMI")
        adx_rows = []
        for _, x in top.head(10).iterrows():
            av, pdi, mdi, ap = daily_adx(str(x["id"]))
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
                             "Data Source": "CoinGecko",
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
            adx_value, plus_di, minus_di, adx_prev = daily_adx(str(x["id"]))
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

            # Structural TRUE BREAKOUT: price + weekly structure + volume + candle quality
            # + BTC-relative strength + liquidity + RSI + StochRSI + ADX/DMI confirmation.
            bm = true_breakout_metrics(exchange, str(x["symbol"]).upper())
            stoch_k, stoch_d = daily_stochrsi(str(x["id"]))
            true_break = is_true_breakout(bm, x["btc_rel_7d"], x["vr"], rs, stoch_k, stoch_d,
                                           adx_value, plus_di, minus_di, adx_prev)

            if true_break:
                status="🚀 TRUE BREAKOUT"
            elif rs >= 80 and score >= 60:
                status="🟠 EXTENDED — WAIT FOR RESET"
            elif float(x["btc_rel_7d"]) > 10 and float(x["vr"]) >= 20 and rs < 70:
                status="💪 RELATIVE-STRENGTH LEADER"
            elif score >= 75:
                status="🟢 STRONG SETUP"
            elif score >= 60:
                status="🟢 MOMENTUM CONFIRMED"
            elif score >= 45:
                status="🟡 WATCH / PULLBACK"
            else:
                status="🔴 WEAK"

            final.append([x["name"],x["symbol"].upper(),round(score,1),status,
                          round(rs,1),round(x["btc_rel_7d"],1),round(x["vr"],1),
                          round(x["price_change_percentage_24h"],1),
                          bm.get("daily_breakout",False), bm.get("weekly_breakout",False),
                          bm.get("volume_confirmed",False), bm.get("close_near_high",False),
                          bm.get("daily_resistance",np.nan), bm.get("weekly_resistance",np.nan),
                          stoch_k, stoch_d, adx_value, plus_di, minus_di, (adx_value > adx_prev if pd.notna(adx_value) and pd.notna(adx_prev) else False), "CoinGecko"])

        finaldf=pd.DataFrame(final,columns=["Coin","Ticker","Score","Signal","Weighted RSI","BTC-rel 7d %","Vol/MCap %","24h %",
                                             "Daily breakout","Weekly breakout","Volume confirmed","Close near high",
                                             "Daily resistance","Weekly resistance","StochRSI %K","StochRSI %D",
                                             "ADX-14","+DI","-DI","ADX rising","Data Source"])
        finaldf=finaldf.sort_values("Score",ascending=False)
        st.dataframe(finaldf,use_container_width=True,hide_index=True)

        st.subheader("Scanner rules")
        st.markdown("""
        **Score components:** multi-timeframe RSI (20) + trend/momentum (10) + BTC-relative strength (10) +
        volume confirmation (15) + liquidity (5) + ADX/DMI trend confirmation (10) + base quality (30), then adjusted by the market regime.

        **🚀 TRUE BREAKOUT:** closed above the previous 20-day high by at least 0.5% AND above the previous 20-week high by at least 0.25%, with daily volume at least 1.5× the prior 20-day average, the breakout candle closing in its top 25%, BTC-relative 7D strength ≥ +5 percentage points, volume/market-cap ≥ 10%, weighted RSI < 75%, daily StochRSI %K ≥ 50 with %K > %D, **ADX-14 ≥ 25 and rising, and +DI > -DI**. ADX measures trend strength while +DI/-DI provide direction; a high ADX alone is not treated as bullish.

        **Important:** this is a ranking/filtering engine, not a prediction or buy/sell system.
        A high score means several measured conditions are aligned; it does not guarantee future performance.
        Extreme volume can represent accumulation or distribution.
        """)

        st.info("The scanner is designed to surface candidates for further analysis. It does not execute trades and does not provide financial advice.")
    except Exception as e:
        st.error(f"Scanner error: {e}")
