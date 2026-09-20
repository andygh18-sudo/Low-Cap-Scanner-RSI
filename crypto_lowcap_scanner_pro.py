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

@st.cache_data(ttl=180)
def exchange_ohlcv(exchange_name, symbol, timeframe, limit=250):
    ex = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

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


def is_true_breakout(metrics, btc_rel_7d, volume_mcap, weighted_rsi):
    """Final TRUE BREAKOUT gate."""
    return all([
        bool(metrics.get("daily_breakout")),
        bool(metrics.get("weekly_breakout")),
        bool(metrics.get("volume_confirmed")),
        bool(metrics.get("close_near_high")),
        pd.notna(btc_rel_7d) and float(btc_rel_7d) >= 5.0,
        pd.notna(volume_mcap) and float(volume_mcap) >= 10.0,
        pd.notna(weighted_rsi) and float(weighted_rsi) < 75.0,
    ])

def signal(row):
    if row["score"]>=75:return "🟢 BREAKOUT CONFIRMATION"
    if row["score"]>=60:return "🟢 MOMENTUM CONFIRMED"
    if row["score"]>=45:return "🟡 WATCH / PULLBACK"
    return "🔴 WEAK"

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

            raw = rsi_component + trend_component + rel_component + volume_component + liquidity_component + 25
            score = min(100,max(0,raw*regime_adj))

            # Structural TRUE BREAKOUT: price + weekly structure + volume + candle quality
            # + BTC-relative strength + liquidity + RSI confirmation.
            bm = true_breakout_metrics(exchange, str(x["symbol"]).upper())
            true_break = is_true_breakout(bm, x["btc_rel_7d"], x["vr"], rs)

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
                          bm.get("daily_resistance",np.nan), bm.get("weekly_resistance",np.nan)])

        finaldf=pd.DataFrame(final,columns=["Coin","Ticker","Score","Signal","Weighted RSI","BTC-rel 7d %","Vol/MCap %","24h %",
                                             "Daily breakout","Weekly breakout","Volume confirmed","Close near high",
                                             "Daily resistance","Weekly resistance"])
        finaldf=finaldf.sort_values("Score",ascending=False)
        st.dataframe(finaldf,use_container_width=True,hide_index=True)

        st.subheader("Scanner rules")
        st.markdown("""
        **Score components:** multi-timeframe RSI (20) + trend/momentum (10) + BTC-relative strength (10) +
        volume confirmation (15) + liquidity (5) + base quality (25), then adjusted by the market regime.

        **🚀 TRUE BREAKOUT:** closed above the previous 20-day high by at least 0.5% AND above the previous 20-week high by at least 0.25%, with daily volume at least 1.5× the prior 20-day average, the breakout candle closing in its top 25%, BTC-relative 7D strength ≥ +5 percentage points, volume/market-cap ≥ 10%, and weighted RSI < 75.

        **Important:** this is a ranking/filtering engine, not a prediction or buy/sell system.
        A high score means several measured conditions are aligned; it does not guarantee future performance.
        Extreme volume can represent accumulation or distribution.
        """)

        st.info("The scanner is designed to surface candidates for further analysis. It does not execute trades and does not provide financial advice.")
    except Exception as e:
        st.error(f"Scanner error: {e}")
