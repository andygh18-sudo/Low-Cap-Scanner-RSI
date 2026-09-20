import time
import numpy as np
import pandas as pd
import ccxt

EXCHANGE_NAMES = ["bybit", "okx", "kraken"]


def rsi(values, period=14):
    s = pd.Series(values, dtype=float).dropna()
    if len(s) < period + 1:
        return np.nan
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # If there were no losses at all, RSI is 100 rather than NaN.
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return float(out.iloc[-1]) if pd.notna(out.iloc[-1]) else np.nan


def _exchange(name):
    return getattr(ccxt, name)({"enableRateLimit": True, "timeout": 30000})


def _find_symbol(ex, ticker):
    t = str(ticker).upper()
    for quote in ("USDT", "USDC", "USD"):
        symbol = f"{t}/{quote}"
        if symbol in ex.markets:
            return symbol
    # Some exchanges expose symbols such as ZEN/USDT:USDT; prefer spot.
    for symbol, market in ex.markets.items():
        if market.get("spot") and str(market.get("base", "")).upper() == t and str(market.get("quote", "")).upper() in {"USDT", "USDC", "USD"}:
            return symbol
    return None


def fetch_ohlcv(ticker, timeframe, limit=300, daily_history=False):
    """Try Bybit -> OKX -> Kraken and return a DataFrame with a UTC DatetimeIndex."""
    last_error = None
    for name in EXCHANGE_NAMES:
        ex = None
        try:
            ex = _exchange(name)
            ex.load_markets()
            symbol = _find_symbol(ex, ticker)
            if not symbol:
                continue
            params = {"paginate": True, "paginationCalls": 4} if daily_history else {}
            candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
            if not candles:
                continue
            df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.drop_duplicates("ts").set_index("ts").sort_index()
            if len(df) >= 15:
                return df
        except Exception as exc:
            last_error = exc
        finally:
            try:
                if ex is not None:
                    ex.close()
            except Exception:
                pass
    return pd.DataFrame()


def timeframe_rsi(ticker, timeframe):
    df = fetch_ohlcv(ticker, timeframe, limit=300, daily_history=False)
    return rsi(df["close"]) if not df.empty else np.nan


def multi_timeframe_rsi(ticker, daily_fallback=None):
    """Return 1H/4H from exchange candles and 1D/1W/1M/3M from daily candles.

    Weekly/monthly/quarterly values are only returned when enough source candles exist.
    This avoids misleading None/zero values when the history is genuinely insufficient.
    """
    out = {"1H": np.nan, "4H": np.nan, "1D": np.nan, "1W": np.nan, "1M": np.nan, "3M": np.nan}
    for tf in ("1h", "4h"):
        out[tf.upper()] = timeframe_rsi(ticker, tf)

    daily = fetch_ohlcv(ticker, "1d", limit=1000, daily_history=True)
    if daily.empty and daily_fallback is not None:
        daily = daily_fallback.copy()
    if daily.empty:
        return out

    s = daily["close"].astype(float).dropna().sort_index()
    out["1D"] = rsi(s)

    # Calendar periods. These require at least 15 completed observations for RSI-14.
    weekly = s.resample("W-SUN").last().dropna()
    monthly = s.resample("ME").last().dropna()
    quarterly = s.resample("QE").last().dropna()
    out["1W"] = rsi(weekly) if len(weekly) >= 15 else np.nan
    out["1M"] = rsi(monthly) if len(monthly) >= 15 else np.nan
    out["3M"] = rsi(quarterly) if len(quarterly) >= 15 else np.nan
    return out


def weighted_rsi(values, weights=None):
    if weights is None:
        weights = {"1H": .05, "4H": .10, "1D": .20, "1W": .25, "1M": .20, "3M": .20}
    usable = [(values[k], weights[k]) for k in weights if k in values and pd.notna(values[k])]
    if not usable:
        return np.nan
    return float(np.average([v for v, _ in usable], weights=[w for _, w in usable]))
