# Crypto Low-Cap Scanner — True Breakout Edition

This version adds a stricter TRUE BREAKOUT signal.

A coin can be labelled **🚀 TRUE BREAKOUT** only when:
- price clears the prior 20-candle resistance by at least 0.5%;
- breakout volume is at least 1.5x the prior 20-candle average on 1D, or 1.75x on 4H;
- the candle closes near its high (>=98% of candle high);
- BTC-relative 7D strength is > +5 percentage points;
- volume/market cap is >=10%;
- weighted RSI is below 75.

The scanner tries Bybit, OKX, then Kraken public OHLCV. The previous generic `RELATIVE-STRENGTH BREAKOUT` label is now `💪 RELATIVE-STRENGTH LEADER` so it is not confused with a price/volume breakout.

The RSI heatmap remains 1H/4H/1D/1W/1M/3M where data permits.
