import yfinance as yf
print("Starting YF download...")
tickers = {'SP100_Price': 'OEF', 'VIX': '^VIX'}
market_data = yf.download(
    list(tickers.values()),
    start="2001-01-01",
    end="2026-01-01",
    interval='1d',
    auto_adjust=True,
    progress=False
)
print("Finished YF download!")
