def get_market_index_for_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()

    # India (NSE)
    if ticker.endswith(".NS"):
        return "^NSEI"   # Nifty 50

    # US / Global default
    return "^GSPC"       # S&P 500
