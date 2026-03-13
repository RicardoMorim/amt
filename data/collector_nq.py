import asyncio
import logging
from alpaca_trade_api.stream import Stream
import pandas as pd
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AlpacaDataCollector:
    """
    Connects to Alpaca Markets WebSocket for real-time stock/crypto data.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in a .env file.
    """
    
    def __init__(self, symbols=["QQQ", "SPY"], callback=None):
        load_dotenv()
        
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            logging.error("ALPACA API Keys missing from environment / .env file!")
            
        self.symbols = symbols
        self.callback = callback
        
        # We can subscribe to multiple streams. 'iex' is free real-time data for stocks.
        self.stream = Stream(self.api_key, self.secret_key, base_url="https://paper-api.alpaca.markets", data_feed='iex')
        
        self.trade_buffer = []
        self.buffer_size = 1000

    async def trade_handler(self, t):
        """
        Alpaca trade objects have:
        t.S (Symbol), t.p (Price), t.s (Size), t.c (Conditions), t.t (Timestamp)
        
        Note: Stocks don't natively expose 'aggressor side' like Crypto futures.
        We have to infer it via tick test (price > previous price = buy).
        """
        # Convert nanoseconds to datetime
        timestamp = pd.to_datetime(t.t, unit='ns')
        
        trade = {
            'timestamp': timestamp,
            'symbol': t.S,
            'price': t.p,
            'volume': t.s,
            # We will rely on our CVD engine to infer the side for Alpaca/Stocks
            'side': None 
        }

        if self.callback:
            self.callback(trade)
            
        self.trade_buffer.append(trade)
        if len(self.trade_buffer) >= self.buffer_size:
            self._flush_buffer()
            
    def _flush_buffer(self):
        df = pd.DataFrame(self.trade_buffer)
        logging.info(f"Buffered {len(df)} Alpca trades. Latest price {df.iloc[-1]['symbol']}: {df.iloc[-1]['price']}")
        self.trade_buffer.clear()

    def start(self):
        logging.info(f"Connecting to Alpaca WS for {self.symbols}...")
        
        # Subscribe to trades
        self.stream.subscribe_trades(self.trade_handler, *self.symbols)
        
        try:
            self.stream.run()
        except KeyboardInterrupt:
            logging.info("\\nDisconnected by user.")
            
if __name__ == "__main__":
    # Simple test execution
    # To use this, create a .env with:
    # ALPACA_API_KEY=your_key
    # ALPACA_SECRET_KEY=your_secret
    collector = AlpacaDataCollector(symbols=["QQQ"]) # QQQ is ETF for Nasdaq100
    collector.start()
