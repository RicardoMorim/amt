import asyncio
import json
import logging
import websockets
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BinanceDataCollector:
    """
    Connects to Binance USD-M Futures WebSocket to collect tick-level trade data.
    This provides the exact data needed for CVD calculation (knowing if trade was buy or sell).
    """
    
    # Binance Futures USD-M aggregate trade stream endpoint
    WS_URL = "wss://fstream.binance.com/ws/{}@aggTrade"
    
    def __init__(self, symbol="BTCUSDT", callback=None):
        self.symbol = symbol.lower()
        self.url = self.WS_URL.format(self.symbol)
        
        # We can pass a callback function that will be triggered on every incoming trade
        # e.g., to update the session profile manager live
        self.callback = callback
        
        # Buffer for batch saving to DB/File if needed
        self.trade_buffer = []
        self.buffer_size = 1000
        
    async def _handle_message(self, message):
        data = json.loads(message)
        
        # Aggregated Trade fields:
        # e: Event type ("aggTrade")
        # E: Event time
        # s: Symbol
        # p: Price (string)
        # q: Quantity (string)
        # m: Is the buyer the market maker? (True means the trade was a SELL at the bid, False means BUY at the ask)
        
        price = float(data['p'])
        qty = float(data['q'])
        is_buyer_maker = data['m']
        timestamp = pd.to_datetime(data['E'], unit='ms')
        
        # Essential for Order Flow (CVD)
        side = 'sell' if is_buyer_maker else 'buy'
        
        trade = {
            'timestamp': timestamp,
            'price': float(price),
            'volume': float(qty),
            'side': side
        }
        
        if self.callback:
            # Send directly to the engine
            self.callback(trade)
            
        # Example of buffering trades to save to parquet/csv periodically
        self.trade_buffer.append(trade)
        if len(self.trade_buffer) >= self.buffer_size:
            self._flush_buffer()
            
    def _flush_buffer(self):
        # In a real system, you'd write this pandas dataframe to InfluxDB, Postgres or .parquet
        df = pd.DataFrame(self.trade_buffer)
        logging.info(f"Buffered {len(df)} trades. Latest price: {df.iloc[-1]['price']}")
        self.trade_buffer.clear()

    async def start(self):
        logging.info(f"Connecting to Binance Futures WS for {self.symbol.upper()}...")
        
        async for websocket in websockets.connect(self.url):
            try:
                logging.info("Connected!")
                async for message in websocket:
                    await self._handle_message(message)
            except websockets.ConnectionClosed:
                logging.warning("Binance WS Connection lost. Reconnecting...")
                continue
            except Exception as e:
                logging.error(f"Binance WS Error: {e}")
                await asyncio.sleep(2)

if __name__ == "__main__":
    # Simple test execution
    try:
        collector = BinanceDataCollector(symbol="btcusdt")
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        print("\\nDisconnected by user.")
