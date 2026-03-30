import asyncio
import json
import logging
import websockets
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Exponential backoff bounds (seconds)
_BACKOFF_MIN = 1
_BACKOFF_MAX = 60


class BinanceDataCollector:
    """
    Connects to Binance USD-M Futures WebSocket to collect tick-level trade data.

    Robustness improvements (C1):
      - Exponential backoff reconnect: waits 1s, 2s, 4s … up to 60s between
        reconnection attempts so we don't hammer Binance on sustained outages.
      - Heartbeat ping every 20 seconds to detect silent disconnections before
        Binance closes the connection server-side.
      - Attempt counter logged so you can see how many times we've reconnected.
      - Graceful shutdown: calling stop() sets a flag that exits the loop cleanly.
    """

    WS_URL = "wss://fstream.binance.com/ws/{}@aggTrade"

    def __init__(self, symbol: str = "BTCUSDT", callback=None):
        self.symbol      = symbol.lower()
        self.url         = self.WS_URL.format(self.symbol)
        self.callback    = callback
        self.trade_buffer: list = []
        self.buffer_size = 1000
        self._running    = True
        self._attempt    = 0

    def stop(self):
        """Signal the collector to stop after the current reconnect cycle."""
        self._running = False

    async def _handle_message(self, message: str):
        data  = json.loads(message)
        price = float(data['p'])
        qty   = float(data['q'])
        side  = 'sell' if data['m'] else 'buy'
        trade = {
            'timestamp': pd.to_datetime(data['E'], unit='ms'),
            'price':     price,
            'volume':    qty,
            'side':      side,
        }
        if self.callback:
            self.callback(trade)
        self.trade_buffer.append(trade)
        if len(self.trade_buffer) >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        df = pd.DataFrame(self.trade_buffer)
        logging.info(
            f"[{self.symbol.upper()}] Buffered {len(df)} trades. "
            f"Latest price: {df.iloc[-1]['price']}"
        )
        self.trade_buffer.clear()

    async def _ping_loop(self, ws):
        """Send a ping every 20 s to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(20)
                await ws.ping()
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass

    async def start(self):
        backoff = _BACKOFF_MIN

        while self._running:
            self._attempt += 1
            logging.info(
                f"[{self.symbol.upper()}] WS connect attempt #{self._attempt} "
                f"(backoff was {backoff}s)"
            )
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=None,   # we handle pings manually
                    close_timeout=5,
                ) as ws:
                    logging.info(f"[{self.symbol.upper()}] ✅ Connected to Binance Futures WS.")
                    backoff = _BACKOFF_MIN   # reset backoff on successful connect

                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for message in ws:
                            await self._handle_message(message)
                    finally:
                        ping_task.cancel()
                        await asyncio.gather(ping_task, return_exceptions=True)

            except websockets.ConnectionClosedOK:
                if not self._running:
                    logging.info(f"[{self.symbol.upper()}] 🛑 WS closed cleanly. Stopping.")
                    break
                logging.warning(
                    f"[{self.symbol.upper()}] WS closed normally — reconnecting in {backoff}s..."
                )

            except websockets.ConnectionClosedError as e:
                logging.warning(
                    f"[{self.symbol.upper()}] WS connection error ({e}) — reconnecting in {backoff}s..."
                )

            except OSError as e:
                logging.error(
                    f"[{self.symbol.upper()}] Network error ({e}) — reconnecting in {backoff}s..."
                )

            except Exception as e:
                logging.error(
                    f"[{self.symbol.upper()}] Unexpected WS error ({type(e).__name__}: {e}) "
                    f"— reconnecting in {backoff}s..."
                )

            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

        logging.info(f"[{self.symbol.upper()}] Collector stopped.")


if __name__ == "__main__":
    try:
        collector = BinanceDataCollector(symbol="btcusdt")
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        print("\nDisconnected by user.")
