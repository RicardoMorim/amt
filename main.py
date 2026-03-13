import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta

from data.collector_binance import BinanceDataCollector
from core.volume_profile import SessionProfileManager
from core.cvd import calculate_cvd
from core.market_state import identify_market_state, check_false_breakout
from signals.balance_breakout import detect_balance_breakout
from signals.volume_imbalance import detect_aggression_spike, detect_cvd_divergence
from alerts.console import ConsoleAlert

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AMTEngine:
    """
    The Main Brain.
    Ties the live data stream to the profile manager and checks for signals dynamically.
    """
    def __init__(self, symbol="btcusdt", candle_timeframe_seconds=60):
        # Tools
        self.profile_mgr = SessionProfileManager(tick_size=0.5, value_area_pct=0.68)
        self.alert = ConsoleAlert()
        self.collector = BinanceDataCollector(symbol=symbol, callback=self.on_trade)
        
        # State
        self.candle_timeframe_seconds = candle_timeframe_seconds
        self.current_candle_start = None
        self.candle_trades = []
        self.historical_candles = pd.DataFrame()
        
    def on_trade(self, trade):
        """ Callback fired on every single trade coming from WebSockets """
        
        # 1. Update Volume Profile
        self.profile_mgr.update(price=trade['price'], volume=trade['volume'])
        
        # 2. Add to current candle builder
        trade_time = trade['timestamp']
        
        if self.current_candle_start is None:
            self.current_candle_start = trade_time.replace(microsecond=0)
            logging.info(f"🟢 Primeira trade detetada. A iniciar construção da vela ({self.candle_timeframe_seconds}s) às {self.current_candle_start}...")
            
        time_elapsed = (trade_time - self.current_candle_start).total_seconds()
        
        if time_elapsed >= self.candle_timeframe_seconds:
            # Time to close the candle and run the heuristic engines!
            self._close_candle()
            self.current_candle_start = trade_time.replace(microsecond=0)
            
        self.candle_trades.append(trade)

    def _close_candle(self):
        """ Aggregates trades into a 1m/5m candle and runs AMT Heuristics """
        if not self.candle_trades:
            return
            
        df_trades = pd.DataFrame(self.candle_trades)
        
        # Calculate Delta directly from trades (accurate)
        df_trades['delta'] = df_trades.apply(lambda x: x['volume'] if x['side'] == 'buy' else -x['volume'], axis=1)
        candle_delta = df_trades['delta'].sum()
        
        # Build OHLCV Candle
        candle = {
            'timestamp': self.current_candle_start,
            'open': df_trades['price'].iloc[0],
            'high': df_trades['price'].max(),
            'low': df_trades['price'].min(),
            'close': df_trades['price'].iloc[-1],
            'volume': df_trades['volume'].sum(),
            'delta': candle_delta
        }
        
        # Append to History
        new_row = pd.DataFrame([candle]).set_index('timestamp')
        self.historical_candles = pd.concat([self.historical_candles, new_row])
        
        # Keep memory clean (keep last 100 candles max)
        if len(self.historical_candles) > 100:
            self.historical_candles = self.historical_candles.iloc[-100:]
            
        logging.info(f"🔒 Vela Fechada [{candle['timestamp']}] | Close: {candle['close']} | Vol: {candle['volume']:.2f} | Delta: {candle['delta']:.2f}")
        
        # Clear trades for the next candle
        self.candle_trades.clear()
        
        # recalculate CVD on history
        self.historical_candles['cvd'] = self.historical_candles['delta'].cumsum()
        
        # ============== RUN HEURISTICS ==============
        self._analyze_market()

    def _analyze_market(self):
        """ Evaluates current state against AMT Rules """
        if len(self.historical_candles) < 10:
            logging.info(f"⏳ A compilar histórico base... ({len(self.historical_candles)}/10 velas completas)")
            return # Need some history
            
        profile_data = self.profile_mgr.get_levels()
        if not profile_data:
            return
            
        latest_candle = self.historical_candles.iloc[-1].to_dict()
        cvd_data = self.historical_candles[['cvd', 'delta']]
        
        # Signal 1: False Breakout (Look Above and Fail)
        prices_list = self.historical_candles['close'].tolist()[-4:-1] # Prev 3 closes
        false_breakout = check_false_breakout(latest_candle['close'], prices_list, profile_data)
        if false_breakout:
            self.alert.send(
                signal_type=false_breakout['signal'], 
                direction=false_breakout['direction'], 
                trigger_price=latest_candle['close'],
                target_poc=profile_data['poc']
            )

        # Signal 2: Initiative Breakout with Volume Expansion
        breakout = detect_balance_breakout(latest_candle, cvd_data, profile_data, self.historical_candles.iloc[-20:-1])
        if breakout:
            self.alert.send(**breakout)
            
        # Signal 3: Exhaustion / Divergence
        divergence = detect_cvd_divergence(self.historical_candles['close'], self.historical_candles['cvd'])
        if divergence:
            self.alert.send(**divergence)
            
        # Signal 4: Aggression Spike
        spike = detect_aggression_spike(self.historical_candles['delta'])
        if spike:
            self.alert.send(trigger_price=latest_candle['close'], **spike)

    async def start(self):
        logging.info("Initializing AMT Engine...")
        try:
            await self.collector.start()
        except KeyboardInterrupt:
            logging.info("Shutting down engine.")

if __name__ == "__main__":
    engine = AMTEngine(symbol="btcusdt", candle_timeframe_seconds=60) # 1 Minute resolution for heuristics
    asyncio.run(engine.start())
