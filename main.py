import asyncio
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta

from data.collector_binance import BinanceDataCollector
from data.collector_nq import AlpacaDataCollector
from core.volume_profile import SessionProfileManager
from core.cvd import calculate_cvd
from core.market_state import identify_market_state, check_false_breakout
from signals.balance_breakout import detect_balance_breakout
from signals.volume_imbalance import detect_aggression_spike, detect_cvd_divergence
from signals.arbitration import SignalArbitrator
from alerts.console import ConsoleAlert

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AMTSession:
    """
    Manages the state and heuristics for a SINGLE asset.
    """
    def __init__(self, symbol, source, candle_timeframe_seconds=60, alert_dispatcher=None, tick_size=0.5):
        self.symbol = symbol
        self.source = source # 'binance' ou 'alpaca'
        self.candle_timeframe_seconds = candle_timeframe_seconds
        
        # Tools
        self.profile_mgr = SessionProfileManager(tick_size=tick_size, value_area_pct=0.68)
        self.alert = alert_dispatcher or ConsoleAlert()
        self.arbitrator = SignalArbitrator()
        
        # State
        self.current_candle_start = None
        self.candle_trades = []
        self.historical_candles = pd.DataFrame()
        
        # Preload history if possible
        self._preload_history()
        
    def _preload_history(self):
        """ Fetch recent candles from REST API so we don't have to wait 10 minutes """
        if self.source == 'binance':
            logging.info(f"[{self.symbol}] ⏳ A carregar histórico da Binance para arranque imediato...")
            # Convert timeframe to binance format (e.g. 60s = '1m', 300s = '5m')
            interval = f"{int(self.candle_timeframe_seconds/60)}m" if self.candle_timeframe_seconds >= 60 else "1m"
            
            try:
                # Fetch last 30 candles
                url = f"https://fapi.binance.com/fapi/v1/klines?symbol={self.symbol.upper()}&interval={interval}&limit=50"
                res = requests.get(url).json()
                
                preload = []
                for k in res:
                    # [Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base volume, Taker buy quote volume, Ignore]
                    candle = {
                        'timestamp': pd.to_datetime(k[0], unit='ms'),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5]),
                        # Approximation of Delta for historic klines: (Taker Buy Volume - Taker Sell Volume)
                        # Where Taker Sell Vol = Total Vol - Taker Buy Vol
                        'delta': float(k[9]) - (float(k[5]) - float(k[9]))
                    }
                    preload.append(candle)
                
                self.historical_candles = pd.DataFrame(preload).set_index('timestamp')
                self.historical_candles['cvd'] = self.historical_candles['delta'].cumsum()
                
                # Preload the Profile Manager with the last 20 candles of volume at the POC/Close
                # (Rough approximation for the session profile just to jumpstart it)
                for _, c in self.historical_candles.iloc[-20:].iterrows():
                    self.profile_mgr.update(price=c['close'], volume=c['volume'])
                    
                logging.info(f"[{self.symbol}] ✅ Histórico carregado. Motor pronto a disparar sinais!")
                
            except Exception as e:
                logging.warning(f"[{self.symbol}] Falha ao pré-carregar histórico: {e}")
        elif self.source == 'alpaca':
            logging.info(f"[{self.symbol}] ⏳ A carregar histórico da Alpaca para arranque imediato...")
            # Alpaca API for historical data (requires API keys and specific endpoint)
            # This is a placeholder. You would typically use Alpaca's SDK or direct REST calls.
            # Example:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.historical import StockHistoricalDataClient
            
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            request_params = StockBarsRequest(
                symbol_or_symbols=[self.symbol],
                timeframe=TimeFrame.Minute, # Adjust based on self.candle_timeframe_seconds
                start=datetime.now() - timedelta(days=1),
                end=datetime.now()
            )
            bars = client.get_stock_bars(request_params)
            
            preload = []
            for bar in bars.data[self.symbol]:
                candle = {
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'delta': 0 # Alpaca historical bars don't provide delta directly, needs approximation
                }
                preload.append(candle)
            
            if preload:
                self.historical_candles = pd.DataFrame(preload).set_index('timestamp')
                self.historical_candles['cvd'] = self.historical_candles['delta'].cumsum()
                for _, c in self.historical_candles.iloc[-20:].iterrows():
                    self.profile_mgr.update(price=c['close'], volume=c['volume'])
                logging.info(f"[{self.symbol}] ✅ Histórico carregado. Motor pronto a disparar sinais!")
            else:
                logging.warning(f"[{self.symbol}] Falha ao pré-carregar histórico da Alpaca: Sem dados.")
            logging.warning(f"[{self.symbol}] Pré-carregamento de histórico da Alpaca não implementado. A iniciar sem histórico.")


    def on_trade(self, trade):
        """ Callback fired on every single trade coming from WebSockets """
        
        # 1. Update Volume Profile
        self.profile_mgr.update(price=trade['price'], volume=trade['volume'])
        
        # 2. Add to current candle builder
        trade_time = trade['timestamp']
        
        if self.current_candle_start is None:
            self.current_candle_start = trade_time.replace(microsecond=0)
            logging.info(f"[{self.symbol}] 🟢 A iniciar construção de vela ({self.candle_timeframe_seconds}s) às {self.current_candle_start}...")
            
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
        if 'side' in df_trades.columns and df_trades['side'].notna().all():
            df_trades['delta'] = df_trades.apply(lambda x: x['volume'] if x['side'] == 'buy' else -x['volume'], axis=1)
        else:
            # Fallback (Tick Test) for Alpaca/Stocks
            price_diff = df_trades['price'].diff()
            direction = price_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).replace(0, pd.NA).ffill().fillna(1)
            df_trades['delta'] = df_trades['volume'] * direction
            
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
            
        logging.info(f"[{self.symbol}] 🔒 Vela Fechada [{candle['timestamp']}] | Close: {candle['close']} | Vol: {candle['volume']:.2f} | Delta: {candle['delta']:.2f}")
        
        # Clear trades for the next candle
        self.candle_trades.clear()
        
        # recalculate CVD on history
        self.historical_candles['cvd'] = self.historical_candles['delta'].cumsum()
        
        # ============== RUN HEURISTICS ==============
        self._analyze_market()

    def _analyze_market(self):
        """ Evaluates current state against AMT Rules """
        if len(self.historical_candles) < 10:
            logging.info(f"[{self.symbol}] ⏳ A compilar histórico base... ({len(self.historical_candles)}/10 velas completas)")
            return # Need some history
            
        profile_data = self.profile_mgr.get_levels()
        if not profile_data:
            return
            
        latest_candle = self.historical_candles.iloc[-1].to_dict()
        cvd_data = self.historical_candles[['cvd', 'delta']]
        
        # -- ML Context & Feature Generation --
        trigger_price = latest_candle['close']
        poc = profile_data['poc']
        vah = profile_data['vah']
        val = profile_data['val']
        
        session_state = identify_market_state(trigger_price, profile_data)
        distance_to_poc = trigger_price - poc
        distance_to_poc_pct = (distance_to_poc / poc) if poc > 0 else 0
        
        recent_vol = self.historical_candles['volume'].iloc[-21:-1]
        vol_std = recent_vol.std()
        volume_zscore = (latest_candle['volume'] - recent_vol.mean()) / (vol_std if vol_std > 0 else 1)
        
        recent_delta = self.historical_candles['delta'].iloc[-21:-1]
        del_std = recent_delta.std()
        delta_zscore = (latest_candle['delta'] - recent_delta.mean()) / (del_std if del_std > 0 else 1)
        
        cvd_slope_short = self.historical_candles['cvd'].iloc[-1] - self.historical_candles['cvd'].iloc[-5] if len(self.historical_candles) >= 5 else 0
        cvd_slope_long = self.historical_candles['cvd'].iloc[-1] - self.historical_candles['cvd'].iloc[-60] if len(self.historical_candles) >= 60 else 0
        
        context = {
            "candle_time": latest_candle['timestamp'].isoformat() + "Z",
            "timeframe_secs": self.candle_timeframe_seconds,
            "asset": self.symbol,
            "trigger_price": trigger_price,
            "close_price": trigger_price,
            "session_id": latest_candle['timestamp'].strftime('%Y-%m-%d'),
            "session_state": session_state,
            "vah": vah,
            "val": val,
            "poc": poc,
            "distance_to_poc": distance_to_poc,
            "distance_to_poc_pct": round(distance_to_poc_pct, 5),
            "volume": latest_candle['volume'],
            "volume_zscore": round(volume_zscore, 2),
            "delta": latest_candle['delta'],
            "delta_zscore": round(delta_zscore, 2),
            "cvd_current": latest_candle['cvd'],
            "cvd_slope_short": cvd_slope_short,
            "cvd_slope_long": cvd_slope_long,
        }
        
        # -- Raw Signal Generation --
        raw_signals = []

        prices_list = self.historical_candles['close'].tolist()[-4:-1]
        false_breakout = check_false_breakout(trigger_price, prices_list, profile_data)
        if false_breakout: raw_signals.append(false_breakout)

        breakout = detect_balance_breakout(latest_candle, cvd_data, profile_data, self.historical_candles.iloc[-20:-1])
        if breakout: raw_signals.append(breakout)
            
        divergence = detect_cvd_divergence(self.historical_candles['close'], self.historical_candles['cvd'])
        if divergence: raw_signals.append(divergence)
            
        spike = detect_aggression_spike(self.historical_candles['delta'])
        if spike: raw_signals.append(spike)
        
        # -- Signal Arbitration & Dispatch --
        if raw_signals:
            try:
                final_signal, all_jsons = self.arbitrator.arbitrate(raw_signals, context)
                
                # Phase 2 ML: Here we have "all_jsons" fully structured. We could save them to .parquet/.csv
                
                if final_signal:
                    if final_signal['direction'] != 'CONFLICT':
                        # Send clear signals to Discord/Telegram/Console
                        self.alert.send(final_signal)
                    else:
                        # Log conflicts silently. E.g., Don't short if there's a huge DELTA_SPIKE long on the same candle!
                        logging.warning(f"[{self.symbol}] ⚠️ Conflito de sinais evitado (Long vs Short) na mesma vela aos {trigger_price}")
                        
            except Exception as e:
                logging.error(f"[{self.symbol}] Falha na Arbitragem de sinais: {e}")

class AMTEngineManager:
    """
    Orchestrator that spins up multiple WebSockets and Sessions for different assets concurrently.
    """
    def __init__(self,):
        self.alert = ConsoleAlert()
        self.sessions = {}
        self.collectors = []
        
    def add_binance_asset(self, symbol, timeframe_sec=60, tick_size=0.5):
        session = AMTSession(symbol, 'binance', timeframe_sec, self.alert, tick_size)
        self.sessions[symbol] = session
        collector = BinanceDataCollector(symbol=symbol, callback=session.on_trade)
        self.collectors.append(collector.start())
        
    def add_alpaca_asset(self, symbol, timeframe_sec=60, tick_size=0.25):
        session = AMTSession(symbol, 'alpaca', timeframe_sec, self.alert, tick_size)
        self.sessions[symbol] = session
        collector = AlpacaDataCollector(symbols=[symbol], callback=session.on_trade)
        # Wrap Alpaca's synchronous start in an asyncio thread to not block Binance
        self.collectors.append(asyncio.to_thread(collector.start))

    async def start_all(self):
        logging.info("🚀 Arranque do Matrix: A iniciar Engine Multiativos...")
        await asyncio.gather(*self.collectors)

if __name__ == "__main__":
    manager = AMTEngineManager()
    
    # Adicionar Bitcoin
    manager.add_binance_asset(symbol="btcusdt", timeframe_sec=60, tick_size=0.1)
    
    # Adicionar Ethereum
    manager.add_binance_asset(symbol="ethusdt", timeframe_sec=60, tick_size=0.01)
    
    # Adicionar Nasdaq (via ETF QQQ no Alpaca - Cuidado: Requer chaves de API no .env)
    # manager.add_alpaca_asset(symbol="QQQ", timeframe_sec=60, tick_size=0.01)
    
    try:
        asyncio.run(manager.start_all())
    except KeyboardInterrupt:
        logging.info("Pára motores.")
