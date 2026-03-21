import asyncio
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

from data.collector_binance import BinanceDataCollector
from core.volume_profile import SessionProfileManager
from core.cvd import calculate_cvd
from core.market_state import identify_market_state, check_false_breakout
from signals.balance_breakout import detect_balance_breakout
from signals.volume_imbalance import detect_aggression_spike, detect_cvd_divergence
from signals.arbitration import SignalArbitrator
from alerts.console import ConsoleAlert
from data.ml_collector import MLDataCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AMTSession:
    """
    Manages the state and heuristics for a SINGLE asset.
    """
    def __init__(
        self,
        symbol,
        source,
        candle_timeframe_seconds=60,
        alert_dispatcher=None,
        tick_size=0.5,
        preload_history=True,          # ← NOVO PARÂMETRO
    ):
        self.symbol = symbol
        self.source = source
        self.candle_timeframe_seconds = candle_timeframe_seconds

        # Tools
        self.profile_mgr = SessionProfileManager(tick_size=tick_size, value_area_pct=0.68)
        self.alert = alert_dispatcher or ConsoleAlert()
        self.arbitrator = SignalArbitrator()
        self.ml_collector = MLDataCollector(look_forward_minutes=15)

        # State
        self.current_candle_start = None
        self.candle_trades = []
        self.historical_candles = pd.DataFrame()
        self.triggered_signals = set()

        # ← SÓ faz preload no modo live
        if preload_history:
            self._preload_history()

    def _floor_to_timeframe(self, ts):
        """Alinha um timestamp ao início do período do timeframe (ex: 14:07:23 → 14:00:00 num TF de 900s)."""
        if isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
        else:
            dt = ts
        tf = self.candle_timeframe_seconds
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (dt - epoch).total_seconds()
        floored_seconds = (seconds_since_epoch // tf) * tf
        return epoch + timedelta(seconds=floored_seconds)

    def _preload_history(self):
        """ Fetch recent candles from REST API so we don't have to wait 10 minutes """
        if self.source == 'binance':
            logging.info(f"[{self.symbol}] ⏳ A carregar histórico da Binance para arranque imediato...")
            interval = f"{int(self.candle_timeframe_seconds/60)}m" if self.candle_timeframe_seconds >= 60 else "1m"

            try:
                url = f"https://fapi.binance.com/fapi/v1/klines?symbol={self.symbol.upper()}&interval={interval}&limit=50"
                res = requests.get(url).json()

                preload = []
                for k in res:
                    candle = {
                        'timestamp': pd.to_datetime(k[0], unit='ms').tz_localize(None),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5]),
                        'delta': float(k[9]) - (float(k[5]) - float(k[9]))
                    }
                    preload.append(candle)

                self.historical_candles = pd.DataFrame(preload).set_index('timestamp')
                self.historical_candles['cvd'] = self.historical_candles['delta'].cumsum()

                for _, c in self.historical_candles.iloc[-20:].iterrows():
                    self.profile_mgr.update(price=c['close'], volume=c['volume'])

                logging.info(f"[{self.symbol}] ✅ Histórico carregado. Motor pronto a disparar sinais!")

            except Exception as e:
                logging.warning(f"[{self.symbol}] Falha ao pré-carregar histórico: {e}")

    def on_trade(self, trade):
        """ Callback fired on every single trade coming from WebSockets """

        # 1. Update Volume Profile
        self.profile_mgr.update(price=trade['price'], volume=trade['volume'])

        # 2. Add to current candle builder
        trade_time = trade['timestamp']

        if self.current_candle_start is None:
            # ← FIX: alinha ao grid do timeframe em vez de só remover microssegundos
            self.current_candle_start = self._floor_to_timeframe(trade_time)
            logging.info(f"[{self.symbol}] 🟢 A iniciar construção de vela ({self.candle_timeframe_seconds}s) às {self.current_candle_start}...")

        time_elapsed = (trade_time - self.current_candle_start).total_seconds()

        if time_elapsed >= self.candle_timeframe_seconds:
            self._close_candle()
            # ← FIX: alinha ao grid do timeframe
            self.current_candle_start = self._floor_to_timeframe(trade_time)

        self.candle_trades.append(trade)

        # Real-time intra-candle analysis (every 50 trades to save CPU)
        if len(self.candle_trades) > 0 and len(self.candle_trades) % 50 == 0:
            live_candle = self._build_candle(self.candle_trades, self.current_candle_start)
            self._analyze_market(live_candle, is_closed=False)

    def _build_candle(self, trades, start_time):
        if not trades: return None

        open_price  = trades[0]['price']
        high_price  = max(t['price'] for t in trades)
        low_price   = min(t['price'] for t in trades)
        close_price = trades[-1]['price']
        volume      = sum(t['volume'] for t in trades)

        if 'side' in trades[0] and trades[0]['side'] is not None:
            candle_delta = sum(t['volume'] if t['side'] == 'buy' else -t['volume'] for t in trades)
        else:
            candle_delta = 0
            last_price = open_price
            last_direction = 1
            for t in trades:
                if t['price'] > last_price:
                    last_direction = 1
                elif t['price'] < last_price:
                    last_direction = -1
                candle_delta += t['volume'] * last_direction
                last_price = t['price']

        last_cvd = self.historical_candles['cvd'].iloc[-1] if not self.historical_candles.empty and 'cvd' in self.historical_candles.columns else 0
        current_cvd = last_cvd + candle_delta

        return {
            'timestamp': start_time,
            'open':   open_price,
            'high':   high_price,
            'low':    low_price,
            'close':  close_price,
            'volume': volume,
            'delta':  candle_delta,
            'cvd':    current_cvd
        }

    def _close_candle(self):
        """ Aggregates trades into a candle and runs AMT Heuristics """
        if not self.candle_trades:
            return

        candle = self._build_candle(self.candle_trades, self.current_candle_start)

        new_row = pd.DataFrame([candle]).set_index('timestamp')
        self.historical_candles = pd.concat([self.historical_candles, new_row])

        if len(self.historical_candles) > 100:
            self.historical_candles = self.historical_candles.iloc[-100:]

        logging.info(f"[{self.symbol}] 🔒 Vela Fechada [{candle['timestamp']}] | Close: {candle['close']} | Vol: {candle['volume']:.2f} | Delta: {candle['delta']:.2f}")

        self.candle_trades.clear()
        self.historical_candles['cvd'] = self.historical_candles['delta'].cumsum()

        self._analyze_market(candle, is_closed=True)
        self.triggered_signals.clear()

    def _analyze_market(self, latest_candle, is_closed=False):
        """ Evaluates current state against AMT Rules """
        if len(self.historical_candles) < 10:
            if is_closed:
                logging.info(f"[{self.symbol}] ⏳ A compilar histórico base... ({len(self.historical_candles)}/10 velas completas)")
            return

        profile_data = self.profile_mgr.get_levels()
        if not profile_data:
            return

        if not is_closed:
            new_row = pd.DataFrame([latest_candle]).set_index('timestamp')
            working_history = pd.concat([self.historical_candles, new_row])
            working_history['cvd'] = working_history['delta'].cumsum()
        else:
            working_history = self.historical_candles

        cvd_data = working_history[['cvd', 'delta']]

        trigger_price = latest_candle['close']
        poc = profile_data['poc']
        vah = profile_data['vah']
        val = profile_data['val']

        session_state       = identify_market_state(trigger_price, profile_data)
        distance_to_poc     = trigger_price - poc
        distance_to_poc_pct = (distance_to_poc / poc) if poc > 0 else 0

        recent_vol   = self.historical_candles['volume'].iloc[-20:]
        vol_std      = recent_vol.std()
        volume_zscore = (latest_candle['volume'] - recent_vol.mean()) / (vol_std if vol_std > 0 else 1)

        recent_delta  = self.historical_candles['delta'].iloc[-20:]
        del_std       = recent_delta.std()
        delta_zscore  = (latest_candle['delta'] - recent_delta.mean()) / (del_std if del_std > 0 else 1)

        cvd_slope_short = working_history['cvd'].iloc[-1] - working_history['cvd'].iloc[-5]  if len(working_history) >= 5  else 0
        cvd_slope_long  = working_history['cvd'].iloc[-1] - working_history['cvd'].iloc[-60] if len(working_history) >= 60 else 0

        # ── FIX: usa o timestamp DA VELA (histórico correto), não datetime.now() ──
        raw_ts = latest_candle['timestamp']
        if isinstance(raw_ts, pd.Timestamp):
            raw_ts = raw_ts.to_pydatetime()
        if hasattr(raw_ts, 'tzinfo') and raw_ts.tzinfo is not None:
            raw_ts = raw_ts.replace(tzinfo=None)
        # Formata como ISO 8601 com Z (UTC)
        candle_time_iso = raw_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        context = {
            "candle_time":          candle_time_iso,   # ← timestamp histórico correto
            "timeframe_secs":       self.candle_timeframe_seconds,
            "asset":                self.symbol,
            "trigger_price":        trigger_price,
            "close_price":          trigger_price,
            "session_id":           raw_ts.strftime('%Y-%m-%d'),
            "session_state":        session_state,
            "vah":                  vah,
            "val":                  val,
            "poc":                  poc,
            "distance_to_poc":      distance_to_poc,
            "distance_to_poc_pct":  round(distance_to_poc_pct, 5),
            "volume":               latest_candle['volume'],
            "volume_zscore":        round(volume_zscore, 2),
            "delta":                latest_candle['delta'],
            "delta_zscore":         round(delta_zscore, 2),
            "cvd_current":          latest_candle['cvd'],
            "cvd_slope_short":      cvd_slope_short,
            "cvd_slope_long":       cvd_slope_long,
        }

        raw_signals = []

        prices_list    = working_history['close'].tolist()[-4:-1]
        false_breakout = check_false_breakout(trigger_price, prices_list, profile_data)
        if false_breakout: raw_signals.append(false_breakout)

        breakout = detect_balance_breakout(latest_candle, cvd_data, profile_data, working_history.iloc[-20:-1])
        if breakout: raw_signals.append(breakout)

        divergence = detect_cvd_divergence(working_history['close'], working_history['cvd'])
        if divergence: raw_signals.append(divergence)

        spike = detect_aggression_spike(working_history['delta'])
        if spike: raw_signals.append(spike)

        if raw_signals:
            try:
                final_signal, all_jsons = self.arbitrator.arbitrate(raw_signals, context)

                for sj in all_jsons:
                    self.ml_collector.insert_signal(sj)
                if final_signal and final_signal.get('is_composite', False):
                    self.ml_collector.insert_signal(final_signal)

                if final_signal and is_closed:
                    if final_signal['direction'] != 'CONFLICT':
                        sig_key = f"{final_signal['signal_type']}_{final_signal['direction']}"
                        if sig_key not in self.triggered_signals:
                            self.triggered_signals.add(sig_key)
                            self.alert.send(final_signal)
                    else:
                        logging.warning(f"[{self.symbol}] ⚠️ Conflito de sinais evitado (Long vs Short) na mesma vela aos {trigger_price}")

            except Exception as e:
                logging.error(f"[{self.symbol}] Falha na Arbitragem de sinais: {e}")

        if is_closed:
            self.ml_collector.update_labels(
                current_time_iso=context['candle_time'],
                current_price=trigger_price,
                history_df=working_history
            )


class AMTEngineManager:
    """
    Orchestrator that spins up multiple WebSockets and Sessions for different assets concurrently.
    """
    def __init__(self):
        self.alert    = ConsoleAlert()
        self.sessions = {}
        self.collectors = []

    def add_binance_asset(self, symbol, timeframe_sec=60, tick_size=0.5):
        session = AMTSession(symbol, 'binance', timeframe_sec, self.alert, tick_size)
        self.sessions[symbol] = session
        collector = BinanceDataCollector(symbol=symbol, callback=session.on_trade)
        self.collectors.append(collector.start())

    async def start_all(self):
        logging.info("🚀 Arranque do Matrix: A iniciar Engine Multiativos...")
        try:
            await asyncio.gather(*self.collectors)
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    manager = AMTEngineManager()
    manager.add_binance_asset(symbol="btcusdt", timeframe_sec=900, tick_size=0.1)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        main_task = loop.create_task(manager.start_all())
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logging.info("\n🛑 Sinal de interrupção recebido. A fechar conexões...")
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.run_until_complete(asyncio.sleep(0.1))
        logging.info("Pára motores.")
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
