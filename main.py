import asyncio
import logging
import pandas as pd
from collections import deque
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
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AMTSession:
    """
    Manages the state and heuristics for a SINGLE asset.

    Performance improvements over original:
      - Rolling accumulators: on_trade() updates _candle_high/low/volume/delta
        incrementally — _build_candle() is O(1) instead of O(n).
      - deque for historical_candles: appending is O(1), no DataFrame copy on
        every candle close.  DataFrame conversion is lazy (only when needed).
      - Dirty flag cache on SessionProfileManager: get_levels() recomputes only
        when new ticks have arrived since the last call.
      - Daily profile reset: volume profile is cleared at UTC midnight so POC/VAH/VAL
        remain statistically meaningful.
      - Async preload: _preload_history uses httpx.AsyncClient instead of blocking
        requests.get(), avoiding event-loop stalls on startup.
    """

    def __init__(
        self,
        symbol,
        source,
        candle_timeframe_seconds=60,
        alert_dispatcher=None,
        tick_size=0.5,
        preload_history=True,
    ):
        self.symbol = symbol
        self.source = source
        self.candle_timeframe_seconds = candle_timeframe_seconds

        self.profile_mgr = SessionProfileManager(tick_size=tick_size, value_area_pct=0.68)
        self.alert = alert_dispatcher or ConsoleAlert()
        self.arbitrator = SignalArbitrator()
        self.ml_collector = MLDataCollector(look_forward_minutes=15)

        # State
        self.current_candle_start = None
        self._candle_trades_count = 0   # lightweight counter (no list)

        # Rolling candle accumulators — updated O(1) per trade
        self._candle_open:   float | None = None
        self._candle_high:   float | None = None
        self._candle_low:    float | None = None
        self._candle_close:  float | None = None
        self._candle_volume: float = 0.0
        self._candle_delta:  float = 0.0
        self._candle_trades: list = []   # kept only for legacy _build_candle compat

        # Lazy DataFrame: store candles as a deque of dicts, convert to DF on demand
        self._candle_deque: deque = deque(maxlen=100)
        self._df_cache: pd.DataFrame = pd.DataFrame()
        self._df_dirty: bool = False

        # Daily reset tracking (UTC date)
        self._current_session_date: str | None = None

        if preload_history:
            asyncio.get_event_loop().run_until_complete(self._preload_history())

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _floor_to_timeframe(self, ts):
        """Align a timestamp to the nearest timeframe grid boundary."""
        if isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
        else:
            dt = ts
        tf = self.candle_timeframe_seconds
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (dt - epoch).total_seconds()
        floored_seconds = (seconds_since_epoch // tf) * tf
        return epoch + timedelta(seconds=floored_seconds)

    def _get_historical_df(self) -> pd.DataFrame:
        """Return historical candles as a DataFrame, rebuilding only when dirty."""
        if self._df_dirty or self._df_cache.empty:
            if not self._candle_deque:
                return pd.DataFrame()
            self._df_cache = pd.DataFrame(list(self._candle_deque)).set_index('timestamp')
            self._df_dirty = False
        return self._df_cache

    def _reset_candle_accumulators(self):
        """Zero out rolling candle state."""
        self._candle_open   = None
        self._candle_high   = None
        self._candle_low    = None
        self._candle_close  = None
        self._candle_volume = 0.0
        self._candle_delta  = 0.0
        self._candle_trades_count = 0
        self._candle_trades.clear()

    async def _preload_history(self):
        """Fetch recent candles via async HTTP — no event-loop blocking."""
        if self.source != 'binance':
            return
        logging.info(f"[{self.symbol}] ⏳ A carregar histórico da Binance para arranque imediato...")
        interval = (
            f"{int(self.candle_timeframe_seconds / 60)}m"
            if self.candle_timeframe_seconds >= 60
            else "1m"
        )
        url = (
            f"https://fapi.binance.com/fapi/v1/klines"
            f"?symbol={self.symbol.upper()}&interval={interval}&limit=50"
        )
        try:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=15) as client:
                    res = (await client.get(url)).json()
            except ImportError:
                # Fallback to requests in a thread executor if httpx not installed
                import requests
                import asyncio as _asyncio
                loop = _asyncio.get_event_loop()
                res = await loop.run_in_executor(
                    None, lambda: requests.get(url, timeout=15).json()
                )

            preload = []
            for k in res:
                candle = {
                    'timestamp': pd.to_datetime(k[0], unit='ms').tz_localize(None),
                    'open':   float(k[1]),
                    'high':   float(k[2]),
                    'low':    float(k[3]),
                    'close':  float(k[4]),
                    'volume': float(k[5]),
                    'delta':  float(k[9]) - (float(k[5]) - float(k[9])),
                }
                preload.append(candle)

            for c in preload:
                self._candle_deque.append(c)

            self._df_dirty = True
            df = self._get_historical_df()
            df['cvd'] = df['delta'].cumsum()
            # Persist cvd back into the deque entries
            for i, c in enumerate(self._candle_deque):
                c['cvd'] = float(df['cvd'].iloc[i])
            self._df_dirty = True

            for _, c in df.iloc[-20:].iterrows():
                self.profile_mgr.update(price=c['close'], volume=c['volume'])

            logging.info(f"[{self.symbol}] ✅ Histórico carregado. Motor pronto a disparar sinais!")

        except Exception as e:
            logging.warning(f"[{self.symbol}] Falha ao pré-carregar histórico: {e}")

    # ── Core trade pipeline ────────────────────────────────────────────────────

    def on_trade(self, trade: dict):
        """Callback fired on every single trade (WebSocket or backfill)."""

        price  = float(trade['price'])
        volume = float(trade['volume'])
        side   = trade.get('side')

        # 1. Update Volume Profile (dirty flag set inside update())
        self.profile_mgr.update(price=price, volume=volume)

        # 2. Daily session reset at UTC midnight
        trade_time = trade['timestamp']
        if isinstance(trade_time, pd.Timestamp):
            trade_dt = trade_time.to_pydatetime()
        else:
            trade_dt = trade_time

        trade_date = trade_dt.strftime('%Y-%m-%d')
        if self._current_session_date is None:
            self._current_session_date = trade_date
        elif trade_date != self._current_session_date:
            logging.info(
                f"[{self.symbol}] 🗓️ Nova sessão UTC ({trade_date}). A resetar volume profile..."
            )
            self.profile_mgr.reset()
            self._current_session_date = trade_date

        # 3. Candle timing
        if self.current_candle_start is None:
            self.current_candle_start = self._floor_to_timeframe(trade_time)
            logging.info(
                f"[{self.symbol}] 🟢 A iniciar construção de vela "
                f"({self.candle_timeframe_seconds}s) às {self.current_candle_start}..."
            )

        time_elapsed = (trade_dt - self.current_candle_start).total_seconds()
        if time_elapsed >= self.candle_timeframe_seconds:
            self._close_candle()
            self.current_candle_start = self._floor_to_timeframe(trade_time)

        # 4. Update rolling accumulators — O(1), no list iteration
        if self._candle_open is None:
            self._candle_open = price
        self._candle_high   = price if self._candle_high is None else max(self._candle_high, price)
        self._candle_low    = price if self._candle_low  is None else min(self._candle_low,  price)
        self._candle_close  = price
        self._candle_volume += volume
        if side == 'buy':
            self._candle_delta += volume
        elif side == 'sell':
            self._candle_delta -= volume
        self._candle_trades.append(trade)   # kept for intra-candle analysis slice
        self._candle_trades_count += 1

        # 5. Intra-candle analysis every 50 trades
        if self._candle_trades_count % 50 == 0:
            live_candle = self._build_candle(self.current_candle_start)
            if live_candle:
                self._analyze_market(live_candle, is_closed=False)

    def _build_candle(self, start_time) -> dict | None:
        """Build candle dict from rolling accumulators — O(1)."""
        if self._candle_open is None:
            return None

        historical_df = self._get_historical_df()
        last_cvd = (
            float(historical_df['cvd'].iloc[-1])
            if not historical_df.empty and 'cvd' in historical_df.columns
            else 0.0
        )
        current_cvd = last_cvd + self._candle_delta

        return {
            'timestamp': start_time,
            'open':   self._candle_open,
            'high':   self._candle_high,
            'low':    self._candle_low,
            'close':  self._candle_close,
            'volume': self._candle_volume,
            'delta':  self._candle_delta,
            'cvd':    current_cvd,
        }

    def _close_candle(self):
        """Aggregate trades into a candle, persist to deque, run analysis."""
        if self._candle_open is None:
            return

        candle = self._build_candle(self.current_candle_start)
        if candle is None:
            return

        self._candle_deque.append(candle)
        self._df_dirty = True

        logging.info(
            f"[{self.symbol}] 🔒 Vela Fechada [{candle['timestamp']}] "
            f"| Close: {candle['close']} "
            f"| Vol: {candle['volume']:.2f} "
            f"| Delta: {candle['delta']:.2f}"
        )

        self._reset_candle_accumulators()

        # Recompute cumulative CVD across deque
        df = self._get_historical_df()
        if not df.empty:
            df['cvd'] = df['delta'].cumsum()
            for i, c in enumerate(self._candle_deque):
                c['cvd'] = float(df['cvd'].iloc[i])
            self._df_dirty = True

        self._analyze_market(candle, is_closed=True)
        self._triggered_signals = set()

    # ── Analysis ───────────────────────────────────────────────────────────────

    def _analyze_market(self, latest_candle: dict, is_closed: bool = False):
        """Evaluate current state against AMT rules."""
        historical_df = self._get_historical_df()

        if len(historical_df) < 10:
            if is_closed:
                logging.info(
                    f"[{self.symbol}] ⏳ A compilar histórico base... "
                    f"({len(historical_df)}/10 velas completas)"
                )
            return

        # get_levels() returns cached result if profile unchanged
        profile_data = self.profile_mgr.get_levels()
        if not profile_data:
            return

        if not is_closed:
            new_row = pd.DataFrame([latest_candle]).set_index('timestamp')
            working_history = pd.concat([historical_df, new_row])
            working_history['cvd'] = working_history['delta'].cumsum()
        else:
            working_history = historical_df

        cvd_data = working_history[['cvd', 'delta']]

        trigger_price = latest_candle['close']
        poc = profile_data['poc']
        vah = profile_data['vah']
        val = profile_data['val']

        session_state       = identify_market_state(trigger_price, profile_data)
        distance_to_poc     = trigger_price - poc
        distance_to_poc_pct = (distance_to_poc / poc) if poc > 0 else 0

        recent_vol    = working_history['volume'].iloc[-20:]
        vol_std       = recent_vol.std()
        volume_zscore = (latest_candle['volume'] - recent_vol.mean()) / (vol_std if vol_std > 0 else 1)

        recent_delta  = working_history['delta'].iloc[-20:]
        del_std       = recent_delta.std()
        delta_zscore  = (latest_candle['delta'] - recent_delta.mean()) / (del_std if del_std > 0 else 1)

        cvd_slope_short = (
            working_history['cvd'].iloc[-1] - working_history['cvd'].iloc[-5]
            if len(working_history) >= 5 else 0
        )
        cvd_slope_long = (
            working_history['cvd'].iloc[-1] - working_history['cvd'].iloc[-60]
            if len(working_history) >= 60 else 0
        )

        raw_ts = latest_candle['timestamp']
        if isinstance(raw_ts, pd.Timestamp):
            raw_ts = raw_ts.to_pydatetime()
        if hasattr(raw_ts, 'tzinfo') and raw_ts.tzinfo is not None:
            raw_ts = raw_ts.replace(tzinfo=None)
        candle_time_iso = raw_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        context = {
            "candle_time":         candle_time_iso,
            "timeframe_secs":      self.candle_timeframe_seconds,
            "asset":               self.symbol,
            "trigger_price":       trigger_price,
            "close_price":         trigger_price,
            "session_id":          raw_ts.strftime('%Y-%m-%d'),
            "session_state":       session_state,
            "vah":                 vah,
            "val":                 val,
            "poc":                 poc,
            "distance_to_poc":     distance_to_poc,
            "distance_to_poc_pct": round(distance_to_poc_pct, 5),
            "volume":              latest_candle['volume'],
            "volume_zscore":       round(volume_zscore, 2),
            "delta":               latest_candle['delta'],
            "delta_zscore":        round(delta_zscore, 2),
            "cvd_current":         latest_candle['cvd'],
            "cvd_slope_short":     cvd_slope_short,
            "cvd_slope_long":      cvd_slope_long,
        }

        raw_signals = []

        prices_list    = working_history['close'].tolist()[-4:-1]
        false_breakout = check_false_breakout(trigger_price, prices_list, profile_data)
        if false_breakout:
            raw_signals.append(false_breakout)

        breakout = detect_balance_breakout(latest_candle, cvd_data, profile_data, working_history.iloc[-20:-1])
        if breakout:
            raw_signals.append(breakout)

        divergence = detect_cvd_divergence(working_history['close'], working_history['cvd'])
        if divergence:
            raw_signals.append(divergence)

        spike = detect_aggression_spike(working_history['delta'])
        if spike:
            raw_signals.append(spike)

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
                        if sig_key not in getattr(self, '_triggered_signals', set()):
                            if not hasattr(self, '_triggered_signals'):
                                self._triggered_signals = set()
                            self._triggered_signals.add(sig_key)
                            self.alert.send(final_signal)
                    else:
                        logging.warning(
                            f"[{self.symbol}] ⚠️ Conflito de sinais evitado "
                            f"(Long vs Short) na mesma vela aos {trigger_price}"
                        )

            except Exception as e:
                logging.error(f"[{self.symbol}] Falha na Arbitragem de sinais: {e}")

        if is_closed:
            self.ml_collector.update_labels(
                current_time_iso=context['candle_time'],
                current_price=trigger_price,
                history_df=working_history,
            )


class AMTEngineManager:
    """
    Orchestrator that spins up multiple WebSockets and Sessions for different assets.
    """

    def __init__(self):
        self.alert = ConsoleAlert()
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
    manager.add_binance_asset(
        symbol=config.SYMBOL,
        timeframe_sec=config.TIMEFRAME_SECS,
        tick_size=config.TICK_SIZE,
    )

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
