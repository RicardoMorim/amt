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
logger = logging.getLogger(__name__)


def _to_naive_utc(ts) -> datetime:
    """
    A3 — Normalise any timestamp to a naive UTC datetime.
    Handles: pd.Timestamp (tz-aware or naive), datetime (tz-aware or naive), int (ms epoch).
    """
    if isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
    elif isinstance(ts, (int, float)):
        dt = datetime.utcfromtimestamp(ts / 1000)
    else:
        dt = ts
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class AMTSession:
    """
    Manages the state and heuristics for a SINGLE asset.

    Fixes vs previous version:
      A1 — working_history cached; pd.concat intra-candle only when _df_dirty.
      A3 — all timestamps normalised via _to_naive_utc().
      B5 — AMTPredictor loaded if model file exists; gates live alerts.
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
        self.alert       = alert_dispatcher or ConsoleAlert()
        self.arbitrator  = SignalArbitrator()
        self.ml_collector = MLDataCollector(look_forward_minutes=config.LOOK_FORWARD_MINUTES)

        # B5 — Load predictor if model exists (optional dependency)
        self.predictor = None
        try:
            import os
            if os.path.exists(config.ML_MODEL_PATH):
                from ml.predictor import AMTPredictor
                self.predictor = AMTPredictor(
                    model_path=config.ML_MODEL_PATH,
                    encoders_path=config.ML_ENCODERS_PATH,
                    confidence_threshold=config.ML_CONFIDENCE_THRESHOLD,
                )
                logger.info(f"[{self.symbol}] 🤖 ML predictor loaded (threshold={config.ML_CONFIDENCE_THRESHOLD:.0%})")
        except Exception as e:
            logger.warning(f"[{self.symbol}] ML predictor not available: {e}")

        # State
        self.current_candle_start    = None
        self._candle_trades_count    = 0
        self._candle_open:  float | None = None
        self._candle_high:  float | None = None
        self._candle_low:   float | None = None
        self._candle_close: float | None = None
        self._candle_volume: float = 0.0
        self._candle_delta:  float = 0.0
        self._candle_trades: list  = []

        self._candle_deque: deque       = deque(maxlen=100)
        self._df_cache: pd.DataFrame    = pd.DataFrame()
        self._df_dirty: bool            = False

        # A1 — working_history cache (includes intra-candle row)
        self._working_history_cache: pd.DataFrame | None = None
        self._working_history_dirty: bool = True

        self._current_session_date: str | None = None
        self._triggered_signals: set = set()

        if preload_history:
            asyncio.run(self._preload_history())

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _floor_to_timeframe(self, ts) -> datetime:
        dt  = _to_naive_utc(ts)
        tf  = self.candle_timeframe_seconds
        epoch = datetime(1970, 1, 1)
        sec   = (dt - epoch).total_seconds()
        return epoch + timedelta(seconds=(sec // tf) * tf)

    def _get_historical_df(self) -> pd.DataFrame:
        if self._df_dirty or self._df_cache.empty:
            if not self._candle_deque:
                return pd.DataFrame()
            self._df_cache = pd.DataFrame(list(self._candle_deque)).set_index('timestamp')
            self._df_dirty = False
            self._working_history_dirty = True   # invalidate working cache too
        return self._df_cache

    def _get_working_history(self, latest_candle: dict, is_closed: bool) -> pd.DataFrame:
        """
        A1 — Return working_history (historical + optional live intra-candle row).
        Rebuilds only when underlying data changed.
        """
        if is_closed:
            return self._get_historical_df()

        if not self._working_history_dirty and self._working_history_cache is not None:
            return self._working_history_cache

        historical_df = self._get_historical_df()
        if historical_df.empty:
            return historical_df

        new_row = pd.DataFrame([latest_candle]).set_index('timestamp')
        wh = pd.concat([historical_df, new_row])
        wh['cvd'] = wh['delta'].cumsum()
        self._working_history_cache  = wh
        self._working_history_dirty  = False
        return wh

    def _reset_candle_accumulators(self):
        self._candle_open   = None
        self._candle_high   = None
        self._candle_low    = None
        self._candle_close  = None
        self._candle_volume = 0.0
        self._candle_delta  = 0.0
        self._candle_trades_count = 0
        self._candle_trades.clear()
        self._working_history_dirty = True

    async def _preload_history(self):
        if self.source != 'binance':
            return
        logger.info(f"[{self.symbol}] ⏳ A carregar histórico da Binance...")
        interval = (
            f"{int(self.candle_timeframe_seconds / 60)}m"
            if self.candle_timeframe_seconds >= 60 else "1m"
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
                import requests
                loop = asyncio.get_event_loop()
                res  = await loop.run_in_executor(
                    None, lambda: requests.get(url, timeout=15).json()
                )

            for k in res:
                c = {
                    'timestamp': pd.to_datetime(k[0], unit='ms').tz_localize(None),
                    'open':   float(k[1]), 'high': float(k[2]),
                    'low':    float(k[3]), 'close': float(k[4]),
                    'volume': float(k[5]),
                    'delta':  float(k[9]) - (float(k[5]) - float(k[9])),
                }
                self._candle_deque.append(c)

            self._df_dirty = True
            df = self._get_historical_df()
            df['cvd'] = df['delta'].cumsum()
            for i, c in enumerate(self._candle_deque):
                c['cvd'] = float(df['cvd'].iloc[i])

            for _, c in df.iloc[-20:].iterrows():
                self.profile_mgr.update(price=c['close'], volume=c['volume'])

            logger.info(f"[{self.symbol}] ✅ Histórico carregado.")
        except Exception as e:
            logger.warning(f"[{self.symbol}] Falha ao pré-carregar histórico: {e}")

    # ── Core trade pipeline ───────────────────────────────────────────────────

    def on_trade(self, trade: dict):
        price  = float(trade['price'])
        volume = float(trade['volume'])
        side   = trade.get('side')

        self.profile_mgr.update(price=price, volume=volume)

        # A3 — normalise timestamp
        trade_dt   = _to_naive_utc(trade['timestamp'])
        trade_date = trade_dt.strftime('%Y-%m-%d')

        if self._current_session_date is None:
            self._current_session_date = trade_date
        elif trade_date != self._current_session_date:
            logger.info(
                f"[{self.symbol}] 🗓️ Nova sessão UTC ({trade_date}). A resetar volume profile..."
            )
            self.profile_mgr.reset()
            self._current_session_date = trade_date

        if self.current_candle_start is None:
            self.current_candle_start = self._floor_to_timeframe(trade_dt)

        time_elapsed = (trade_dt - self.current_candle_start).total_seconds()
        if time_elapsed >= self.candle_timeframe_seconds:
            self._close_candle()
            self.current_candle_start = self._floor_to_timeframe(trade_dt)

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
        self._candle_trades.append(trade)
        self._candle_trades_count += 1
        self._working_history_dirty = True   # intra-candle accumulator changed

        if self._candle_trades_count % 50 == 0:
            live_candle = self._build_candle(self.current_candle_start)
            if live_candle:
                self._analyze_market(live_candle, is_closed=False)

    def _build_candle(self, start_time) -> dict | None:
        if self._candle_open is None:
            return None
        historical_df = self._get_historical_df()
        last_cvd = (
            float(historical_df['cvd'].iloc[-1])
            if not historical_df.empty and 'cvd' in historical_df.columns
            else 0.0
        )
        return {
            'timestamp': start_time,
            'open':   self._candle_open,
            'high':   self._candle_high,
            'low':    self._candle_low,
            'close':  self._candle_close,
            'volume': self._candle_volume,
            'delta':  self._candle_delta,
            'cvd':    last_cvd + self._candle_delta,
        }

    def _close_candle(self):
        if self._candle_open is None:
            return
        candle = self._build_candle(self.current_candle_start)
        if candle is None:
            return

        self._candle_deque.append(candle)
        self._df_dirty = True

        logger.info(
            f"[{self.symbol}] 🔒 Vela Fechada [{candle['timestamp']}] "
            f"| Close: {candle['close']} | Vol: {candle['volume']:.2f} "
            f"| Delta: {candle['delta']:.2f}"
        )

        self._reset_candle_accumulators()

        df = self._get_historical_df()
        if not df.empty:
            df['cvd'] = df['delta'].cumsum()
            for i, c in enumerate(self._candle_deque):
                c['cvd'] = float(df['cvd'].iloc[i])
            self._df_dirty = True

        self._analyze_market(candle, is_closed=True)
        self._triggered_signals = set()

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _analyze_market(self, latest_candle: dict, is_closed: bool = False):
        historical_df = self._get_historical_df()
        if len(historical_df) < 10:
            if is_closed:
                logger.info(
                    f"[{self.symbol}] ⏳ Compilando histórico base... "
                    f"({len(historical_df)}/10 velas)"
                )
            return

        profile_data = self.profile_mgr.get_levels()  # cached unless dirty
        if not profile_data:
            return

        # A1 — use cached working_history, rebuilt only when dirty
        working_history = self._get_working_history(latest_candle, is_closed)
        cvd_data        = working_history[['cvd', 'delta']]

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

        recent_delta = working_history['delta'].iloc[-20:]
        del_std      = recent_delta.std()
        delta_zscore = (latest_candle['delta'] - recent_delta.mean()) / (del_std if del_std > 0 else 1)

        cvd_slope_short = (
            working_history['cvd'].iloc[-1] - working_history['cvd'].iloc[-5]
            if len(working_history) >= 5 else 0
        )
        cvd_slope_long = (
            working_history['cvd'].iloc[-1] - working_history['cvd'].iloc[-60]
            if len(working_history) >= 60 else 0
        )

        raw_ts = _to_naive_utc(latest_candle['timestamp'])
        candle_time_iso = raw_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        context = {
            'candle_time':         candle_time_iso,
            'timeframe_secs':      self.candle_timeframe_seconds,
            'asset':               self.symbol,
            'trigger_price':       trigger_price,
            'close_price':         trigger_price,
            'session_id':          raw_ts.strftime('%Y-%m-%d'),
            'session_state':       session_state,
            'vah': vah, 'val': val, 'poc': poc,
            'distance_to_poc':     distance_to_poc,
            'distance_to_poc_pct': round(distance_to_poc_pct, 5),
            'volume':              latest_candle['volume'],
            'volume_zscore':       round(volume_zscore, 2),
            'delta':               latest_candle['delta'],
            'delta_zscore':        round(delta_zscore, 2),
            'cvd_current':         latest_candle['cvd'],
            'cvd_slope_short':     cvd_slope_short,
            'cvd_slope_long':      cvd_slope_long,
        }

        raw_signals = []
        prices_list    = working_history['close'].tolist()[-4:-1]
        false_breakout = check_false_breakout(trigger_price, prices_list, profile_data)
        if false_breakout:
            raw_signals.append(false_breakout)

        breakout = detect_balance_breakout(
            latest_candle, cvd_data, profile_data, working_history.iloc[-20:-1]
        )
        if breakout:
            raw_signals.append(breakout)

        divergence = detect_cvd_divergence(
            working_history['close'], working_history['cvd']
        )
        if divergence:
            raw_signals.append(divergence)

        spike = detect_aggression_spike(working_history['delta'])
        if spike:
            raw_signals.append(spike)

        if not raw_signals:
            return

        try:
            final_signal, all_jsons = self.arbitrator.arbitrate(raw_signals, context)

            for sj in all_jsons:
                self.ml_collector.insert_signal(sj)
            if final_signal and final_signal.get('is_composite', False):
                self.ml_collector.insert_signal(final_signal)

            if final_signal and is_closed and final_signal['direction'] != 'CONFLICT':
                sig_key = f"{final_signal['signal_type']}_{final_signal['direction']}"
                if sig_key not in self._triggered_signals:
                    self._triggered_signals.add(sig_key)

                    # B5 — Gate alert behind ML predictor if model loaded
                    should_alert = True
                    ml_confidence = None
                    if self.predictor is not None:
                        decision = self.predictor.should_trade(final_signal)
                        ml_confidence = decision['confidence']
                        should_alert  = decision['action'] != 'SKIP'
                        if not should_alert:
                            logger.info(
                                f"[{self.symbol}] 🤖 Sinal FILTRADO pelo modelo ML "
                                f"({decision['skip_reason']})",
                                extra={'asset': self.symbol,
                                       'signal_type': final_signal.get('signal_type'),
                                       'confidence': ml_confidence}
                            )

                    if should_alert:
                        if ml_confidence is not None:
                            final_signal['ml_confidence'] = round(ml_confidence, 3)
                        logger.info(
                            f"[{self.symbol}] 🚨 SINAL APROVADO "
                            f"{final_signal['signal_type']} {final_signal['direction']} "
                            f"@ {trigger_price}"
                            f"{f' (ML: {ml_confidence:.0%})' if ml_confidence else ''}",
                            extra={'asset': self.symbol,
                                   'signal_type': final_signal.get('signal_type'),
                                   'confidence': ml_confidence or 1.0}
                        )
                        self.alert.send(final_signal)

            elif final_signal and final_signal.get('direction') == 'CONFLICT':
                logger.warning(
                    f"[{self.symbol}] ⚠️ Conflito de sinais na vela @ {trigger_price}"
                )

        except Exception as e:
            logger.error(f"[{self.symbol}] Falha na Arbitragem: {e}")

        if is_closed:
            self.ml_collector.update_labels(
                current_time_iso=context['candle_time'],
                current_price=trigger_price,
                history_df=working_history,
            )


class AMTEngineManager:
    def __init__(self):
        self.alert      = ConsoleAlert()
        self.sessions   = {}
        self.collectors = []

    def add_binance_asset(self, symbol, timeframe_sec=60, tick_size=0.5):
        session   = AMTSession(symbol, 'binance', timeframe_sec, self.alert, tick_size)
        collector = BinanceDataCollector(symbol=symbol, callback=session.on_trade)
        self.sessions[symbol] = session
        self.collectors.append(collector.start())

    async def start_all(self):
        logger.info("🚀 Arranque do Matrix: A iniciar Engine Multiativos...")
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
        loop.run_until_complete(manager.start_all())
    except KeyboardInterrupt:
        logger.info("\n🛑 Interrompido. A fechar...")
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
