import sqlite3
import logging
import pandas as pd
from datetime import datetime, timedelta


class MLDataCollector:
    """
    Collects signals from the AMT Engine and saves them to SQLite.

    Performance:
      - Signals buffered in memory and flushed in batches.
      - Bulk end-of-day labeling via label_all_pending().

    Thread-safety (A2 fix):
      If an external_conn is provided (e.g. from ThreadSafeDBManager in
      historical_runner), all writes go through that connection instead of
      opening a second, uncoordinated one.
    """

    def __init__(
        self,
        db_path: str = "amt_ml_dataset.db",
        look_forward_minutes: int = 15,
        flush_every: int = 200,
        external_conn=None,          # ← A2: shared connection from historical_runner
    ):
        self.db_path = db_path
        self.look_forward_minutes = look_forward_minutes
        self.flush_every = flush_every
        self._buffer = []
        self._owns_connection = external_conn is None

        if external_conn is not None:
            self.conn = external_conn
        else:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")

        self._create_tables()

    def _create_tables(self):
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id                   TEXT PRIMARY KEY,
            timestamp_event      TEXT,
            asset                TEXT,
            timeframe_secs       INTEGER,
            signal_type          TEXT,
            direction            TEXT,
            is_composite         BOOLEAN,
            trigger_price        REAL,
            session_state        TEXT,
            distance_to_poc_pct  REAL,
            volume_zscore        REAL,
            delta_zscore         REAL,
            cvd_slope_short      REAL,
            cvd_slope_long       REAL,
            label_max_fwd_price  REAL,
            label_min_fwd_price  REAL,
            label_win_pct        REAL,
            label_loss_pct       REAL,
            is_labeled           BOOLEAN DEFAULT 0
        )
        ''')
        self.conn.commit()

    def insert_signal(self, signal: dict):
        try:
            self._buffer.append((
                str(signal.get('id', 'unknown')),
                str(signal.get('timestamp_event')),
                str(signal.get('asset', 'unknown')),
                int(signal.get('timeframe_secs', 0)),
                str(signal.get('signal_type', 'unknown')),
                str(signal.get('direction', 'unknown')),
                bool(signal.get('is_composite', False)),
                float(signal.get('trigger_price', 0.0)),
                str(signal.get('session_state', 'unknown')),
                float(signal.get('distance_to_poc_pct', 0.0)),
                float(signal.get('volume_zscore', 0.0)),
                float(signal.get('delta_zscore', 0.0)),
                float(signal.get('cvd_slope_short', 0.0)),
                float(signal.get('cvd_slope_long', 0.0)),
            ))
        except Exception as e:
            logging.warning(f"[MLDataCollector] Failed to buffer signal: {e}")
            return

        if len(self._buffer) >= self.flush_every:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self._buffer:
            return
        try:
            self.conn.executemany('''
            INSERT OR IGNORE INTO signals (
                id, timestamp_event, asset, timeframe_secs, signal_type, direction,
                is_composite, trigger_price, session_state, distance_to_poc_pct,
                volume_zscore, delta_zscore, cvd_slope_short, cvd_slope_long
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', self._buffer)
            self.conn.commit()
            self._buffer.clear()
        except Exception as e:
            logging.error(f"[MLDataCollector] Flush error: {e}")

    def label_all_pending(self, history_df: pd.DataFrame):
        self._flush_buffer()

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, timestamp_event, trigger_price, direction FROM signals WHERE is_labeled = 0"
        )
        unlabeled = cursor.fetchall()

        if not unlabeled or history_df.empty:
            return

        updates = []
        for sig_id, sig_time_str, trigger_price, direction in unlabeled:
            try:
                clean  = sig_time_str.replace('Z', '').replace('+00:00', '')
                sig_dt = datetime.fromisoformat(clean)
            except Exception:
                continue

            target_end = sig_dt + timedelta(minutes=self.look_forward_minutes)

            try:
                idx = history_df.index
                naive_idx = idx.tz_localize(None) if hasattr(idx, 'tz') and idx.tz else idx
                mask = (naive_idx > sig_dt) & (naive_idx <= target_end)
                forward_window = history_df.loc[mask]
            except Exception:
                continue

            if forward_window.empty:
                max_p = float(trigger_price)
                min_p = float(trigger_price)
            else:
                max_p = float(forward_window['high'].max())
                min_p = float(forward_window['low'].min())

            tp = float(trigger_price)
            if direction == 'LONG':
                win_pct  = (max_p - tp) / tp
                loss_pct = (min_p - tp) / tp
            elif direction == 'SHORT':
                win_pct  = (tp - min_p) / tp
                loss_pct = (tp - max_p) / tp
            else:
                win_pct = loss_pct = 0.0

            updates.append((max_p, min_p, win_pct, loss_pct, sig_id))

        if updates:
            self.conn.executemany('''
            UPDATE signals
            SET label_max_fwd_price=?, label_min_fwd_price=?,
                label_win_pct=?, label_loss_pct=?, is_labeled=1
            WHERE id=?
            ''', updates)
            self.conn.commit()
            logging.info(f"[MLDataCollector] 🏷️ Labeled {len(updates)} signals in bulk.")

    def update_labels(self, current_time_iso: str, current_price: float, history_df: pd.DataFrame):
        """Legacy no-op kept for live-bot compatibility."""
        pass

    def close(self):
        self._flush_buffer()
        if self._owns_connection:
            self.conn.close()
