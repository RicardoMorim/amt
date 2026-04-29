import sqlite3
import pandas as pd
import ccxt
from datetime import datetime, timezone, timedelta
from pathlib import Path
import config

# Configuração
DB_PATH = config.DB_PATH
SYMBOL = config.SYMBOL.upper()
TIMEFRAME = "15m" # Match 900s
TIMEFRAME_SECS = config.TIMEFRAME_SECS

def backfill():
    print(f"START: Backfilling candles for {SYMBOL} ({TIMEFRAME})...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Criar tabela se não existir
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            timeframe_secs INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, timeframe_secs, timestamp)
        )
    """)

    start_date = datetime.strptime(config.BACKFILL_START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(config.BACKFILL_END, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    current_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    total_inserted = 0
    
    print(f"Downloading from {start_date} to {end_date}...")

    while current_ms < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=current_ms, limit=1000)
            if not ohlcv:
                break
                
            rows = []
            for candle in ohlcv:
                ts_ms = candle[0]
                if ts_ms > end_ms:
                    continue
                
                ts_iso = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
                rows.append((
                    config.SYMBOL.lower(),
                    TIMEFRAME_SECS,
                    ts_iso,
                    candle[1], # O
                    candle[2], # H
                    candle[3], # L
                    candle[4], # C
                    candle[5]  # V
                ))
            
            cursor.executemany("""
                INSERT OR IGNORE INTO candles 
                (symbol, timeframe_secs, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            
            conn.commit()
            total_inserted += len(rows)
            
            # Avançar o ponteiro
            last_ms = ohlcv[-1][0]
            if last_ms <= current_ms: # Evitar loop infinito se não houver progresso
                break
            current_ms = last_ms + 1
            
            print(f"   [OK] {datetime.fromtimestamp(current_ms/1000, tz=timezone.utc).date()} | Total: {total_inserted} candles")
            
        except Exception as e:
            print(f"   [ERROR]: {e}")
            break

    conn.close()
    print(f"\nDONE! {total_inserted} candles inserted in {DB_PATH}")
    print("Now you can run 'python -m ml.relabel'.")

if __name__ == "__main__":
    backfill()
