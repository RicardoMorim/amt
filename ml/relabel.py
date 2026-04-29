"""
ml/relabel.py
=============
CLI wrapper for robust, deterministic historical labeling.

Usage examples:
    python -m ml.relabel
    python -m ml.relabel --horizon-candles 8 --tp-pct 0.005 --sl-pct 0.003
    python -m ml.relabel --symbol btcusdt --timeframe-secs 900 --only-unlabeled
"""

import sys
import argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ml.labeler import LabelerConfig, SameCandlePolicy, relabel_sqlite


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robust AMT historical relabeling (SQLite candles-backed)")
    p.add_argument("--db-path", default=config.DB_PATH, help="Path to SQLite dataset DB")
    p.add_argument("--symbol", default=None, help="Optional symbol filter (e.g. btcusdt)")
    p.add_argument("--timeframe-secs", type=int, default=None, help="Optional timeframe filter")

    p.add_argument("--horizon-candles", type=int, default=8, help="Forward horizon in candles")
    p.add_argument("--tp-pct", type=float, default=0.005, help="Take-profit percentage (0.005 = 0.5%%)")
    p.add_argument("--sl-pct", type=float, default=0.003, help="Stop-loss percentage (0.003 = 0.3%%)")
    p.add_argument("--fee-pct", type=float, default=0.0004, help="Per-side fee percentage")
    p.add_argument("--slippage-pct", type=float, default=0.0002, help="Per-side slippage percentage")
    p.add_argument(
        "--same-candle-policy",
        type=str,
        default="SL_FIRST",
        choices=[p.value for p in SameCandlePolicy],
        help="Resolution when TP and SL are touched in the same candle",
    )
    p.add_argument("--only-unlabeled", action="store_true", help="Label only rows where is_labeled = 0")
    p.add_argument(
        "--min-forward-candles",
        type=int,
        default=1,
        help="Minimum forward candles required; otherwise row becomes SKIP",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = LabelerConfig(
        horizon_candles=args.horizon_candles,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        fee_pct=args.fee_pct,
        slippage_pct=args.slippage_pct,
        same_candle_policy=SameCandlePolicy(args.same_candle_policy),
        min_forward_candles_required=args.min_forward_candles,
        relabel_all=not args.only_unlabeled,
    )

    print("=" * 70)
    print("🏷️  AMT Robust Relabel")
    print("=" * 70)
    print(f"DB                : {args.db_path}")
    print(f"Symbol filter     : {args.symbol or '(all)'}")
    print(f"Timeframe filter  : {args.timeframe_secs or '(all)'}")
    print(f"Horizon candles   : {cfg.horizon_candles}")
    print(f"TP / SL           : {cfg.tp_pct:.4%} / {cfg.sl_pct:.4%}")
    print(f"Fees / Slippage   : {cfg.fee_pct:.4%} / {cfg.slippage_pct:.4%} (per side)")
    print(f"Same candle policy: {cfg.same_candle_policy.value}")
    print(f"Relabel all       : {cfg.relabel_all}")
    print("-" * 70)

    stats = relabel_sqlite(
        db_path=args.db_path,
        config=cfg,
        symbol=args.symbol,
        timeframe_secs=args.timeframe_secs,
    )

    print("Result:")
    print(f"  Processed: {stats['processed']}")
    print(f"  WIN     : {stats['WIN']}")
    print(f"  LOSS    : {stats['LOSS']}")
    print(f"  TIMEOUT : {stats['TIMEOUT']}")
    print(f"  SKIP    : {stats['SKIP']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
