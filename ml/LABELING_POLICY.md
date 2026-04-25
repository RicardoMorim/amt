# AMT Labeling Policy (FASE 1)

This document defines the deterministic labeling policy used by `ml/labeler.py`.

## Goal

Produce historical labels that are:
- auditable,
- walk-forward friendly,
- conservative in ambiguous cases,
- free from lookahead bias in feature construction.

## Inputs

- Signals from SQLite `signals` table.
- Future candles from SQLite `candles` table.

Minimum candle columns:
- `symbol`, `timeframe_secs`, `timestamp`, `open`, `high`, `low`, `close`.

## Core labeling rules

Given a signal at time `t0` and entry `trigger_price`:

1. Use **strictly future candles only** (`timestamp > t0`).
2. Limit search to `horizon_candles`.
3. For each candle, check TP/SL barriers:
   - LONG:
     - TP hit if `high >= entry * (1 + tp_pct)`
     - SL hit if `low  <= entry * (1 - sl_pct)`
   - SHORT:
     - TP hit if `low  <= entry * (1 - tp_pct)`
     - SL hit if `high >= entry * (1 + sl_pct)`
4. If TP and SL are both touched in the same candle:
   - default: `SL_FIRST` (conservative)
   - alternatives: `TP_FIRST`, `SKIP`.
5. If no barrier is hit within horizon:
   - `TIMEOUT`, exit on last candle close.
6. If invalid input or insufficient forward candles:
   - `SKIP` and `is_labeled = 0`.

## Costs model

Roundtrip transaction cost:

`roundtrip_cost_pct = 2 * (fee_pct + slippage_pct)`

Saved label fields are net-of-costs:
- `label_win_pct = max(0, max_favorable_pct - roundtrip_cost_pct)`
- `label_loss_pct = min(0, max_adverse_pct - roundtrip_cost_pct)`

## Status values

- `WIN`
- `LOSS`
- `TIMEOUT`
- `SKIP`

Extra metadata per signal:
- `label_status`, `label_reason`
- `label_exit_price`, `label_exit_time`
- `label_horizon_candles`, `label_tp_pct`, `label_sl_pct`
- `label_fee_pct`, `label_slippage_pct`
- `label_timeout_return_pct`
- `label_ambiguity_flag`

## Why this is walk-forward friendly

- Labels are generated from **future prices only** (relative to signal timestamp).
- Features in `dataset_builder` are time-safe:
  - explicit temporal sorting,
  - `merge_asof(..., direction='backward', allow_exact_matches=False)` for candle joins,
  - rolling indicators shifted by 1 (`shift(1)`) before merge,
  - no global normalization using full dataset future.
