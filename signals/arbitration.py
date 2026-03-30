import uuid
import datetime
from typing import List, Dict, Tuple


class SignalArbitrator:
    """
    Takes multiple raw signals generated in a single candle for the same asset
    and arbitrates them to prevent conflicting alerts and prepare ML datasets.

    Conflict resolution (improved):
      Instead of always emitting CONFLICT when signals disagree, the arbitrator
      now resolves by weighted vote.  Each signal type carries a predefined weight;
      if one direction's total weight exceeds the other's by at least
      `conflict_threshold`, the dominant direction wins and the composite is marked
      as RESOLVED_BY_WEIGHT instead of CONFLICT.  Only a near-tie produces a true
      CONFLICT signal (which is still saved for ML but suppressed from live alerts).
    """

    def __init__(self, conflict_threshold: float = 1.2):
        self.weights = {
            'INITIATIVE_BREAKOUT':       2.0,
            'DELTA_SPIKE':               1.5,
            'FALSE_BREAKOUT':            1.0,
            'CVD_DIVERGENCE_EXHAUSTION': 1.0,
        }
        # Ratio of winning_weight / losing_weight that must be exceeded to resolve
        self.conflict_threshold = conflict_threshold

    def _craft_json(self, raw_signal: dict, context: dict) -> dict:
        """Formats a raw signal dict into the ML-ready dataset structure."""
        signal = dict(context)

        if 'candle_time' in context:
            now_str = context['candle_time']
        else:
            now_str = (
                datetime.datetime.now(datetime.timezone.utc)
                .replace(tzinfo=None)
                .isoformat() + "Z"
            )

        signal['id'] = str(uuid.uuid4())
        signal['timestamp_event'] = now_str
        signal['signal_type'] = raw_signal['signal_type']
        signal['direction'] = raw_signal['direction']

        meta = {}
        for k, v in raw_signal.items():
            if k not in ['signal_type', 'direction', 'asset']:
                meta[k] = v
        signal['meta'] = meta

        return signal

    def _resolve_conflict(self, all_jsons: List[dict]) -> Tuple[str, str]:
        """
        Weighted vote to resolve conflicting signal directions.

        Returns:
            (direction, signal_type) — signal_type is 'COMPOSITE_CONFLICT' if the
            vote is too close to call, or 'COMPOSITE_CONFLUENCE' / 'RESOLVED_BY_WEIGHT'
            otherwise.
        """
        weight_by_dir: dict[str, float] = {}
        for s in all_jsons:
            d = s['direction']
            w = self.weights.get(s['signal_type'], 1.0)
            weight_by_dir[d] = weight_by_dir.get(d, 0.0) + w

        if len(weight_by_dir) == 1:
            direction = list(weight_by_dir.keys())[0]
            return direction, 'COMPOSITE_CONFLUENCE'

        sorted_dirs = sorted(weight_by_dir.items(), key=lambda x: x[1], reverse=True)
        winner_dir, winner_w = sorted_dirs[0]
        _, loser_w = sorted_dirs[1]

        ratio = winner_w / loser_w if loser_w > 0 else float('inf')

        if ratio >= self.conflict_threshold:
            return winner_dir, 'RESOLVED_BY_WEIGHT'
        else:
            return 'CONFLICT', 'COMPOSITE_CONFLICT'

    def arbitrate(self, raw_signals: List[Dict], context: Dict) -> Tuple[Dict, List[Dict]]:
        """
        Returns:
            (composite_signal_dict, list_of_all_raw_json_signals)
        """
        all_jsons = [self._craft_json(rs, context) for rs in raw_signals]

        if not all_jsons:
            return None, []

        if len(all_jsons) == 1:
            single = dict(all_jsons[0])
            single['is_composite'] = False
            return single, all_jsons

        direction, signal_type = self._resolve_conflict(all_jsons)

        composite = dict(context)
        composite['id'] = str(uuid.uuid4())
        composite['timestamp_event'] = (
            context.get('candle_time')
            or datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z"
        )
        composite['is_composite'] = True
        composite['component_signals'] = [
            {
                'signal_type': s['signal_type'],
                'direction': s['direction'],
                'weight': self.weights.get(s['signal_type'], 1.0),
            }
            for s in all_jsons
        ]
        composite['direction'] = direction
        composite['signal_type'] = signal_type
        composite['conflict'] = direction == 'CONFLICT'
        composite['meta'] = {
            'description': (
                f"Confluência de {len(all_jsons)} sinais."
                if direction != 'CONFLICT'
                else f"Sinais opostos detetados na mesma vela (ratio abaixo de {self.conflict_threshold})."
            )
        }

        return composite, all_jsons
