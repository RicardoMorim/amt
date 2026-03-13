import uuid
import datetime
from typing import List, Dict, Tuple

class SignalArbitrator:
    """
    Takes multiple raw signals generated in a single candle for the same asset
    and arbitrates them to prevent conflicting alerts and prepare ML datasets.
    """
    def __init__(self):
        # Optional weights if we want to determine direction heavily
        self.weights = {
            'INITIATIVE_BREAKOUT': 2.0,
            'DELTA_SPIKE': 1.5,
            'FALSE_BREAKOUT': 1.0,
            'CVD_DIVERGENCE_EXHAUSTION': 1.0
        }

    def _craft_json(self, raw_signal: dict, context: dict) -> dict:
        """ Formats a raw signal dict into the ML-ready dataset structure """
        signal = dict(context) # Copy base context metrics
        now_str = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z"
        
        signal['id'] = str(uuid.uuid4())
        signal['timestamp_event'] = now_str
        signal['signal_type'] = raw_signal['signal_type']
        signal['direction'] = raw_signal['direction']
        
        # Place heuristic-specific metadata in 'meta'
        meta = {}
        for k, v in raw_signal.items():
            if k not in ['signal_type', 'direction', 'asset']:
                meta[k] = v
        signal['meta'] = meta
        
        return signal

    def arbitrate(self, raw_signals: List[Dict], context: Dict) -> Tuple[Dict, List[Dict]]:
        """
        Returns:
            (composite_signal_dict, list_of_all_raw_json_signals)
        """
        all_jsons = [self._craft_json(rs, context) for rs in raw_signals]
        
        if not all_jsons:
            return None, []
            
        if len(all_jsons) == 1:
            # Only one signal, so the composite is effectively the single signal
            single = dict(all_jsons[0])
            single['is_composite'] = False
            return single, all_jsons
            
        # Handle multiple signals (Conflict vs Agreement)
        directions = set([s['direction'] for s in all_jsons])
        
        composite = dict(context)
        composite['id'] = str(uuid.uuid4())
        composite['timestamp_event'] = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z"
        composite['is_composite'] = True
        composite['component_signals'] = [
            {'signal_type': s['signal_type'], 'direction': s['direction'], 'weight': self.weights.get(s['signal_type'], 1.0)}
            for s in all_jsons
        ]
        
        if len(directions) == 1:
            # Complete agreement
            composite['direction'] = list(directions)[0]
            composite['signal_type'] = 'COMPOSITE_CONFLUENCE'
            composite['conflict'] = False
            composite['meta'] = {'description': f"Confluência de {len(all_jsons)} sinais."}
        else:
            # Conflict detected!
            composite['direction'] = 'CONFLICT'
            composite['signal_type'] = 'COMPOSITE_CONFLICT'
            composite['conflict'] = True
            composite['meta'] = {'description': f"Sinais opostos detetados na mesma vela."}
            
        return composite, all_jsons
