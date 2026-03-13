import logging
import json
from .base import BaseAlert

logging.basicConfig(level=logging.INFO, format='%(message)s') # Keep purely functional formatting 

class ConsoleAlert(BaseAlert):
    \"\"\"
    Simple console logger for alert signals. 
    Ideal for local testing before routing messages to generic webhooks.
    \"\"\"
    
    def __init__(self, raw_json=False):
        self.raw_json = raw_json

    def send(self, payload: dict):
        if self.raw_json:
            print(json.dumps(payload))
        else:
            signal_type = payload.get('signal_type', 'UNKNOWN')
            direction = payload.get('direction', 'NEUTRAL')
            asset = payload.get('asset', 'unknown').upper()
            trigger_price = payload.get('trigger_price', 0.0)
            
            if direction == 'LONG': emoji = \"🟢\"
            elif direction == 'SHORT': emoji = \"🔴\"
            elif direction == 'CONFLICT': emoji = \"⚠️\"
            else: emoji = \"⚪\"
            
            # Formatter
            header = f\"\\n{'='*40}\\n{emoji} AMT ORDER FLOW SIGNAL: {signal_type}\\n{'='*40}\"
            body = f\"  >> Asset: {asset}\\n  >> Direction: {direction}\\n  >> Trigger Price: {trigger_price}\"
            
            if payload.get('is_composite'):
                body += f\"\\n  >> Conflict: {payload.get('conflict', False)}\"
                comps = payload.get('component_signals', [])
                comp_str = \", \".join([f\"{c['signal_type']}({c['direction']})\" for c in comps])
                body += f\"\\n  >> Components: {comp_str}\"
            
            meta = payload.get('meta', {})
            for key, val in meta.items():
                body += f\"\\n  >> {key.capitalize().replace('_', ' ')}: {val}\"
                
            footer = f\"\\n{'='*40}\\n\"
            
            logging.info(f\"{header}\\n{body}{footer}\")

if __name__ == \"__main__\":
    alert = ConsoleAlert()
    alert.send({'signal_type': 'FALSE_BREAKOUT', 'direction': 'SHORT', 'asset': 'btcusdt', 'trigger_price': 69420.50, 'meta': {'target_poc': 68000.0, 'confidence': 'HIGH'}})
