import logging
import json
from .base import BaseAlert

logging.basicConfig(level=logging.INFO, format='%(message)s') # Keep purely functional formatting 

class ConsoleAlert(BaseAlert):
    """
    Simple console logger for alert signals. 
    Ideal for local testing before routing messages to generic webhooks.
    """
    
    def __init__(self, raw_json=False):
        self.raw_json = raw_json

    def send(self, signal_type: str, direction: str, trigger_price: float, **kwargs):
        payload = {
            'signal': signal_type,
            'direction': direction,
            'price': trigger_price,
            **kwargs
        }
        
        if self.raw_json:
            # Good for piping output straight to another system's stdout reader
            print(json.dumps(payload))
        else:
            # Good for human reading during terminal debugging
            emoji = "🟢" if direction == 'LONG' else "🔴"
            
            # Formatter
            header = f"\\n{'='*40}\\n{emoji} AMT ORDER FLOW SIGNAL: {signal_type}\\n{'='*40}"
            body = f"  >> Direction: {direction}\\n  >> Trigger Price: {trigger_price}"
            
            # Print kwargs nicely
            for key, val in kwargs.items():
                body += f"\\n  >> {key.capitalize().replace('_', ' ')}: {val}"
                
            footer = f"\\n{'='*40}\\n"
            
            logging.info(f"{header}\\n{body}{footer}")

if __name__ == "__main__":
    # Simple Local test
    alert = ConsoleAlert()
    alert.send("FALSE_BREAKOUT", "SHORT", 69420.50, target=68000.0, volume_spike="200%")
