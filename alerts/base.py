from abc import ABC, abstractmethod

class BaseAlert(ABC):
    """
    Abstract Base Class for all alert dispatchers in the AMT Engine.
    Forces all concrete implementations (Discord, Telegram, Console) 
    to have a standard `send` method signature.
    """
    
    @abstractmethod
    def send(self, signal_type: str, direction: str, trigger_price: float, **kwargs):
        """
        Dispatches the alert payload.
        
        Args:
            signal_type: Type of the heuristic (e.g. 'INITIATIVE_BREAKOUT', 'DELTA_SPIKE').
            direction: Trade direction ('LONG' or 'SHORT').
            trigger_price: The price point when the heuristic fired.
            **kwargs: Additional data specific to the signal (e.g., confidence, stop losses, time).
        """
        pass
