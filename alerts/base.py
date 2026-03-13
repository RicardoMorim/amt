from abc import ABC, abstractmethod

class BaseAlert(ABC):
    \"\"\"
    Abstract Base Class for all alert dispatchers in the AMT Engine.
    Forces all concrete implementations (Discord, Telegram, Console) 
    to have a standard `send` method signature.
    \"\"\"
    
    @abstractmethod
    def send(self, payload: dict):
        \"\"\"
        Dispatches the alert payload.
        
        Args:
            payload: A structured JSON dictionary containing all signal features and ML context.
        \"\"\"
        pass
