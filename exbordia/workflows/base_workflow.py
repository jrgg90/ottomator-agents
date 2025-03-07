from typing import Dict, Any, Optional

class BaseWorkflow:
    """
    Clase base para todos los workflows.
    """
    
    def __init__(self, openai_client=None, conversation_service=None):
        self.openai_client = openai_client
        self.conversation_service = conversation_service
    
    async def process(self, user_id: str, session_id: str, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Procesa un mensaje y devuelve una respuesta.
        Debe ser implementado por las clases hijas.
        """
        raise NotImplementedError("Este método debe ser implementado por las clases hijas")
