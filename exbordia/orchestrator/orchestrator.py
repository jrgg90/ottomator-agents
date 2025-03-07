from typing import Dict, Any, Optional
from services.conversation_service import ConversationService
from state.state_manager import StateManager

class Orchestrator:
    """
    Orquestador principal que coordina workflows y agentes.
    """
    
    def __init__(self, conversation_service=None, state_manager=None):
        self.conversation_service = conversation_service or ConversationService()
        self.state_manager = state_manager or StateManager()
        self.workflows = {}
        self.agents = {}
    
    async def process_message(self, user_id: str, session_id: str, message: str) -> Dict[str, Any]:
        """
        Procesa un mensaje y devuelve una respuesta.
        """
        # Por ahora, simplemente devuelve un eco del mensaje
        return {
            "response": f"Orchestrator received: {message}",
            "session_id": session_id
        }