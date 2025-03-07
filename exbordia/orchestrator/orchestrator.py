from typing import Dict, Any, Optional
from services.conversation_service import ConversationService
from state.state_manager import StateManager
from workflows import initialize_workflows

class Orchestrator:
    """
    Orquestador principal que coordina workflows y agentes.
    """
    
    def __init__(self, conversation_service=None, state_manager=None, openai_client=None):
        self.conversation_service = conversation_service or ConversationService()
        self.state_manager = state_manager or StateManager()
        self.openai_client = openai_client
        
        # Inicializar workflows
        self.workflows = initialize_workflows(
            openai_client=openai_client,
            conversation_service=conversation_service
        )
    
    async def process_message(self, user_id: str, session_id: str, message: str) -> Dict[str, Any]:
        """
        Procesa un mensaje y devuelve una respuesta.
        """
        # Por ahora, simplemente usamos el workflow general para todas las preguntas
        # En el futuro, aquí determinaríamos qué workflow usar basado en el mensaje
        workflow = self.workflows["general"]
        
        # Procesar el mensaje con el workflow seleccionado
        result = await workflow.process(
            user_id=user_id,
            session_id=session_id,
            message=message
        )
        
        return result