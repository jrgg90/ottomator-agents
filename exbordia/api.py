from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from openai import AsyncOpenAI
from supabase import create_client
from dotenv import load_dotenv

from config import DEBUG
from orchestrator.orchestrator import Orchestrator
from services.conversation_service import ConversationService
from state.state_manager import StateManager

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Exbordia API",
    description="API para el sistema de orquestación de agentes Exbordia",
    version="0.1.0",
    debug=DEBUG
)

# Inicializar servicios
load_dotenv()
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
conversation_service = ConversationService(database=supabase)

# Inicializar cliente de OpenAI
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Inicializar el orquestador
orchestrator = Orchestrator(
    conversation_service=conversation_service,
    state_manager=StateManager(),
    openai_client=openai_client
)

# Modelos de datos
class MessageRequest(BaseModel):
    telegram_id: int = 0
    session_id: int = 0
    query: str

class MessageResponse(BaseModel):
    response: str
    session_id: str

# Endpoints
@app.post("/message", response_model=MessageResponse)
async def process_message(request: MessageRequest):
    """
    Procesa un mensaje del usuario y devuelve una respuesta.
    """
    try:
        # Usar el orquestador para procesar el mensaje
        result = await orchestrator.process_message(
            user_id=request.telegram_id,
            session_id=request.session_id,
            message=request.query
        )
        
        return {
            "response": result["response"],
            "session_id": str(request.session_id) if request.session_id else "new_session"
        }
    except Exception as e:
        print(f"Error al procesar mensaje: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Endpoint para verificar el estado de la API.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)