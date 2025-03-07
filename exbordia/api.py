from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import time
from openai import AsyncOpenAI
from supabase import create_client
from dotenv import load_dotenv

from config import DEBUG
from orchestrator.orchestrator import Orchestrator
from services.conversation_service import ConversationService

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
        start_time = time.time()
        
        # Obtener contexto de conversaciones anteriores
        context = await conversation_service.get_conversation_context(
            telegram_id=request.telegram_id,
            session_id=request.session_id,
            limit=5  # Últimas 5 conversaciones
        )
        
        # Preparar mensajes para OpenAI
        messages = [
            {"role": "system", "content": "Eres un asistente de IA avanzado que proporciona respuestas detalladas, actualizadas y precisas. Responde con contexto relevante y menciona información clave cuando sea necesario. "}
        ]
        
        # Añadir mensajes del contexto si existen
        if context.get("messages"):
            messages.extend(context.get("messages"))
        
        # Añadir el mensaje actual del usuario
        messages.append({"role": "user", "content": request.query})
        
        # Generar respuesta con OpenAI
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=1000
        )
        
        # Extraer la respuesta
        response = completion.choices[0].message.content
        
        # Calcular tiempo de ejecución y tokens
        execution_time = time.time() - start_time
        total_tokens = completion.usage.total_tokens
        
        # Guardar la conversación
        conversation = await conversation_service.save_conversation(
            telegram_id=request.telegram_id,
            session_id=request.session_id,
            question=request.query,
            answer=response,
            total_tokens=total_tokens,
            execution_time=execution_time
        )
        
        # Iniciar análisis en segundo plano (sin esperar)
        if conversation and "id" in conversation:
            import asyncio
            asyncio.create_task(
                conversation_service.analyze_and_update_conversation(
                    conversation_id=conversation["id"],
                    openai_client=openai_client
                )
            )
        
        return {
            "response": response,
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