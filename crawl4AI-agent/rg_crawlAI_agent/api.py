from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client
import uuid

# Importación corregida para la ubicación actual
from rg_pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps

load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    telegram_id: int = 0
    session_id: int = 0

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # Configurar clientes
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
        
        # Extraer el ID de Telegram
        telegram_id = request.telegram_id
        
        # Buscar el usuario en telegram_users
        result = supabase.from_("telegram_users") \
            .select("auth_user_id, is_activated") \
            .eq("telegram_id", telegram_id) \
            .execute()
        
        # Verificar si el usuario existe y está activado
        if not result.data or not result.data[0]["is_activated"]:
            return {"response": "Lo siento, tu cuenta no está activada. Por favor activa tu cuenta primero."}
        
        auth_user_id = result.data[0]["auth_user_id"]
        
        # Verificar suscripción
        subscription = supabase.from_("subscriptions") \
            .select("status, plan") \
            .eq("user_id", auth_user_id) \
            .eq("status", "active") \
            .execute()
        
        if not subscription.data:
            return {"response": "Tu suscripción no está activa. Por favor renueva tu suscripción."}
        
        # Crear dependencias
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client
        )
        
        # Ejecutar el agente
        agent_result = await pydantic_ai_expert.run(request.query, deps=deps)
        
        # Extraer la respuesta de texto del resultado
        if hasattr(agent_result, 'output'):
            response = agent_result.output
        elif hasattr(agent_result, 'result'):
            response = agent_result.result
        else:
            # Si no podemos encontrar la respuesta, convertimos el objeto a string
            response = str(agent_result)
        
        # Extraer métricas de uso
        total_tokens = 0
        execution_time = 0

        # Obtener información de uso
        if hasattr(agent_result, 'usage'):
            try:
                # usage es un método, necesitamos llamarlo
                usage_data = agent_result.usage()
                print(f"Usage data type: {type(usage_data)}")
                print(f"Usage data: {usage_data}")
                
                # Intentar extraer total_tokens de diferentes maneras
                if hasattr(usage_data, 'total_tokens'):
                    total_tokens = usage_data.total_tokens
                elif isinstance(usage_data, dict) and 'total_tokens' in usage_data:
                    total_tokens = usage_data['total_tokens']
                
                print(f"Total tokens: {total_tokens}")
            except Exception as e:
                print(f"Error al llamar a usage(): {e}")

        # Imprimir información de debugging
        print(f"Agent result type: {type(agent_result)}")
        print(f"Agent result attributes: {dir(agent_result)}")
        
        # Guardar la conversación con el auth_user_id y las métricas
        supabase.table("user_conversations").insert({
            "user_id": auth_user_id,  # Usamos el auth_user_id, no el telegram_id
            "session_id": request.session_id,
            "question": request.query,
            "answer": response,
            "marketplace": "general",
            "sources": [],
            "total_tokens": total_tokens,
            "execution_time": execution_time
        }).execute()
        
        return {"response": response}
    
    except Exception as e:
        print(f"Error: {e}")
        return {"response": "Lo siento, ocurrió un error al procesar tu consulta."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 