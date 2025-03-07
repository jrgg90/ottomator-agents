import time
import asyncio
from typing import Dict, Any, Optional
from .base_workflow import BaseWorkflow

class GeneralWorkflow(BaseWorkflow):
    """
    Workflow para procesar preguntas generales.
    """
    
    async def process(self, user_id: str, session_id: str, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Procesa un mensaje general y devuelve una respuesta.
        """
        start_time = time.time()
        
        # Obtener contexto de conversaciones anteriores
        conversation_context = await self.conversation_service.get_conversation_context(
            telegram_id=user_id,
            session_id=session_id,
            limit=5  # Últimas 5 conversaciones
        )
        
        # Preparar mensajes para OpenAI
        messages = [
            {"role": "system", "content": "Eres un asistente de IA avanzado que proporciona respuestas detalladas, actualizadas y precisas. Responde con contexto relevante y menciona información clave cuando sea necesario."}
        ]
        
        # Añadir mensajes del contexto si existen
        if conversation_context.get("messages"):
            messages.extend(conversation_context.get("messages"))
        
        # Añadir el mensaje actual del usuario
        messages.append({"role": "user", "content": message})
        
        # Generar respuesta con OpenAI
        completion = await self.openai_client.chat.completions.create(
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
        conversation = await self.conversation_service.save_conversation(
            telegram_id=user_id,
            session_id=session_id,
            question=message,
            answer=response,
            total_tokens=total_tokens,
            execution_time=execution_time
        )
        
        # Iniciar análisis en segundo plano (sin esperar)
        if conversation and "id" in conversation:
            asyncio.create_task(
                self.conversation_service.analyze_and_update_conversation(
                    conversation_id=conversation["id"],
                    openai_client=self.openai_client
                )
            )
        
        return {
            "response": response,
            "session_id": session_id,
            "workflow": "general"
        } 