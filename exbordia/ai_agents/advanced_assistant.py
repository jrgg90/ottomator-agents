import logging
import asyncio
from agents import Agent, Runner, set_default_openai_key
from config import OPENAI_API_KEY
from services.conversation_service import ConversationService

# Configurar logging
logger = logging.getLogger(__name__)

# Configurar la API key de OpenAI
set_default_openai_key(OPENAI_API_KEY)

# Inicializar el servicio de conversación
conversation_service = ConversationService()

# Crear agentes especializados
logistics_agent = Agent(
    name="Logistics Expert",
    handoff_description="Especialista en logística y envíos a Estados Unidos",
    instructions="""Eres un experto en logística y envíos para vendedores mexicanos en Amazon USA.
    
    Proporciona información detallada sobre:
    - Opciones de envío desde México a USA
    - Fulfillment by Amazon (FBA) vs. envío propio
    - Costos de envío y almacenamiento
    - Trámites aduaneros y documentación necesaria
    - Mejores prácticas para embalaje y etiquetado
    - Solución de problemas comunes de logística
    """,
)

marketing_agent = Agent(
    name="Marketing Expert",
    handoff_description="Especialista en marketing y optimización de listados en Amazon",
    instructions="""Eres un experto en marketing y optimización de listados para vendedores mexicanos en Amazon USA.
    
    Proporciona información detallada sobre:
    - Optimización de títulos, bullets y descripciones
    - Estrategias de palabras clave
    - Fotografía de productos
    - A+ Content y Brand Store
    - PPC y publicidad en Amazon
    - Promociones y cupones
    - Estrategias para mejorar reseñas
    """,
)

# Crear el agente principal con capacidad de handoff
amazon_seller_agent = Agent(
    name="Amazon Seller Expert",
    instructions="""Eres un experto en ayudar a vendedores mexicanos a expandirse al mercado de Amazon USA.
    
    Proporciona información clara, precisa y útil sobre todos los aspectos de vender en Amazon USA.
    Si la pregunta es específicamente sobre logística o marketing, considera hacer un handoff al agente especializado.
    
    Sé amable, profesional y proporciona ejemplos concretos cuando sea posible.
    Si no conoces la respuesta a algo, admítelo honestamente en lugar de inventar información.
    """,
    handoffs=[logistics_agent, marketing_agent],
)

async def process_message(user_id, message_text):
    """
    Procesa un mensaje de usuario con el agente.
    
    Args:
        user_id: ID del usuario de Telegram
        message_text: Texto del mensaje
        
    Returns:
        str: Respuesta del agente
    """
    try:
        # Convertir user_id a string para usarlo como clave
        user_id_str = str(user_id)
        
        # Obtener la sesión activa o crear una nueva
        session_id = conversation_service.get_active_session(user_id_str)
        
        # Obtener historial formateado
        conversation_history = conversation_service.format_conversation_history(
            user_id_str, 
            session_id, 
            limit=5
        )
        
        # Añadir el mensaje actual al historial
        if conversation_history:
            # Si hay historial, añadimos contexto
            full_message = f"{conversation_history}\nNueva pregunta: {message_text}"
        else:
            full_message = message_text
        
        # Medir tiempo de ejecución
        start_time = asyncio.get_event_loop().time()
        
        # Ejecutar el agente
        result = await Runner.run(amazon_seller_agent, full_message)
        
        # Calcular tiempo de ejecución
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Obtener la respuesta
        response = result.final_output
        
        # Guardar la conversación en la base de datos
        conversation_service.save_conversation(
            user_id=user_id_str,
            session_id=session_id,
            question=message_text,
            answer=response,
            execution_time=execution_time,
            # Guardar metadatos sobre qué agente respondió (si hubo handoff)
            metadata={
                "agent_used": result.agent_name if hasattr(result, 'agent_name') else "Amazon Seller Expert",
                "had_handoff": hasattr(result, 'handoff_details') and result.handoff_details is not None
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error al procesar mensaje con el agente: {e}")
        return f"Lo siento, ocurrió un error al procesar tu mensaje: {str(e)}" 