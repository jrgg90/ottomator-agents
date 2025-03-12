import logging
import asyncio
import sys
import os
from agents import Agent, Runner, set_default_openai_key, trace, WebSearchTool
from config import OPENAI_API_KEY
from services.conversation_service import ConversationService
from pydantic import BaseModel
# Añadir el directorio raíz al path para evitar conflictos de importación
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configurar logging
logger = logging.getLogger(__name__)

# Configurar la API key de OpenAI
set_default_openai_key(OPENAI_API_KEY)

# Inicializar el servicio de conversación
conversation_service = ConversationService()

# Crear el agente principal
amazon_seller_agent = Agent(
    name="Amazon Seller Expert",
    instructions="""Eres un experto en ayudar a vendedores mexicanos a expandirse al mercado de Amazon USA.
    
    Proporciona información clara, precisa y útil sobre:
    - Cómo registrarse como vendedor en Amazon USA
    - Requisitos legales y fiscales
    - Estrategias de envío y logística
    - Optimización de listados de productos
    - Estrategias de precios y promociones
    - Servicio al cliente para compradores estadounidenses
    - Herramientas y recursos recomendados
    
    Sé amable, profesional y proporciona ejemplos concretos cuando sea posible.
    Si no conoces la respuesta a algo, admítelo honestamente en lugar de inventar información.
    """,
    handoff_description="Especialista en ventas en Amazon USA",
    tools=[WebSearchTool()],
)

Onboarding_agent = Agent(
    name="Onboarding Agent",
    handoff_description="""Especialista en dar de alta y la benvenida al los nuevos 
    usuarios de Exbordia. Podrás reconocerlos si dicen que son nuevos usuarios de Exbordia.""",
    instructions="""Tu vas a ayudar a que los nuevos usuarios se sientan cómodos
    de usar Exbordia.
    Tu objetivo principal es recabar información de los usuarios haciendoles algunas preguntas:
    - ¿Cuál es tu nombre?
    - ¿En que industria estas? (ejemplo, textil, alimentos, cosmeticos, etc)
    - ¿Ya has vendido en Estados Unidos?
    - ¿Ya tienes cuenta de Amazon en México?
    - ¿Aproximadamente cuantos productos vas a vender?
    - ¿Me puedes compartir un link a alguno de tus productos en Amazon?
    No debes de obligar al usuario a que te conteste todas las preguntas. Si sientes
    que el usuario ya no se siente cómodo, puedes pasarle el control al siguiente agente.
    """,
    model="gpt-4o-mini",
)

# Definir agente de triage
triage_agent = Agent(
    name="Triage Agent",
    instructions="Determina si la pregunta es sobre ventas en Amazon o sobre historia",
    handoffs=[amazon_seller_agent, Onboarding_agent],
)


# Diccionario para almacenar los resultados previos por user_id
thread_results = {}

async def process_message(user_id, message_text, session_id):
    """
    Procesa un mensaje de usuario con el agente.
    
    Args:
        user_id: ID del usuario de Telegram
        message_text: Texto del mensaje
        session_id: ID de la sesión
        
    Returns:
        dict: Diccionario con la respuesta y metadatos
    """
    try:
        # Convertir user_id a string para usarlo como clave
        user_id_str = str(user_id)
        
        # Medir tiempo de ejecución
        start_time = asyncio.get_event_loop().time()
        
        # Usar el user_id como thread_id para mantener el contexto
        with trace(workflow_name="Triage Agent", group_id=user_id_str):
            # Verificar si hay un resultado previo para este usuario
            if user_id_str in thread_results:
                # Convertir el resultado anterior a formato de entrada y añadir el nuevo mensaje
                previous_result = thread_results[user_id_str]
                input_list = previous_result.to_input_list() + [{"role": "user", "content": message_text}]
                
                # Ejecutar el agente de triage con el historial y el nuevo mensaje
                result = await Runner.run(triage_agent, input_list)
            else:
                # Primera interacción, ejecutar el agente de triage solo con el mensaje actual
                result = await Runner.run(triage_agent, message_text)
            
            # Guardar el resultado para futuras interacciones
            thread_results[user_id_str] = result
        
        # Calcular tiempo de ejecución
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Obtener la respuesta
        response = result.final_output
        
        # Devolver un diccionario con la respuesta y metadatos
        return {
            "response": response,
            "execution_time": execution_time,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error al procesar mensaje con el agente: {e}")
        return {
            "response": f"Lo siento, ocurrió un error al procesar tu mensaje: {str(e)}",
            "execution_time": 0,
            "session_id": None
        } 