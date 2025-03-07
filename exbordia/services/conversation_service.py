from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import json

class ConversationService:
    """
    Servicio para gestionar y almacenar conversaciones.
    """
    
    def __init__(self, database=None):
        """
        Inicializa el servicio de conversación.
        
        Args:
            database: Cliente de base de datos (Supabase)
        """
        self.database = database
        self._user_id_cache = {}  # Caché para auth_user_id
    
    async def _get_auth_user_id(self, telegram_id: int) -> str:
        """
        Obtiene el auth_user_id a partir del telegram_id, utilizando caché.
        
        Args:
            telegram_id: ID de Telegram del usuario
            
        Returns:
            ID de usuario en el sistema de autenticación
            
        Raises:
            ValueError: Si el usuario no existe
        """
        # Verificar caché primero
        if telegram_id in self._user_id_cache:
            return self._user_id_cache[telegram_id]
        
        if not self.database:
            raise ValueError("No hay conexión a la base de datos")
        
        # Consultar base de datos
        user_result = self.database.from_("telegram_users") \
            .select("auth_user_id") \
            .eq("telegram_id", telegram_id) \
            .execute()
        
        if not user_result.data:
            raise ValueError(f"Usuario con telegram_id {telegram_id} no encontrado")
        
        auth_user_id = user_result.data[0]["auth_user_id"]
        
        # Guardar en caché
        self._user_id_cache[telegram_id] = auth_user_id
        
        return auth_user_id
    
    async def save_conversation(self, 
                               telegram_id: int, 
                               session_id: int, 
                               question: str, 
                               answer: str,
                               total_tokens: int = 0,
                               execution_time: float = 0.0) -> Dict[str, Any]:
        """
        Guarda una conversación completa (pregunta y respuesta) en la base de datos.
        
        Args:
            telegram_id: ID de Telegram del usuario
            session_id: ID de la sesión
            question: Pregunta del usuario
            answer: Respuesta del sistema
            total_tokens: Total de tokens utilizados
            execution_time: Tiempo de ejecución en segundos
            
        Returns:
            Diccionario con los datos de la conversación guardada
        """
        if not self.database:
            # Si no hay base de datos, solo imprimimos y devolvemos datos simulados
            print(f"[User] {question}")
            print(f"[Assistant] {answer}")
            return {
                "id": str(uuid.uuid4()),
                "telegram_id": telegram_id,
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "created_at": datetime.now().isoformat(),
                "total_tokens": total_tokens,
                "execution_time": execution_time
            }
        
        try:
            # Obtener el auth_user_id usando la función con caché
            auth_user_id = await self._get_auth_user_id(telegram_id)
            
            # Obtener el último message_sequence para esta sesión
            sequence_result = self.database.from_("user_conversations") \
                .select("message_sequence") \
                .eq("user_id", auth_user_id) \
                .eq("session_id", session_id) \
                .order("message_sequence", desc=True) \
                .limit(1) \
                .execute()
            
            # Determinar el siguiente número de secuencia
            next_sequence = 1
            if sequence_result.data:
                next_sequence = sequence_result.data[0]["message_sequence"] + 1
            
            # Insertar la nueva conversación
            result = self.database.from_("user_conversations").insert({
                "user_id": auth_user_id,
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "total_tokens": total_tokens,
                "execution_time": execution_time,
                "message_sequence": next_sequence,
                "metadata": {}  # Inicialmente vacío
            }).execute()
            
            return result.data[0] if result.data else {}
            
        except Exception as e:
            print(f"Error al guardar conversación: {e}")
            # En caso de error, devolvemos un diccionario vacío
            return {}
    
    async def get_recent_conversations(self, 
                                      telegram_id: int, 
                                      session_id: int, 
                                      limit: int = 5) -> List[Dict[str, Any]]:
        """
        Obtiene las conversaciones más recientes para un usuario y sesión.
        
        Args:
            telegram_id: ID de Telegram del usuario
            session_id: ID de la sesión
            limit: Número máximo de conversaciones a recuperar
            
        Returns:
            Lista de conversaciones ordenadas por secuencia (más recientes primero)
        """
        if not self.database:
            # Si no hay base de datos, devolvemos una lista vacía
            return []
        
        try:
            # Obtener el auth_user_id usando la función con caché
            auth_user_id = await self._get_auth_user_id(telegram_id)
            
            # Obtener las conversaciones recientes
            result = self.database.from_("user_conversations") \
                .select("*") \
                .eq("user_id", auth_user_id) \
                .eq("session_id", session_id) \
                .order("message_sequence", desc=True) \
                .limit(limit) \
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error al obtener conversaciones recientes: {e}")
            return []
    
    async def get_conversation_by_id(self, conversation_id: str) -> Dict[str, Any]:
        """
        Obtiene una conversación específica por su ID.
        
        Args:
            conversation_id: ID de la conversación
            
        Returns:
            Diccionario con los datos de la conversación
        """
        if not self.database:
            return {}
        
        try:
            result = self.database.from_("user_conversations") \
                .select("*") \
                .eq("id", conversation_id) \
                .execute()
            
            return result.data[0] if result.data else {}
            
        except Exception as e:
            print(f"Error al obtener conversación por ID: {e}")
            return {}

    async def get_conversation_context(self, 
                                      telegram_id: int, 
                                      session_id: int,
                                      limit: int = 5) -> Dict[str, Any]:
        """
        Obtiene el contexto de conversación para usar con agentes de IA.
        
        Args:
            telegram_id: ID de Telegram del usuario
            session_id: ID de la sesión
            limit: Número máximo de conversaciones a incluir
            
        Returns:
            Diccionario con el contexto formateado para el agente
        """
        # Obtener conversaciones recientes
        conversations = await self.get_recent_conversations(
            telegram_id=telegram_id,
            session_id=session_id,
            limit=limit
        )
        
        # Si no hay conversaciones, devolver contexto vacío
        if not conversations:
            return {
                "messages": [],
                "has_history": False
            }
        
        # Formatear conversaciones para el contexto
        formatted_messages = self.format_conversation_for_context(conversations)
        
        # Extraer metadatos relevantes
        metadata = {}
        for conv in conversations:
            if conv.get("metadata"):
                metadata.update(conv.get("metadata", {}))
        
        return {
            "messages": formatted_messages,
            "has_history": len(formatted_messages) > 0,
            "metadata": metadata
        }

    def format_conversation_for_context(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Formatea las conversaciones para su uso en el contexto del agente.
        
        Args:
            conversations: Lista de conversaciones desde la base de datos
            
        Returns:
            Lista de mensajes formateados para el agente
        """
        # Ordenar conversaciones por secuencia (ascendente)
        sorted_conversations = sorted(
            conversations, 
            key=lambda x: x.get("message_sequence", 0)
        )
        
        formatted_messages = []
        
        for conv in sorted_conversations:
            # Añadir mensaje del usuario
            if conv.get("question"):
                formatted_messages.append({
                    "role": "user",
                    "content": conv.get("question", "")
                })
            
            # Añadir respuesta del asistente
            if conv.get("answer"):
                formatted_messages.append({
                    "role": "assistant",
                    "content": conv.get("answer", "")
                })
        
        return formatted_messages

    async def analyze_and_update_conversation(self,
                                             conversation_id: str,
                                             openai_client=None) -> Dict[str, Any]:
        """
        Analiza una conversación y actualiza sus metadatos con información enriquecida.
        
        Args:
            conversation_id: ID de la conversación a analizar
            openai_client: Cliente de OpenAI para realizar el análisis
            
        Returns:
            Diccionario con los datos actualizados de la conversación
        """
        if not self.database or not openai_client:
            return {}
        
        try:
            # Obtener la conversación
            conversation = await self.get_conversation_by_id(conversation_id)
            if not conversation:
                return {}
            
            # Extraer pregunta y respuesta
            question = conversation.get("question", "")
            answer = conversation.get("answer", "")
            
            # Realizar análisis con LLM
            analysis = await self._analyze_with_llm(
                question=question,
                answer=answer,
                openai_client=openai_client
            )
            
            # Actualizar la conversación con los resultados del análisis
            if analysis:
                updates = {
                    "sentiment": analysis.get("sentiment", ""),
                    "summary": analysis.get("summary", ""),
                    "topics": analysis.get("topics", []),
                    "metadata": {
                        **conversation.get("metadata", {}),
                        "analysis_timestamp": datetime.now().isoformat(),
                        "entities": analysis.get("entities", []),
                        "intent": analysis.get("intent", "")
                    }
                }
                
                # Actualizar en la base de datos
                result = self.database.from_("user_conversations") \
                    .update(updates) \
                    .eq("id", conversation_id) \
                    .execute()
                
                # Combinar los datos originales con las actualizaciones
                return {**conversation, **updates}
            
            return conversation
            
        except Exception as e:
            print(f"Error al analizar y actualizar conversación: {e}")
            return {}

    async def _analyze_with_llm(self, 
                               question: str, 
                               answer: str, 
                               openai_client) -> Dict[str, Any]:
        """
        Utiliza un LLM para analizar una conversación y extraer información relevante.
        
        Args:
            question: Pregunta del usuario
            answer: Respuesta del sistema
            openai_client: Cliente de OpenAI
            
        Returns:
            Diccionario con los resultados del análisis
        """
        try:
            # Construir el prompt para el análisis
            prompt = f"""
            Analiza la siguiente conversación entre un usuario y un asistente:
            
            Usuario: {question}
            
            Asistente: {answer}
            
            Por favor, proporciona la siguiente información en formato JSON:
            1. sentiment: El sentimiento general del usuario (positivo, negativo o neutral)
            2. summary: Un breve resumen de la conversación (máximo 100 caracteres)
            3. topics: Una lista de hasta 3 temas principales discutidos
            4. entities: Una lista de entidades mencionadas (productos, lugares, personas, etc.)
            5. intent: La intención principal del usuario (consulta, queja, solicitud, etc.)
            
            Responde solo con el JSON, sin texto adicional.
            """
            
            # Realizar la llamada al LLM
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente especializado en análisis de conversaciones."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Extraer y parsear la respuesta
            content = response.choices[0].message.content
            analysis = json.loads(content)
            
            return analysis
            
        except Exception as e:
            print(f"Error al analizar con LLM: {e}")
            return {}

    async def generate_session_summary(self,
                                      telegram_id: int,
                                      session_id: int,
                                      openai_client=None,
                                      max_conversations: int = 10) -> str:
        """
        Genera un resumen de toda la sesión de conversación.
        
        Args:
            telegram_id: ID de Telegram del usuario
            session_id: ID de la sesión
            openai_client: Cliente de OpenAI
            max_conversations: Número máximo de conversaciones a considerar
            
        Returns:
            Resumen de la sesión
        """
        if not self.database or not openai_client:
            return "No se pudo generar un resumen de la sesión."
        
        try:
            # Obtener las conversaciones de la sesión
            conversations = await self.get_recent_conversations(
                telegram_id=telegram_id,
                session_id=session_id,
                limit=max_conversations
            )
            
            if not conversations:
                return "No hay conversaciones para resumir."
            
            # Formatear las conversaciones para el prompt
            conversation_text = ""
            for conv in sorted(conversations, key=lambda x: x.get("message_sequence", 0)):
                conversation_text += f"Usuario: {conv.get('question', '')}\n"
                conversation_text += f"Asistente: {conv.get('answer', '')}\n\n"
            
            # Construir el prompt para el resumen
            prompt = f"""
            A continuación se muestra una conversación entre un usuario y un asistente.
            Por favor, genera un resumen conciso pero informativo de toda la conversación,
            destacando los puntos principales, las preguntas clave del usuario y las
            soluciones proporcionadas.
            
            Conversación:
            {conversation_text}
            
            Resumen:
            """
            
            # Realizar la llamada al LLM
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente especializado en resumir conversaciones."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=250
            )
            
            # Extraer el resumen
            summary = response.choices[0].message.content.strip()
            
            return summary
            
        except Exception as e:
            print(f"Error al generar resumen de sesión: {e}")
            return "Error al generar el resumen de la sesión."