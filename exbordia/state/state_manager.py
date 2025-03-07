from typing import Dict, Any, Optional

class StateManager:
    """
    Gestor de estado para mantener el contexto de las conversaciones.
    """
    
    def __init__(self):
        self.states = {}
    
    async def get_state(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado actual para un usuario y sesión.
        
        Args:
            user_id: ID del usuario
            session_id: ID de la sesión
            
        Returns:
            Estado actual
        """
        key = f"{user_id}:{session_id}"
        return self.states.get(key, {})
    
    async def update_state(self, user_id: str, session_id: str, state_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualiza el estado para un usuario y sesión.
        
        Args:
            user_id: ID del usuario
            session_id: ID de la sesión
            state_updates: Actualizaciones a aplicar al estado
            
        Returns:
            Estado actualizado
        """
        key = f"{user_id}:{session_id}"
        current_state = self.states.get(key, {})
        updated_state = {**current_state, **state_updates}
        self.states[key] = updated_state
        return updated_state
    
    async def clear_state(self, user_id: str, session_id: str) -> None:
        """
        Limpia el estado para un usuario y sesión.
        
        Args:
            user_id: ID del usuario
            session_id: ID de la sesión
        """
        key = f"{user_id}:{session_id}"
        if key in self.states:
            del self.states[key]
