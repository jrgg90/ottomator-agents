from .general_workflow import GeneralWorkflow

# Función para inicializar todos los workflows
def initialize_workflows(openai_client=None, conversation_service=None):
    return {
        "general": GeneralWorkflow(
            openai_client=openai_client,
            conversation_service=conversation_service
        )
    }
