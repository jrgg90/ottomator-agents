import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Configuración de la aplicación
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Modelos de IA
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")