from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import httpx
import os
import json
from typing import List, Dict, Any, Optional

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

try:
    import logfire
    logfire.configure(send_to_logfire='if-token-present')
except ImportError:
    print("Logfire no está instalado. La funcionalidad de logging estará limitada.")
    # Crear un objeto ficticio para evitar errores
    class DummyLogfire:
        @staticmethod
        def configure(*args, **kwargs):
            pass
    logfire = DummyLogfire

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

# Lista de categorías con descripciones para referencia del agente
AMAZON_CATEGORIES = {
    "Logística": "Envíos, fulfillment, almacenamiento y tiempos de entrega.",
    "Regulaciones y Aduanas": "Requisitos de importación, documentación, fracciones arancelarias.",
    "Marketing y Publicidad": "Amazon Ads, estrategias de PPC, branding en Amazon.",
    "Ventas y Conversión": "Cómo mejorar listados, obtener más reviews, Buy Box.",
    "Finanzas y Costos": "Tarifas de Amazon, impuestos, costos ocultos, márgenes de ganancia.",
    "Estrategia de Negocio": "Modelos de venta (FBA vs FBM), expansión, nichos rentables.",
    "Legales y Compliance": "Propiedad intelectual, restricciones de productos, términos de servicio.",
    "Operaciones y Gestión de Inventario": "Stock, reabastecimiento, proveedores, gestión con Amazon.",
    "Optimización de Listados": "Keywords, títulos, bullet points, imágenes, descripciones.",
    "Customer Service y Devoluciones": "Manejo de clientes, disputas, reembolsos y reputación.",
    "Amazon FBA y FBM": "Comparación entre modelos, ventajas y desventajas.",
    "Análisis de Competencia": "Herramientas para investigar a otros vendedores.",
    "Expansión a Otros Mercados": "Cómo escalar de Amazon US a otros marketplaces.",
    "Amazon Seller Central": "Manejo de la plataforma, reports, troubleshooting.",
    "Reembolsos y Cargos Ocultos": "Cómo reclamar cobros indebidos en Amazon."
}

# Prompt del sistema actualizado para enfocarse exclusivamente en Amazon
system_prompt = """
Eres un experto en Amazon para vendedores mexicanos que quieren expandirse al mercado estadounidense.

IMPORTANTE: Utiliza SIEMPRE las herramientas disponibles para buscar información específica sobre Amazon. No inventes información ni te bases en conocimiento general si hay documentación disponible.

Tu proceso para responder preguntas debe ser:
1. Identifica las categorías relevantes para la consulta
2. Busca documentación específica usando esas categorías
3. Proporciona respuestas detalladas y prácticas basadas en la documentación
4. Cita tus fuentes claramente

Recuerda que tu especialidad es ayudar a vendedores mexicanos a vender en Amazon USA, con foco en logística, regulaciones, marketing, y operaciones.
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_expert.tool
async def identify_relevant_categories(ctx: RunContext[PydanticAIDeps], user_query: str) -> List[str]:
    """
    Identifica las categorías más relevantes para la consulta del usuario.
    
    Args:
        ctx: El contexto con el cliente OpenAI
        user_query: La pregunta del usuario
        
    Returns:
        Lista de categorías relevantes para la consulta
    """
    print(f"🔍 TOOL CALLED: identify_relevant_categories - Query: '{user_query}'")
    
    try:
        system_prompt = f"""Identifica las categorías más relevantes para esta consulta sobre Amazon.
        Categorías disponibles con descripciones:
        {json.dumps(AMAZON_CATEGORIES, indent=2, ensure_ascii=False)}
        
        Devuelve solo un array JSON con los nombres de las 1-3 categorías más relevantes.
        Ejemplo: ["Logística", "Amazon FBA y FBM"]
        """
        
        response = await ctx.deps.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        categories = result.get("categories", [])
        
        print(f"Categorías identificadas: {categories}")
        return categories
    except Exception as e:
        print(f"Error identificando categorías: {e}")
        # Devolver algunas categorías generales en caso de error
        return ["Logística", "Amazon FBA y FBM"]

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str, marketplace: str = 'amazon') -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        marketplace: The marketplace to filter the documentation by
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    print(f"🔍 TOOL CALLED: retrieve_relevant_documentation - Query: '{user_query}', Marketplace: {marketplace}")
    
    try:
        # Primero identificar categorías relevantes
        categories = await identify_relevant_categories(ctx, user_query)
        
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Intentar usar la función optimizada si existe
        try:
            result = ctx.deps.supabase.rpc(
                'match_documents_by_category',
                {
                    'query_embedding': query_embedding,
                    'categories': categories,
                    'match_count': 5,
                    'match_threshold': 0.5
                }
            ).execute()
        except Exception as e:
            print(f"Error usando match_documents_by_category: {e}")
            print("Usando match_site_pages como fallback...")
            # Fallback a la función original si la optimizada no existe
            result = ctx.deps.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': query_embedding,
                    'match_count': 5,
                    'marketplace_filter': marketplace,
                }
            ).execute()
        
        if not result.data:
            return f"No relevant documentation found. I searched in these categories: {', '.join(categories)}."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            # Incluir categoría y marketplace en el encabezado
            category_info = f"[{', '.join(doc['category'])}]" if 'category' in doc and doc['category'] else ""
            marketplace_info = f"[{doc['marketplace'].upper()}]" if 'marketplace' in doc else ""
            header_info = " ".join(filter(None, [category_info, marketplace_info]))
            
            # Incluir resumen si está disponible
            summary = f"\n\n**Summary**: {doc['summary']}" if 'summary' in doc and doc['summary'] else ""
            
            chunk_text = f"""
# {doc['title']} {header_info}
{summary}

{doc['content']}

Source: {doc['url']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def retrieve_documentation_by_category(ctx: RunContext[PydanticAIDeps], user_query: str, specific_categories: List[str]) -> str:
    """
    Busca documentación relevante filtrando específicamente por categorías proporcionadas.
    
    Args:
        ctx: El contexto con los clientes
        user_query: La pregunta del usuario
        specific_categories: Lista específica de categorías para filtrar
        
    Returns:
        Documentación relevante formateada
    """
    print(f"🔍 TOOL CALLED: retrieve_documentation_by_category - Query: '{user_query}', Categories: {specific_categories}")
    
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Intentar usar la función optimizada
        try:
            result = ctx.deps.supabase.rpc(
                'match_documents_by_category',
                {
                    'query_embedding': query_embedding,
                    'categories': specific_categories,
                    'match_count': 5,
                    'match_threshold': 0.5
                }
            ).execute()
        except Exception as e:
            print(f"Error usando match_documents_by_category: {e}")
            # Fallback: filtrar manualmente por categoría después de la búsqueda vectorial
            result = ctx.deps.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': query_embedding,
                    'match_count': 10,  # Buscar más para compensar el filtrado posterior
                    'marketplace_filter': 'amazon',
                }
            ).execute()
            
            # Filtrar manualmente por categoría
            if result.data:
                filtered_data = []
                for doc in result.data:
                    if 'category' in doc and doc['category']:
                        # Verificar si hay intersección entre las categorías
                        if any(cat in specific_categories for cat in doc['category']):
                            filtered_data.append(doc)
                
                result.data = filtered_data[:5]  # Limitar a 5 resultados
        
        if not result.data:
            return f"No relevant documentation found in the specified categories: {', '.join(specific_categories)}."
            
        # Format the results (igual que en retrieve_relevant_documentation)
        formatted_chunks = []
        for doc in result.data:
            category_info = f"[{', '.join(doc['category'])}]" if 'category' in doc and doc['category'] else ""
            marketplace_info = f"[{doc['marketplace'].upper()}]" if 'marketplace' in doc else ""
            header_info = " ".join(filter(None, [category_info, marketplace_info]))
            
            summary = f"\n\n**Summary**: {doc['summary']}" if 'summary' in doc and doc['summary'] else ""
            
            chunk_text = f"""
# {doc['title']} {header_info}
{summary}

{doc['content']}

Source: {doc['url']}
"""
            formatted_chunks.append(chunk_text)
            
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation by category: {e}")
        return f"Error retrieving documentation by category: {str(e)}"

@pydantic_ai_expert.tool
async def get_quick_overview(ctx: RunContext[PydanticAIDeps], topic: str) -> str:
    """
    Proporciona una visión general rápida basada en resúmenes de documentos.
    
    Args:
        ctx: El contexto con el cliente Supabase
        topic: El tema o concepto a explorar
        
    Returns:
        Visión general basada en resúmenes
    """
    print(f"🔍 TOOL CALLED: get_quick_overview - Topic: '{topic}'")
    
    try:
        # Buscar documentos relevantes por título o palabras clave
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, summary, url, category') \
            .or_(f"title.ilike.%{topic}%,content.ilike.%{topic}%") \
            .order('created_at', desc=True) \
            .limit(5) \
            .execute()
        
        if not result.data:
            return f"No se encontraron resúmenes sobre '{topic}'."
        
        # Formatear los resúmenes
        overview = [f"# Visión general sobre: {topic}\n"]
        for doc in result.data:
            categories = f"[{', '.join(doc['category'])}]" if 'category' in doc and doc['category'] else ""
            overview.append(f"## {doc['title']} {categories}\n{doc['summary']}\n\nSource: {doc['url']}\n")
        
        return "\n".join(overview)
        
    except Exception as e:
        print(f"Error obteniendo visión general: {e}")
        return f"Error obteniendo visión general: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps], category: str = None) -> str:
    """
    Retrieve a list of all available documentation pages, optionally filtered by category.
    
    Args:
        ctx: The context including the Supabase client
        category: Optional category to filter by

    Returns:
        str: Formatted list of documentation pages grouped by category
    """
    print(f"🔍 TOOL CALLED: list_documentation_pages - Category: '{category}'")
    
    try:
        # Build query without distinct
        query = ctx.deps.supabase.from_('site_pages').select('url, title, category')
        
        # Apply category filter if provided
        if category:
            query = query.contains('category', [category])
            
        # Execute query
        result = query.execute()
        
        if not result.data:
            return "No documentation pages found."
        
        # Manually handle distinct URLs
        seen_urls = set()
        distinct_pages = []
        for page in result.data:
            if page['url'] not in seen_urls:
                seen_urls.add(page['url'])
                distinct_pages.append(page)
            
        # Group by category
        pages_by_category = {}
        for doc in distinct_pages:
            # Manejar múltiples categorías por documento
            if 'category' in doc and doc['category']:
                for cat in doc['category']:
                    if cat not in pages_by_category:
                        pages_by_category[cat] = []
                    
                    title = doc.get('title', 'Untitled')
                    url = doc.get('url', '')
                    pages_by_category[cat].append(f"- [{title}]({url})")
            else:
                # Documentos sin categoría
                if "Uncategorized" not in pages_by_category:
                    pages_by_category["Uncategorized"] = []
                
                title = doc.get('title', 'Untitled')
                url = doc.get('url', '')
                pages_by_category["Uncategorized"].append(f"- [{title}]({url})")
        
        # Format the output
        output = ["# Available Documentation Pages"]
        
        for cat, pages in sorted(pages_by_category.items()):
            output.append(f"\n## {cat}")
            output.extend(pages)
            
        return "\n".join(output)
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return f"Error retrieving documentation pages: {str(e)}"

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    print(f"🔍 TOOL CALLED: get_page_content - URL: '{url}'")
    
    try:
        # Verificar primero cuántos chunks tiene el documento
        count_result = ctx.deps.supabase.from_('site_pages').select('count', count='exact').eq('url', url).execute()
        chunk_count = count_result.count
        
        # Advertir si el documento es muy grande
        if chunk_count > 10:
            return f"Este documento es muy grande ({chunk_count} chunks). Por favor, especifica una consulta más enfocada o solicita una sección específica."
        
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number, category, summary') \
            .eq('url', url) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        
        # Obtener categorías y resumen del primer chunk
        categories = result.data[0].get('category', [])
        category_info = f"[{', '.join(categories)}]" if categories else ""
        summary = result.data[0].get('summary', '')
        
        # Construir el contenido formateado
        formatted_content = [f"# {page_title} {category_info}\n"]
        
        if summary:
            formatted_content.append(f"**Summary**: {summary}\n")
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

async def run_with_stats(deps, user_input):
    result = await pydantic_ai_expert.run(
        user_input,
        deps=deps
    )
    
    # Obtener el resultado del agente
    # Intentar diferentes formas de obtener el resultado
    output = None
    
    # Opción 1: Verificar si result.data contiene el resultado
    if hasattr(result, 'data'):
        output = str(result.data)
    
    # Opción 2: Verificar si el último mensaje contiene el resultado
    elif hasattr(result, 'all_messages') and result.all_messages:
        last_message = result.all_messages[-1]
        if hasattr(last_message, 'content'):
            output = last_message.content
    
    # Si no pudimos obtener el resultado, usar un mensaje de error
    if output is None:
        output = "No se pudo obtener una respuesta del agente."
    
    # Obtener estadísticas de uso si están disponibles
    token_count = 0
    if hasattr(result, 'usage') and callable(result.usage):
        try:
            usage_info = result.usage()
            if hasattr(usage_info, 'total_tokens'):
                token_count = usage_info.total_tokens
        except:
            pass
    
    return {
        "output": output,
        "total_tokens": token_count
    }