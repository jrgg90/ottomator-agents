from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

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

system_prompt = """
You are an expert advisor on cross-border e-commerce, specializing in helping Mexican sellers expand to US marketplaces including Amazon, eBay, and Etsy.

Your knowledge comes from official documentation, seller guides, and best practices for international selling. You have access to detailed information about shipping, taxes, listing optimization, account setup, payments, and other critical aspects of cross-border selling.

When responding to questions:
1. Always search for the most relevant information in your knowledge base first
2. Provide specific, actionable advice tailored to Mexican sellers entering the US market
3. Cite your sources and mention which marketplace (Amazon, eBay, Etsy) the information comes from
4. Clarify when policies differ between marketplaces
5. Be honest when you don't have specific information on a topic
6. Be patient and understanding with the user, if stuck, akcknoledge that you are an expert advisor and you are here to help.

Your goal is to help sellers navigate the complexities of international e-commerce with accurate, practical guidance. Focus on compliance requirements, best practices, and strategies that minimize costs and maximize success rates.

When answering, structure your responses clearly with headings and bullet points when appropriate. Include specific steps, requirements, or processes when available.
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
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str, marketplace: str = 'general') -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        marketplace: The marketplace to filter the documentation by
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'marketplace_filter': marketplace,
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            # Incluir el marketplace en el encabezado
            marketplace_info = f"[{doc['marketplace'].upper()}]" if 'marketplace' in doc else ""
            chunk_text = f"""
# {doc['title']} {marketplace_info}

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
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps], marketplace: str = None) -> str:
    """
    Retrieve a list of all available documentation pages, optionally filtered by marketplace.
    
    Args:
        ctx: The context including the Supabase client
        marketplace: Optional marketplace to filter by (Amazon, eBay, Etsy, or general)

    Returns:
        str: Formatted list of documentation pages grouped by marketplace
    """
    try:
        # Build query
        query = ctx.deps.supabase.from_('site_pages').select('url, marketplace, title').distinct('url')
        
        # Apply marketplace filter if provided
        if marketplace:
            query = query.eq('marketplace', marketplace.lower())
            
        # Execute query
        result = query.execute()
        
        if not result.data:
            return "No documentation pages found."
            
        # Group by marketplace
        pages_by_marketplace = {}
        for doc in result.data:
            mkt = doc.get('marketplace', 'general').upper()
            if mkt not in pages_by_marketplace:
                pages_by_marketplace[mkt] = []
            
            title = doc.get('title', 'Untitled')
            url = doc.get('url', '')
            pages_by_marketplace[mkt].append(f"- [{title}]({url})")
        
        # Format the output
        output = ["# Available Documentation Pages"]
        
        for mkt, pages in sorted(pages_by_marketplace.items()):
            output.append(f"\n## {mkt}")
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
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"