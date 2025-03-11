import os
import sys
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from dotenv import load_dotenv

from notion_client import AsyncClient as NotionClient
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)
notion_client = NotionClient(auth=os.getenv("NOTION_API_KEY"))

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    marketplace: str
    category: List[str]
    source_name: List[str]
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, doc_title: str, chunk_number: int) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    # If it's the first chunk, use the document title
    if chunk_number == 0 and doc_title:
        title = doc_title
    else:
        # For subsequent chunks, generate a title
        system_prompt = """You are an AI that extracts titles from documentation chunks.
        Return a JSON object with a 'title' key.
        Create a concise, descriptive title for this chunk of content."""
        
        try:
            response = await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Content:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
                ],
                response_format={ "type": "json_object" }
            )
            title_data = json.loads(response.choices[0].message.content)
            title = title_data.get('title', f"{doc_title} (Part {chunk_number+1})")
        except Exception as e:
            print(f"Error getting title: {e}")
            title = f"{doc_title} (Part {chunk_number+1})"
    
    # Generate summary
    system_prompt = """You are an AI that creates summaries from documentation chunks.
    Return a JSON object with a 'summary' key.
    Create a concise summary of the main points in this chunk."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Content:\n{chunk[:1500]}..."}  # Send first 1500 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        summary_data = json.loads(response.choices[0].message.content)
        summary = summary_data.get('summary', "No summary available")
    except Exception as e:
        print(f"Error getting summary: {e}")
        summary = "No summary available"
    
    return {"title": title, "summary": summary}

async def get_embedding(text: str) -> List[float]:
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

def extract_text_from_rich_text(rich_text):
    """Extract plain text from Notion's rich_text array."""
    if not rich_text:
        return ""
    return "".join([text_obj["plain_text"] for text_obj in rich_text])

def extract_content_from_blocks(blocks):
    """Extract content from Notion blocks recursively."""
    content = []
    
    for block in blocks:
        block_type = block["type"]
        
        if block_type == "paragraph":
            text = extract_text_from_rich_text(block["paragraph"]["rich_text"])
            if text:
                content.append(text)
                
        elif block_type == "heading_1":
            text = extract_text_from_rich_text(block["heading_1"]["rich_text"])
            if text:
                content.append(f"# {text}")
                
        elif block_type == "heading_2":
            text = extract_text_from_rich_text(block["heading_2"]["rich_text"])
            if text:
                content.append(f"## {text}")
                
        elif block_type == "heading_3":
            text = extract_text_from_rich_text(block["heading_3"]["rich_text"])
            if text:
                content.append(f"### {text}")
                
        elif block_type == "bulleted_list_item":
            text = extract_text_from_rich_text(block["bulleted_list_item"]["rich_text"])
            if text:
                content.append(f"- {text}")
                
        elif block_type == "numbered_list_item":
            text = extract_text_from_rich_text(block["numbered_list_item"]["rich_text"])
            if text:
                content.append(f"1. {text}")
                
        elif block_type == "code":
            language = block["code"].get("language", "")
            text = extract_text_from_rich_text(block["code"]["rich_text"])
            if text:
                content.append(f"```{language}\n{text}\n```")
                
        elif block_type == "to_do":
            checked = block["to_do"].get("checked", False)
            text = extract_text_from_rich_text(block["to_do"]["rich_text"])
            if text:
                content.append(f"- {'[x]' if checked else '[ ]'} {text}")
                
        elif block_type == "toggle":
            text = extract_text_from_rich_text(block["toggle"]["rich_text"])
            if text:
                content.append(f"**{text}**")
                
        elif block_type == "quote":
            text = extract_text_from_rich_text(block["quote"]["rich_text"])
            if text:
                content.append(f"> {text}")
                
        # Add more block types as needed
    
    return "\n\n".join(content)

async def process_chunk(chunk: str, chunk_number: int, doc_id: str, doc_title: str, 
                       marketplace: str, category: str, source_name: List[str]) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, doc_title, chunk_number)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Extract categories using AI
    ai_categories = await extract_categories_with_ai(chunk)
    
    # Create URL using doc_id
    url = f"notion://{doc_id}"
    
    # Create metadata
    metadata = {
        "source": "notion",
        "doc_id": doc_id,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Si tenemos categorías de IA, las usamos; de lo contrario, usamos la categoría de Notion
    final_categories = ai_categories if ai_categories and ai_categories != ["uncategorized"] else [category]
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        marketplace=marketplace,
        category=final_categories,
        source_name=source_name,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "marketplace": chunk.marketplace,
            "category": chunk.category,
            "source_name": chunk.source_name,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
            # Otros campos
            "source_id": chunk.metadata.get("doc_id"),
            "related_links": chunk.metadata.get("related_links")
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_notion_page(page):
    """Process a single Notion page and store it in Supabase."""
    try:
        # Extract page properties
        page_id = page["id"]
        properties = page["properties"]
        
        # Extract document title
        title = "Untitled Document"
        for prop_name, prop_data in properties.items():
            if prop_data["type"] == "title" and prop_data["title"]:
                title = " ".join([text_obj["plain_text"] for text_obj in prop_data["title"]])
                break
        
        # Extract marketplace
        marketplace = "general"
        if "Marketplace" in properties and properties["Marketplace"]["type"] == "select":
            select_data = properties["Marketplace"]["select"]
            if select_data:
                marketplace = select_data.get("name", "general").lower()
        
        # Extract category from Notion (como respaldo)
        notion_category = "uncategorized"  # Valor por defecto
        if "Category" in properties and properties["Category"]["type"] == "multi_select":
            multi_select_data = properties["Category"]["multi_select"]
            if multi_select_data and multi_select_data[0].get("name"):
                notion_category = multi_select_data[0].get("name", "uncategorized").lower()
        elif "Category" in properties and properties["Category"]["type"] == "select":
            select_data = properties["Category"]["select"]
            if select_data and select_data.get("name"):
                notion_category = select_data.get("name", "uncategorized").lower()
        
        # Extract source_name
        source_name = ["notion"]  # Valor por defecto
        if "source_name" in properties and properties["source_name"]["type"] == "multi_select":
            multi_select_data = properties["source_name"]["multi_select"]
            if multi_select_data:
                source_name = [item.get("name", "").lower() for item in multi_select_data]
                if not source_name:  # Si la lista está vacía después de procesar
                    source_name = ["notion"]
        
        # Extract document ID (use custom ID if available, otherwise use page_id)
        doc_id = page_id
        if "ID" in properties and properties["ID"]["type"] == "number":
            custom_id = properties["ID"]["number"]
            if custom_id is not None:
                doc_id = f"custom-{custom_id}"
        
        # Create URL for the document
        url = f"notion://{doc_id}"
        
        # Check if document already exists in Supabase
        result = supabase.from_("site_pages").select("url, chunk_number").eq("url", url).execute()
        existing_chunks = result.data
        
        if existing_chunks:
            print(f"Document {doc_id} already exists with {len(existing_chunks)} chunks. Updating...")
            # Delete existing chunks
            for chunk in existing_chunks:
                supabase.from_("site_pages").delete().eq("url", url).eq("chunk_number", chunk["chunk_number"]).execute()
            print(f"Deleted {len(existing_chunks)} existing chunks for document {doc_id}")
        
        # Get page content
        blocks_response = await notion_client.blocks.children.list(block_id=page_id)
        blocks = blocks_response.get("results", [])
        
        # Extract content from blocks
        content = extract_content_from_blocks(blocks)
        
        if not content.strip():
            print(f"Warning: No content extracted from page {title} ({doc_id})")
            return
        
        # Split content into chunks
        chunks = chunk_text(content)
        print(f"Processing {len(chunks)} chunks for document: {title}")
        
        # Process chunks in parallel
        tasks = [
            process_chunk(chunk, i, doc_id, title, marketplace, notion_category, source_name) 
            for i, chunk in enumerate(chunks)
        ]
        processed_chunks = await asyncio.gather(*tasks)
        
        # Store chunks in parallel
        insert_tasks = [
            insert_chunk(chunk) 
            for chunk in processed_chunks
        ]
        await asyncio.gather(*insert_tasks)
        
        print(f"Successfully processed document: {title} ({doc_id})")
        
    except Exception as e:
        print(f"Error processing page {page['id']}: {e}")

async def fetch_notion_documents(database_id: str):
    """Fetch all documents from a Notion database."""
    try:
        print(f"Fetching documents from Notion database: {database_id}")
        
        # Query the database
        response = await notion_client.databases.query(database_id=database_id)
        pages = response.get("results", [])
        
        if not pages:
            print("No documents found in the database")
            return
        
        print(f"Found {len(pages)} documents in Notion database")
        
        # Process each page
        for page in pages:
            await process_notion_page(page)
            
        print("Finished processing all documents")
        
    except Exception as e:
        print(f"Error fetching Notion documents: {e}")

async def extract_categories_with_ai(chunk: str) -> List[str]:
    """
    Utiliza IA para extraer categorías relevantes del contenido del chunk.
    
    Args:
        chunk: El texto del chunk a analizar
        
    Returns:
        Lista de categorías identificadas
    """
    categories_list = [
        "Logística", "Regulaciones y Aduanas", "Marketing y Publicidad", 
        "Ventas y Conversión", "Finanzas y Costos", "Estrategia de Negocio", 
        "Legales y Compliance", "Operaciones y Gestión de Inventario", 
        "Optimización de Listados", "Customer Service y Devoluciones", 
        "Amazon FBA y FBM", "Análisis de Competencia", 
        "Expansión a Otros Mercados", "Amazon Seller Central", 
        "Reembolsos y Cargos Ocultos"
    ]
    
    categories_with_descriptions = {
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
    
    # Crear una descripción detallada de cada categoría para el prompt
    categories_description = "\n".join([f"- {cat}: {desc}" for cat, desc in categories_with_descriptions.items()])
    
    system_prompt = f"""Eres un agente especializado en analizar contenido relacionado con Amazon Seller.
Tu tarea es identificar las categorías más relevantes para el texto proporcionado.

CATEGORÍAS DISPONIBLES:
{categories_description}

INSTRUCCIONES:
1. Lee cuidadosamente el contenido proporcionado.
2. Identifica las categorías que mejor representan el tema principal del contenido.
3. Puedes seleccionar una o varias categorías, pero solo de la lista proporcionada.
4. Devuelve un objeto JSON con una clave "categories" que contenga un array de las categorías seleccionadas.
5. Si no encuentras ninguna categoría relevante, incluye "uncategorized" en el array.

Ejemplo de respuesta:
{{"categories": ["Logística", "Amazon FBA y FBM"]}}
"""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contenido a categorizar:\n{chunk[:2000]}..."}  # Enviamos los primeros 2000 caracteres
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        extracted_categories = result.get("categories", ["uncategorized"])
        
        # Validar que las categorías estén en la lista permitida
        valid_categories = [cat for cat in extracted_categories if cat in categories_list or cat.lower() == "uncategorized"]
        
        # Si no hay categorías válidas, usar "uncategorized"
        if not valid_categories:
            valid_categories = ["uncategorized"]
            
        return valid_categories
        
    except Exception as e:
        print(f"Error extrayendo categorías con IA: {e}")
        return ["uncategorized"]

async def main():
    # Get Notion database ID from environment variable
    database_id = os.getenv("NOTION_DATABASE_ID")
    if not database_id:
        print("Error: NOTION_DATABASE_ID environment variable not set")
        return
    
    # Process all documents in the database
    await fetch_notion_documents(database_id)

if __name__ == "__main__":
    asyncio.run(main())