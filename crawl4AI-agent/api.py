from fastapi import FastAPI, Body
import uvicorn
from pydantic import BaseModel
import asyncio
from typing import List, Dict, Any

# Importar tus funciones de RAG
from pydantic_ai_expert import retrieve_relevant_chunks, generate_answer

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    # Recuperar chunks relevantes
    relevant_chunks = await retrieve_relevant_chunks(request.query)
    
    # Generar respuesta
    answer = await generate_answer(request.query, relevant_chunks)
    
    return {
        "answer": answer,
        "sources": [{"url": chunk["url"], "title": chunk["title"]} for chunk in relevant_chunks]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 