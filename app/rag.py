import httpx, textwrap
from typing import List, Dict, Any
from app.settings import get_settings
from app import ingest_qdrant as qd

S = get_settings()

SYSTEM = """Eres el asistente virtual oficial de la Cámara de Comercio de Pamplona (Colombia).
Responde exclusivamente sobre servicios, trámites, horarios, eventos y contenidos de la CCP.
Sé claro, conciso y útil. Si algo no está en el contexto, responde: 
"No encuentro esa información en los datos de la Cámara de Comercio de Pamplona." 
Responde siempre en español.
"""

def build_prompt(query: str, docs: List[Dict[str, Any]]) -> str:
    ctx = []
    for d in docs:
        pl = d.get("payload", {})
        snippet = pl.get("text", "")[:1000]
        src = pl.get("source", "desconocido")
        ctx.append(f"- Fuente: {src}\n{snippet}")
    context_block = "\n\n".join(ctx) if ctx else "No hay contexto."
    prompt = f"""{SYSTEM}

Contexto:
{context_block}

Pregunta del usuario: {query}

Responde en 4-7 líneas, usando viñetas si conviene, y sin inventar datos.
"""
    return textwrap.dedent(prompt).strip()

async def answer_with_rag(user_text: str) -> str:
    # retrieve
    try:
        await qd.ensure_collection()
        hits = await qd.search(user_text, top_k=5)
    except Exception:
        hits = []

    prompt = build_prompt(user_text, hits)

    # LLM Groq (OpenAI-compatible)
    headers = {
        "Authorization": f"Bearer {S.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": S.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.groq.com/openai/v1/chat/completions",
                              headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

# app/rag.py (fragmento)
from app.providers import GroqClient
_groq = GroqClient()

def answer_with_rag(question: str, context_chunks):
    # arma el prompt RAG
    ctx = "\n\n".join(context_chunks) if context_chunks else "N/A"
    messages = [
        {"role": "system", "content": "Eres el asistente virtual oficial de la CCP. Responde breve y preciso."},
        {"role": "user", "content": f"Contexto:\n{ctx}\n\nPregunta: {question}"}
    ]
    return _groq.chat(messages, max_tokens=512, temperature=0.2)
