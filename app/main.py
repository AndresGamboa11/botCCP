# app/main.py
import os, json, logging, re
import httpx
import pandas as pd
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv, find_dotenv

from app.settings import get_settings
from app.rag import answer_with_rag, debug_qdrant_sample
from app.whatsapp import send_whatsapp_text, send_typing_on
from app.mcp_server import router as mcp_router

# ─────────────────────────────────────────────────────────────
# Cargar .env (NO sobreescribir variables de Render)
# ─────────────────────────────────────────────────────────────
_dotenv = find_dotenv(usecwd=True)
if _dotenv:
    load_dotenv(_dotenv, override=False)

S = get_settings()
app = FastAPI(title="CCP Chatbot")

# ─────────────────────────────────────────────────────────────
# Logs
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ccp")

# ─────────────────────────────────────────────────────────────
# Configuración LLM y CSV de afiliados
# ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
GROQ_MODEL = os.getenv("GROQ_MODEL") or "llama-3.1-70b-versatile"

AFFILIATES_CSV_PATH = os.getenv("AFFILIATES_CSV_PATH", "data/afiliados_camara_pamplona_demo.csv")

_affiliates_df = None


def _get_affiliates_df() -> pd.DataFrame:
    global _affiliates_df
    if _affiliates_df is not None:
        return _affiliates_df

    try:
        logger.info("Cargando CSV de afiliados desde %s", AFFILIATES_CSV_PATH)
        _affiliates_df = pd.read_csv(AFFILIATES_CSV_PATH)
    except Exception as e:
        logger.exception("Error cargando el CSV de afiliados: %s", e)
        _affiliates_df = pd.DataFrame()

    return _affiliates_df


def _call_llm(messages, temperature: float = 0.0) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("Falta GROQ_API_KEY en el entorno.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": float(temperature),
    }
    with httpx.Client(timeout=60) as cli:
        r = cli.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def route_query_mode(user_text: str) -> str:
    system = (
        "Eres un enrutador de consultas para el chatbot de la Cámara de Comercio de Pamplona.\n"
        "Clasifica cada pregunta en 'rag' o 'consult'."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    try:
        raw = _call_llm(messages, temperature=0.0)
    except Exception as e:
        logger.exception("Error en router LLM → usando RAG por defecto: %s", e)
        return "rag"

    raw = (raw or "").strip().lower()
    return "consult" if "consult" in raw else "rag"


def answer_with_consult(user_text: str) -> str:
    df = _get_affiliates_df()
    if df.empty:
        return "En este momento no puedo acceder a la base de afiliados."

    q = user_text.lower()

    import numpy as np
    subdf = df

    # Detectar NIT
    nit_match = re.search(r"\b(\d{9,10}-?\d?)\b", q)
    if nit_match:
        nit = nit_match.group(1)
        subdf = df[df["nit"].astype(str).str.contains(nit, na=False)]
    else:
        # Detectar número de registro mercantil
        rm_match = re.search(r"\b(\d{4,8})\b", q)
        if rm_match and "numero_registro_mercantil" in df.columns:
            try:
                rm = int(rm_match.group(1))
                subdf = df[df["numero_registro_mercantil"] == rm]
            except:
                pass

        # Filtrar tokens
        if subdf is df:
            tokens = [t for t in re.findall(r"[a-záéíóúñ0-9]+", q) if len(t) > 3]
            mask = np.full(len(df), True)
            for tok in tokens:
                col_masks = []
                for col in ["razon_social_o_nombre", "nombre_comercial", "sector_economico"]:
                    if col in df.columns:
                        col_masks.append(
                            df[col].astype(str).str.lower().str.contains(tok, na=False)
                        )
                if col_masks:
                    combined = col_masks[0]
                    for m in col_masks[1:]:
                        combined = combined | m
                    mask = mask & combined
            subdf = df[mask]

    matches = subdf.head(10).to_dict(orient="records")

    system = (
        "Eres el asistente de la Cámara de Comercio de Pamplona. "
        "Responde usando ÚNICAMENTE la información del JSON."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"query": user_text, "matches": matches}, ensure_ascii=False)}
    ]

    try:
        return _call_llm(messages, temperature=0.0)
    except:
        return "No pude generar una respuesta en este momento."


# ─────────────────────────────────────────────────────────────
# SMALL TALK INSTITUCIONAL
# ─────────────────────────────────────────────────────────────

SMALL_TALK_RESPONSES = {
    "hola": "¡Hola! 👋 Estoy aquí para ayudarte con información de la Cámara de Comercio de Pamplona. ¿En qué puedo asistirte?",
    "buenas": "¡Buenas! 👋 ¿En qué puedo ayudarte hoy?",
    "buenos dias": "¡Buenos días! ☀️ ¿Qué deseas consultar?",
    "buenas tardes": "¡Buenas tardes! 🌆 ¿En qué puedo ayudarte?",
    "buenas noches": "¡Buenas noches! 🌙 ¿Qué información necesitas?",
    "como estas": "¡Muy bien! Gracias por preguntar 😊 ¿En qué puedo ayudarte hoy?",
    "gracias": "¡Con gusto! 😊 Si necesitas más información, estoy aquí.",
    "ok": "Perfecto 👍 ¿Deseas consultar algo más?",
    "listo": "Genial 👍 ¿Qué otra información necesitas?",
    "jaja": "😄 Me alegra sacarte una sonrisa. ¿Qué deseas consultar?",
    "jeje": "😄 ¿En qué puedo ayudarte hoy?",
    "hey": "¡Hola! 👋 ¿Qué deseas consultar?",
}

def small_talk_reply(text: str):
    t = text.lower().strip()
    t = t.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")

    for key in SMALL_TALK_RESPONSES:
        if t.startswith(key):
            return SMALL_TALK_RESPONSES[key]

    return None


# ─────────────────────────────────────────────────────────────
# Archivos estáticos
# ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

# ─────────────────────────────────────────────────────────────
# MCP
# ─────────────────────────────────────────────────────────────
app.include_router(mcp_router)

# ─────────────────────────────────────────────────────────────
# Salud
# ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"ok": True, "service": "Chatbot CCP online"}


# ─────────────────────────────────────────────────────────────
# Webhook WhatsApp
# ─────────────────────────────────────────────────────────────
WA_VERIFY_TOKEN = (os.getenv("WA_VERIFY_TOKEN") or "").strip()

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == WA_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "OK")
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request):
    try:
        payload = await request.json()
    except:
        return {"ok": True}

    logger.info("WA IN: %s", json.dumps(payload)[:2000])

    from_number, text = None, None
    try:
        entry = payload["entry"][0]["changes"][0]["value"]
        if "messages" in entry:
            msg = entry["messages"][0]
            from_number = msg["from"]
            text = msg.get("text", {}).get("body", "")
    except:
        return {"ok": True}

    if not text:
        await send_whatsapp_text(from_number, "Hola 👋, por favor escribe tu consulta.")
        return {"ok": True}

    # ----- SMALL TALK (ANTES DE TODO) -----
    st = small_talk_reply(text)
    if st:
        await send_whatsapp_text(from_number, st)
        return {"ok": True}

    # ----- Modo RAG vs CONSULT -----
    try:
        mode = route_query_mode(text)
    except:
        mode = "rag"

    if mode == "consult":
        answer = answer_with_consult(text)
    else:
        answer = answer_with_rag(text)

    if not answer.strip():
        answer = "Lo siento, no encontré información específica sobre tu consulta."

    await send_whatsapp_text(from_number, answer)
    return {"ok": True}
