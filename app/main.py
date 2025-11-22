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

# Ruta del CSV con los afiliados. Puedes cambiarla vía variable de entorno.
AFFILIATES_CSV_PATH = os.getenv("AFFILIATES_CSV_PATH", "afiliados_camara_pamplona_demo.csv")

_affiliates_df = None


def _get_affiliates_df() -> pd.DataFrame:
    """Carga el CSV de afiliados una sola vez y lo reutiliza.

    Si hay algún problema al leer el archivo, devuelve un DataFrame vacío.
    """
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
    """Llama al modelo de Groq (misma API que se usa en RAG) y devuelve el texto.

    `messages` debe ser una lista de dicts con `role` y `content`.
    """
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
    """Usa el LLM para decidir si la pregunta va a RAG o a CONSULT (CSV).

    Retorna:
        'rag'     → usar answer_with_rag (base Qdrant).
        'consult' → usar answer_with_consult (CSV de afiliados).
    """
    system = (
        "Eres un enrutador de consultas para el chatbot de la Cámara de Comercio de Pamplona.\n"
        "Clasifica cada pregunta en una de dos categorías:\n"
        "1) 'rag' → cuando la pregunta trata sobre información general: trámites, requisitos, "
        "tarifas, horarios, servicios, eventos, etc.\n"
        "2) 'consult' → cuando la pregunta pide datos concretos de una o varias empresas "
        "(por ejemplo NIT, número de registro mercantil, razón social, nombre comercial, "
        "sector económico, tamaño de empresa, número de empleados, etc.).\n"
        "Responde ÚNICAMENTE con la palabra 'rag' o 'consult', sin explicaciones adicionales."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    try:
        raw = _call_llm(messages, temperature=0.0)
    except Exception as e:
        logger.exception("Error en router LLM, se usará RAG por defecto: %s", e)
        return "rag"

    raw = (raw or "").strip().lower()
    if "consult" in raw:
        return "consult"
    return "rag"


def answer_with_consult(user_text: str) -> str:
    """Consulta el CSV de afiliados y deja que el LLM arme la respuesta final.

    1. Carga el CSV en pandas.
    2. Filtra filas que parezcan relevantes para la pregunta.
    3. Construye un JSON con las coincidencias.
    4. Llama al LLM para que responda en lenguaje natural usando SOLO ese JSON.
    """
    df = _get_affiliates_df()
    if df.empty:
        return "En este momento no puedo acceder a la base de afiliados."

    q = user_text or ""
    q_lower = q.lower()

    # --- Heurísticas simples para buscar en el CSV ---
    import numpy as np

    subdf = df

    # 1) Si hay algo que parezca NIT en la pregunta
    nit_match = re.search(r"\b(\d{9,10}-?\d?)\b", q)
    if nit_match is not None:
        nit = nit_match.group(1)
        logger.info("[CONSULT] Detectado posible NIT: %s", nit)
        subdf = df[df["nit"].astype(str).str.contains(nit, na=False)]
    else:
        # 2) Si menciona 'registro' o 'matrícula' y hay número
        if "registro" in q_lower or "matr" in q_lower:
            rm_match = re.search(r"\b(\d{4,8})\b", q)
            if rm_match is not None and "numero_registro_mercantil" in df.columns:
                try:
                    rm = int(rm_match.group(1))
                    logger.info("[CONSULT] Detectado posible número de registro: %s", rm)
                    subdf = df[df["numero_registro_mercantil"] == rm]
                except ValueError:
                    pass

        # 3) Búsqueda por nombre / razón social / sector cuando no hay identificadores claros
        if subdf is df:  # aún no hemos filtrado
            tokens = [t for t in re.findall(r"[a-záéíóúñ0-9]+", q_lower) if len(t) > 3]
            if tokens:
                mask = np.full(len(df), True)
                for tok in tokens:
                    col_masks = []
                    for col in [
                        "razon_social_o_nombre",
                        "nombre_comercial",
                        "sector_economico",
                        "municipio",
                        "departamento",
                    ]:
                        if col in df.columns:
                            col_masks.append(
                                df[col]
                                .astype(str)
                                .str.lower()
                                .str.contains(tok, na=False)
                            )
                    if col_masks:
                        combined = col_masks[0]
                        for m in col_masks[1:]:
                            combined = combined | m
                        mask = mask & combined
                subdf = df[mask]

    # Limitar número de filas para no saturar al modelo
    subdf = subdf.head(10)
    matches = subdf.to_dict(orient="records")

    context = {
        "query": user_text,
        "matches": matches,
    }

    system = (
        "Eres el asistente de la Cámara de Comercio de Pamplona.\n"
        "Se te entrega un JSON con información de empresas afiliadas.\n"
        "Responde SIEMPRE en español, de forma clara y breve, usando SOLO los datos de ese JSON.\n"
        "Si la lista 'matches' está vacía, responde que no se encontró ninguna empresa que coincida con la consulta."
    )

    user_prompt = (
        "Pregunta del usuario:\n"
        + user_text
        + "\n\n"
        + "JSON con posibles coincidencias (campo 'matches'):\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    try:
        answer = _call_llm(messages, temperature=0.0)
        return answer or "No pude generar una respuesta en este momento."
    except Exception as e:
        logger.exception("Error en LLM de consulta CSV: %s", e)
        if matches:
            # Respaldo: devolvemos el JSON para inspección
            return "Tu consulta coincidió con estas empresas: " + json.dumps(
                matches, ensure_ascii=False
            )
        return "No se encontró ninguna empresa que coincida con tu consulta."


# ─────────────────────────────────────────────────────────────
# Archivos estáticos
# ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

# ─────────────────────────────────────────────────────────────
# MCP
# ─────────────────────────────────────────────────────────────
app.include_router(mcp_router)

# ─────────────────────────────────────────────────────────────
# Salud y diagnóstico
# ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"ok": True, "service": "Chatbot CCP online"}


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "env": os.environ.get("RENDER", "local"),
        "port": os.environ.get("PORT"),
    }


@app.get("/debug/env")
def debug_env():
    def ok(k): return bool(os.getenv(k))
    return {
        "HF_API_TOKEN": ok("HF_API_TOKEN"),
        "HF_EMBED_MODEL": os.getenv("HF_EMBED_MODEL"),
        "QDRANT_URL": ok("QDRANT_URL"),
        "QDRANT_API_KEY": ok("QDRANT_API_KEY"),
        "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION"),
        "GROQ_API_KEY": ok("GROQ_API_KEY"),
        "GROQ_MODEL": os.getenv("GROQ_MODEL"),
        "WA_ACCESS_TOKEN": ok("WA_ACCESS_TOKEN"),
        "WA_PHONE_NUMBER_ID": ok("WA_PHONE_NUMBER_ID"),
        "WA_VERIFY_TOKEN": ok("WA_VERIFY_TOKEN"),
        "WA_API_VERSION": os.getenv("WA_API_VERSION"),
    }


@app.get("/debug/rag")
def debug_rag(q: str = ""):
    if not q.strip():
        return JSONResponse({"error": "falta parámetro q"}, status_code=400)
    try:
        ans = answer_with_rag(q)
        return {"query": q, "answer": ans}
    except Exception as e:
        logger.exception("[/debug/rag] Error")
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────
# Debug Qdrant
# ─────────────────────────────────────────────────────────────
@app.get("/debug/qdrant")
def debug_qdrant():
    try:
        info = debug_qdrant_sample()
        return info
    except Exception as e:
        logger.exception("[/debug/qdrant] Error")
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────
# WhatsApp Webhook
# ─────────────────────────────────────────────────────────────
WA_VERIFY_TOKEN = (os.getenv("WA_VERIFY_TOKEN") or "").strip()

# 1) Verificación webhook (GET)
@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    mode = (hub_mode or "").lower()
    token = (hub_verify_token or "").strip()

    if mode == "subscribe" and WA_VERIFY_TOKEN and token == WA_VERIFY_TOKEN:
        # Meta espera que devolvamos el challenge en texto plano
        return PlainTextResponse(hub_challenge or "OK", status_code=200)

    return PlainTextResponse("Forbidden", status_code=403)


# 2) Extraer mensaje entrante
def _extract_wa_message(payload: dict):
    try:
        entry = payload.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        msgs = value.get("messages") or []
        if msgs:
            msg = msgs[0]
            from_number = msg.get("from")
            msg_type = msg.get("type")
            text = None

            if msg_type == "text":
                text = (msg.get("text", {}).get("body") or "").strip()

            elif msg_type == "interactive":
                interactive = msg.get("interactive", {})
                itype = interactive.get("type")
                if itype == "button_reply":
                    text = (
                        interactive.get("button_reply", {})
                        .get("title", "")
                        .strip()
                    )
                elif itype == "list_reply":
                    text = (
                        interactive.get("list_reply", {})
                        .get("title", "")
                        .strip()
                    )

            elif msg_type == "image":
                text = "imagen"

            return from_number, text

        if "statuses" in value:
            return None, None

    except Exception:
        return None, None


# 3) Recepción del webhook (POST)
@app.post("/webhook")
async def receive_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": True}, status_code=200)

    logger.info("WA IN: %s", json.dumps(payload)[:2000])

    from_number, text = _extract_wa_message(payload)
    if not from_number:
        return JSONResponse({"ok": True, "note": "no user message"}, status_code=200)

    try:
        await send_typing_on(from_number)
    except Exception:
        pass

    if not text:
        await send_whatsapp_text(from_number, "Hola 👋, por favor escribe tu consulta.")
        return JSONResponse({"ok": True}, status_code=200)

    # ----- Enrutamiento: RAG vs CONSULT (CSV) -----
    try:
        mode = route_query_mode(text)
    except Exception:
        logger.exception("Error al clasificar la consulta, se usará RAG por defecto.")
        mode = "rag"

    # ----- Generar respuesta según modo -----
    try:
        if mode == "consult":
            logger.info("[ROUTER] Usando modo CONSULT (CSV de afiliados)")
            answer = answer_with_consult(text)
        else:
            logger.info("[ROUTER] Usando modo RAG (Qdrant)")
            answer = answer_with_rag(text)

        if not isinstance(answer, str) or not answer.strip():
            answer = (
                "Lo siento, no encontré información exacta sobre eso. "
                "¿Puedes reformular tu pregunta?"
            )
    except Exception:
        logger.exception("Error interno generando la respuesta")
        answer = "Ocurrió un error interno, intenta nuevamente."

    # ----- enviar respuesta -----
    try:
        wa_res = await send_whatsapp_text(from_number, answer)
        logger.info("WA OUT: %s", wa_res)
    except Exception:
        logger.exception("WA send error")

    return JSONResponse({"ok": True}, status_code=200)
