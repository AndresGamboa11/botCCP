# app/main.py
import os, json, logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.settings import get_settings
from app.rag import answer_with_rag              # <- función SÍNCRONA (no usar await)
from app.whatsapp import send_whatsapp_text, send_typing_on  # <- funciones ASYNC
from app.mcp_server import router as mcp_router

S = get_settings()
app = FastAPI(title="CCP Chatbot")

# ---------------- Logs ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ccp")

# ---------------- Archivos estáticos (opcional) ----------------
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

# ---------------- MCP ----------------
app.include_router(mcp_router)

# ---------------- Salud ----------------
@app.get("/")
def home():
    return {"ok": True, "service": "Chatbot CCP online"}

@app.get("/healthz")
async def healthz():
    return {"ok": True, "env": os.environ.get("RENDER", "render"), "port": os.environ.get("PORT")}

# ---------------- Diagnóstico RAG ----------------
@app.get("/debug/rag")
def debug_rag(q: str = ""):
    if not q.strip():
        return JSONResponse({"error": "falta parámetro q"}, status_code=400)
    try:
        # answer_with_rag es SINCRONA → NO await
        ans = answer_with_rag(q)
        return {"query": q, "answer": ans}
    except Exception as e:
        logger.exception("[/debug/rag] Error")
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------------- WhatsApp Webhook ----------------
WA_VERIFY_TOKEN = (os.getenv("WA_VERIFY_TOKEN") or "").strip()

# 1) Verificación (GET)
@app.get("/webhook")
async def verify_webhook(
    mode: str = "",
    hub_mode: str = "",
    hub_challenge: str = "",
    hub_verify_token: str = ""
):
    _mode = (mode or hub_mode or "").lower()
    _token = (hub_verify_token or "").strip()

    if _mode == "subscribe" and WA_VERIFY_TOKEN and _token == WA_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "OK", status_code=200)
    return PlainTextResponse("Forbidden", status_code=403)

def _extract_wa_message(payload: dict):
    """
    Devuelve (from_number, text) si hay un mensaje de usuario.
    Si no hay mensaje (p. ej. 'statuses'), devuelve (None, None).
    """
    try:
        entry = payload.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        # mensajes entrantes
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
                    text = (interactive.get("button_reply", {}).get("title") or "").strip()
                elif itype == "list_reply":
                    text = (interactive.get("list_reply", {}).get("title") or "").strip()

            elif msg_type == "image":
                text = "imagen"

            return from_number, text

        # estatus de entrega (ignorar)
        if "statuses" in value:
            return None, None

    except Exception:
        return None, None

    return None, None

# 2) Recepción (POST)
@app.post("/webhook")
async def receive_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        # Meta a veces manda empty pings; responde 200 para evitar reintentos
        return JSONResponse({"ok": True}, status_code=200)

    logger.info("WA IN: %s", json.dumps(payload)[:2000])

    from_number, text = _extract_wa_message(payload)
    if not from_number:
        return JSONResponse({"ok": True, "note": "no user message"}, status_code=200)

    # Señal de 'escribiendo' (opcional)
    try:
        await send_typing_on(from_number)
    except Exception as e:
        logger.debug("typing_on error: %s", e)

    if not text:
        await send_whatsapp_text(from_number, "Hola 👋, por favor escribe tu consulta en texto.")
        return JSONResponse({"ok": True}, status_code=200)

    # ----- RAG -----
    try:
        # IMPORTANTE: NO usar await (answer_with_rag es síncrona)
        answer = answer_with_rag(text)
        if not answer or not answer.strip():
            answer = "Lo siento, no encontré información exacta sobre eso. ¿Puedes reformular tu pregunta?"
    except Exception as e:
        logger.exception("RAG error")
        # mensaje corto y claro para el usuario; logs muestran detalle técnico
        answer = "Hubo un error procesando tu solicitud. Intenta de nuevo en unos minutos."

    # ----- Responder por WhatsApp -----
    try:
        wa_res = await send_whatsapp_text(from_number, answer)  # esta sí es ASYNC
        logger.info("WA OUT: %s", wa_res)
    except Exception as e:
        logger.exception("WA send error")

    return JSONResponse({"ok": True}, status_code=200)
