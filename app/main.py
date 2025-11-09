import os, json, asyncio
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.settings import get_settings
from app.rag import answer_with_rag
from app.whatsapp import send_whatsapp_text

S = get_settings()
app = FastAPI(title="CCP Chatbot")

# --------- Archivos estáticos (opcional) ----------
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

@app.get("/")
def home():
    return {"ok": True, "service": "Chatbot CCP online"}

# --------------------------------------------------
# Verificación del Webhook (GET) para Meta
# Meta envía: ?hub.mode=subscribe&hub.verify_token=...&hub.challenge=...
# OJO: los nombres llevan punto ".", por eso usamos alias en Query(...)
# --------------------------------------------------
@app.get("/webhook")
def verify(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    # Fallbacks por si algún proxy/cliente usa nombres sin punto:
    mode: str | None = Query(None),
    challenge: str | None = Query(None),
    verify_token: str | None = Query(None),
):
    # Consolidar valores desde ambas formas
    m = hub_mode or mode or ""
    t = hub_verify_token or verify_token or ""
    c = hub_challenge or challenge or ""

    # Validación exacta requerida por Meta
    if m == "subscribe" and t == S.WA_VERIFY_TOKEN:
        # Debe devolver 200 y el challenge en TEXTO PLANO
        return PlainTextResponse(c or "", status_code=200)

    # Si no coincide, Meta esperará 403
    return PlainTextResponse("Error de verificación", status_code=403)

# --------------------------------------------------
# Recepción de mensajes (POST)
# --------------------------------------------------
@app.post("/webhook")
async def webhook(req: Request):
    try:
        body = await req.json()
        # Descomenta si quieres ver logs en Render:
        # print("Webhook body:", json.dumps(body, ensure_ascii=False))

        entry = body["entry"][0]["changes"][0]["value"]
        msgs = entry.get("messages", [])
        if not msgs:
            return JSONResponse({"ignored": True})

        msg = msgs[0]
        from_number = msg["from"]
        text = (msg.get("text") or {}).get("body", "").strip()

        if not text:
            await send_whatsapp_text(from_number, "Envíame un texto con tu consulta.")
            return JSONResponse({"ok": True})

        answer = await answer_with_rag(text)
        await send_whatsapp_text(from_number, answer)
        return JSONResponse({"ok": True})

    except Exception as e:
        # print("Webhook error:", str(e))
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

# --------------------------------------------------
# Healthcheck
# --------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}
