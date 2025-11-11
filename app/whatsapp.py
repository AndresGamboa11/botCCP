# app/whatsapp.py
import os
import httpx
import logging
import asyncio

logger = logging.getLogger("ccp.whatsapp")

WA_TOKEN = os.getenv("WA_ACCESS_TOKEN", "")
WA_PHONE_ID = os.getenv("WA_PHONE_NUMBER_ID", "")
WA_VER = os.getenv("WA_API_VERSION", "v21.0")

# ------------------- Enviar texto -------------------
async def send_whatsapp_text(to_number: str, body: str) -> dict:
    """
    Envía un mensaje de texto al usuario por WhatsApp Cloud API.
    Requiere WA_ACCESS_TOKEN, WA_PHONE_NUMBER_ID.
    """
    if not WA_TOKEN or not WA_PHONE_ID:
        logger.error("❌ Faltan WA_ACCESS_TOKEN o WA_PHONE_NUMBER_ID en el entorno")
        return {"ok": False, "error": "Falso WA_ACCESS_TOKEN o WA_PHONE_NUMBER_ID"}

    url = f"https://graph.facebook.com/{WA_VER}/{WA_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": body[:4096]},
    }

    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.post(url, headers=headers, json=payload)
        try:
            data = r.json()
        except Exception:
            data = {"text": r.text}
        ok = r.is_success
        if not ok:
            logger.error("❌ Error enviando mensaje: %s", data)
        return {"ok": ok, "status": r.status_code, "resp": data}

# ------------------- Enviar "escribiendo..." -------------------
async def send_typing_on(to_number: str) -> dict:
    """
    Envía la señal de 'typing_on' (escribiendo...) al chat.
    Opcional: mejora la UX.
    """
    if not WA_TOKEN or not WA_PHONE_ID:
        return {"ok": False, "error": "sin credenciales WA"}

    url = f"https://graph.facebook.com/{WA_VER}/{WA_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "action",
        "action": {"typing": "on"},
    }

    async with httpx.AsyncClient(timeout=15) as cli:
        try:
            r = await cli.post(url, headers=headers, json=payload)
            return {"ok": r.is_success, "status": r.status_code}
        except Exception as e:
            logger.debug("No se pudo enviar 'typing_on': %s", e)
            return {"ok": False, "error": str(e)}
