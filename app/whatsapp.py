# app/whatsapp.py
import os, httpx

WA_TOKEN    = os.getenv("WA_ACCESS_TOKEN", "")
WA_PHONE_ID = os.getenv("WA_PHONE_NUMBER_ID", "")
WA_VER      = os.getenv("WA_API_VERSION", "v21.0")

async def send_whatsapp_text(to_number: str, body: str) -> dict:
    if not WA_TOKEN or not WA_PHONE_ID:
        return {"ok": False, "error": "Falso WA_ACCESS_TOKEN o WA_PHONE_NUMBER_ID"}
    url = f"https://graph.facebook.com/{WA_VER}/{WA_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}
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
        return {"ok": r.is_success, "status": r.status_code, "resp": data}
