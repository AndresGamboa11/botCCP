import httpx, os
from app.settings import get_settings

S = get_settings()

async def send_whatsapp_text(to_number: str, body: str) -> dict:
    url = f"https://graph.facebook.com/{S.WA_API_VERSION}/{S.WA_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {S.WA_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": body[:4096]},
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        return {"status": r.status_code, "data": r.json() if r.content else {}}
