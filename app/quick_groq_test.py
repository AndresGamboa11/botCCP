# quick_groq_test.py
import os, httpx, json

API_KEY   = os.getenv("GROQ_API_KEY", "").strip()
MODEL     = os.getenv("GROQ_MODEL", "gemma2-9b-it")

assert API_KEY, "Falta GROQ_API_KEY en el entorno"

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": "Di 'ok' y nada más."}
    ],
    "temperature": 0.2,
    "max_tokens": 64,
    # "stream": False,  # opcional
}

with httpx.Client(timeout=60) as client:
    r = client.post(url, headers=headers, json=payload)
    if r.status_code >= 400:
        print("STATUS:", r.status_code)
        print("HEADERS:", r.headers)
        print("BODY:", r.text)  # <-- aquí Groq explica el motivo exacto
        raise SystemExit(1)

    data = r.json()
    print(json.dumps(data, indent=2))
    print("\nRespuesta:", data["choices"][0]["message"]["content"])
