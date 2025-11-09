import os, json, httpx, hashlib
from typing import List, Dict, Any
from app.settings import get_settings

S = get_settings()

# ---------------- Embeddings (Hugging Face Inference API) ----------------
async def hf_embed(texts: List[str]) -> List[List[float]]:
    api_url = S.HF_API_URL.rstrip("/") + f"/{S.HF_EMBED_MODEL}"
    headers = {"Authorization": f"Bearer {S.HF_API_TOKEN}"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(api_url, headers=headers, json={"inputs": texts})
        r.raise_for_status()
        data = r.json()
        # Si el endpoint devuelve un solo embedding para cada entrada ya viene como lista
        return data if isinstance(data[0][0], (int, float)) else data  # robust

# ---------------- Qdrant (REST) ----------------
def _q_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if S.QDRANT_API_KEY:
        headers["api-key"] = S.QDRANT_API_KEY
    return headers

async def ensure_collection() -> None:
    url = f"{S.QDRANT_URL}/collections/{S.QDRANT_COLLECTION}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=_q_headers())
        if r.status_code == 200:
            return
        # crear
        payload = {
            "vectors": {
                "size": S.QDRANT_VECTOR_SIZE,
                "distance": S.QDRANT_DISTANCE
            }
        }
        rc = await client.put(url, headers=_q_headers(), json=payload)
        rc.raise_for_status()

async def upsert_points(texts: List[str], metas: List[Dict[str, Any]]) -> None:
    vecs = await hf_embed(texts)
    points = []
    for i, (v, m) in enumerate(zip(vecs, metas)):
        pid = hashlib.md5((m.get("id") or m.get("path") or f"doc-{i}").encode()).hexdigest()
        points.append({"id": pid, "vector": v, "payload": m})
    url = f"{S.QDRANT_URL}/collections/{S.QDRANT_COLLECTION}/points?wait=true"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.put(url, headers=_q_headers(), json={"points": points})
        r.raise_for_status()

async def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    qvec = (await hf_embed([query]))[0]
    url = f"{S.QDRANT_URL}/collections/{S.QDRANT_COLLECTION}/points/search"
    payload = {"vector": qvec, "limit": top_k, "with_payload": True, "with_vector": False}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_q_headers(), json=payload)
        r.raise_for_status()
        out = r.json()
        return out.get("result", [])
