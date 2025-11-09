import os, glob, asyncio
from typing import List, Dict
from app.settings import get_settings
from app.ingest_qdrant import ensure_collection, upsert_points

S = get_settings()

def load_texts_from_folder(folder: str = "knowledge") -> List[Dict]:
    items = []
    for path in glob.glob(os.path.join(folder, "**", "*.*"), recursive=True):
        if any(path.lower().endswith(ext) for ext in (".md", ".txt")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            items.append({"path": path, "text": txt, "source": os.path.basename(path)})
        # (sencillo) si quieres PDF, conviértelo antes a .txt o usa pypdf en otra versión
    return items

async def main():
    await ensure_collection()
    docs = load_texts_from_folder("knowledge")
    if not docs:
        print("No hay documentos en knowledge/.")
        return
    texts = [d["text"] for d in docs]
    metas = [{"id": d["path"], "source": d["source"], "text": d["text"][:2000]} for d in docs]
    await upsert_points(texts, metas)
    print(f"Ingesta OK: {len(docs)} documentos.")

if __name__ == "__main__":
    asyncio.run(main())
