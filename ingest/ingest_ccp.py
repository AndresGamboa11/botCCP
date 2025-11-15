# ingest/ingest_ccp.py
import os
import logging
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv, find_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ingest_ccp")

# ─────────────────────────────────────────────────────────────
# Carga .env (solo si existe)
# ─────────────────────────────────────────────────────────────
_dotenv = find_dotenv(usecwd=True)
if _dotenv:
    load_dotenv(_dotenv, override=False)

# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────
QDRANT_URL        = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY    = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION") or "ccp_docs").strip()

# Modelo preferido (por ti) – puede o no estar soportado por FastEmbed
EMBED_MODEL_ENV   = (os.getenv("HF_EMBED_MODEL") or "intfloat/multilingual-e5-small").strip()

# Archivo de conocimiento
KNOW_FILE_ENV     = (os.getenv("CCP_KNOW_FILE") or "knowledge/CCPAMPLONA.md").strip()

# ─────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────
def resolve_knowledge_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1]  # carpeta raíz del proyecto
    path = Path(KNOW_FILE_ENV)
    if not path.is_absolute():
        path = base_dir / path
    return path


def read_markdown_chunks(path: Path, max_chars: int = 600) -> List[str]:
    """
    Lee un archivo .md y lo divide en fragmentos de tamaño razonable.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de conocimiento: {path}")

    log.info(f"📄 Leyendo archivo: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")

    # Dividir por bloques separados por líneas en blanco
    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    chunks: List[str] = []
    for block in raw_blocks:
        if len(block) <= max_chars:
            chunks.append(block)
        else:
            # Si el bloque es muy grande, cortarlo en trozos
            start = 0
            while start < len(block):
                end = start + max_chars
                chunks.append(block[start:end])
                start = end

    log.info(f"✅ Fragmentos: {len(chunks)}")
    return chunks


def _normalize_supported_models(raw_supported) -> List[str]:
    """
    Convierte lo que devuelva FastEmbed (str / dict / objeto raro)
    a una lista de nombres de modelo (str).
    """
    names: List[str] = []
    for m in raw_supported:
        if isinstance(m, str):
            names.append(m)
        elif isinstance(m, dict):
            # FastEmbed suele usar claves tipo "model" o "name"
            name = m.get("model") or m.get("name") or m.get("id")
            if name:
                names.append(str(name))
        else:
            # Último recurso: cast a string
            names.append(str(m))
    # Eliminar duplicados conservando orden
    seen = set()
    unique: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def create_embedder_with_fallback(preferred_model: str) -> Tuple[TextEmbedding, str]:
    """
    Crea un TextEmbedding usando un modelo soportado por FastEmbed.
    - Intenta primero el modelo indicado en .env (si está en los soportados).
    - Luego una lista de modelos recomendados.
    - Luego recurre a cualquiera de los soportados.
    """
    raw_supported = TextEmbedding.list_supported_models()
    supported_names = _normalize_supported_models(raw_supported)

    log.info("🧠 Modelos soportados por FastEmbed:")
    for n in supported_names:
        log.info(f"   • {n}")

    candidates: List[str] = []

    # 1) Modelo que pusiste en .env
    if preferred_model:
        candidates.append(preferred_model)

    # 2) Modelos recomendados (multilingüe + algunos comunes)
    for m in (
        "intfloat/multilingual-e5-base",
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
    ):
        if m not in candidates:
            candidates.append(m)

    # 3) Añadir todos los soportados como últimos candidatos
    for n in supported_names:
        if n not in candidates:
            candidates.append(n)

    last_err: Exception | None = None

    for name in candidates:
        # Si tenemos lista de soportados, filtramos por ella
        if supported_names and name not in supported_names:
            log.warning(f"⚠ Modelo '{name}' no está en la lista soportada de FastEmbed. Se omite.")
            continue
        try:
            log.info(f"🧠 Cargando modelo FastEmbed: {name}")
            embedder = TextEmbedding(model_name=name)
            log.info(f"✅ Usando modelo de embeddings: {name}")
            return embedder, name
        except Exception as e:
            last_err = e
            log.warning(f"⚠ No se pudo inicializar modelo '{name}': {e}")

    raise RuntimeError(
        f"No se pudo inicializar ningún modelo de embeddings. Último error: {last_err}"
    )


def ensure_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
):
    """
    Crea (o recrea) la colección en Qdrant con el tamaño de vector correcto.
    """
    log.info(f"🗃️ Asegurando colección Qdrant '{collection_name}' (dim: {vector_size})")

    try:
        client.get_collection(collection_name)
        # Si existe, la recreamos para limpiar datos antiguos:
        log.info(f"🔁 Colección '{collection_name}' ya existe, se recreará.")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
        )
    except Exception:
        log.info(f"📦 Creando colección nueva '{collection_name}'")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
        )


def upload_documents_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: List[str],
    vectors: List[List[float]],
    source_name: str,
    batch_size: int = 64,
):
    """
    Sube los textos y sus vectores a Qdrant.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Número de textos ({len(chunks)}) y vectores ({len(vectors)}) no coincide."
        )

    log.info(f"🚀 Subiendo {len(chunks)} puntos a Qdrant (batch_size={batch_size})")

    points: List[qmodels.PointStruct] = []
    for idx, (text, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            "text": text,
            "source": source_name,
            "index": idx,
        }
        points.append(
            qmodels.PointStruct(
                id=idx,
                vector=vector,
                payload=payload,
            )
        )

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=batch_size,
    )

    log.info("✅ Ingesta completada y subida a Qdrant.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Debug de config
    print(f"🔎 DEBUG QDRANT_URL      : {QDRANT_URL}")
    print(f"🔎 DEBUG QDRANT_COLLECTION: {QDRANT_COLLECTION}")
    print(f"🔎 DEBUG EMBED_MODEL_ENV  : {EMBED_MODEL_ENV}")

    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL no está definido en el entorno (.env).")
    if not QDRANT_COLLECTION:
        raise RuntimeError("QDRANT_COLLECTION no está definido en el entorno (.env).")

    knowledge_path = resolve_knowledge_path()
    chunks = read_markdown_chunks(knowledge_path, max_chars=600)

    # Crear embedder con fallback robusto
    embedder, used_model = create_embedder_with_fallback(EMBED_MODEL_ENV)
    print(f"🧠 Modelo de embeddings FINAL: {used_model}")

    # Generar embeddings
    log.info("🧠 Generando embeddings con FastEmbed...")
    vectors: List[List[float]] = []
    for emb in embedder.embed(chunks):
        vectors.append(list(emb))

    if not vectors:
        raise RuntimeError("No se generaron vectores de embeddings (lista vacía).")

    dim = len(vectors[0])
    log.info(f"📐 Dimensión de vector: {dim}")

    # Conectar a Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        prefer_grpc=False,
    )

    # Asegurar colección
    ensure_qdrant_collection(client, QDRANT_COLLECTION, dim)

    # Subir datos
    upload_documents_to_qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        chunks=chunks,
        vectors=vectors,
        source_name=knowledge_path.name,
        batch_size=64,
    )

    print("🎉 Ingesta finalizada correctamente.")


if __name__ == "__main__":
    main()
