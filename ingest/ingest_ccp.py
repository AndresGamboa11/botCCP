# ingest/ingest_ccp.py (fragmento relevante)

import os
import logging
from fastembed import TextEmbedding

logger = logging.getLogger("ingest")

# ─────────────────────────────────────────────
# ENV: modelo de embeddings
# ─────────────────────────────────────────────
# Modelo por defecto recomendado (multilingüe, 384 dims)
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

EMBED_MODEL = (os.getenv("EMBED_MODEL") or DEFAULT_EMBED_MODEL).strip()


def create_embedder_safe() -> TextEmbedding:
    """
    Crea el modelo de FastEmbed, y si el que viene por env no está soportado,
    hace fallback a DEFAULT_EMBED_MODEL.
    """
    try:
        modelos = TextEmbedding.list_supported_models()
        soportados = {m["model"] for m in modelos}
    except Exception as e:
        logger.warning(f"[FastEmbed] No se pudo listar modelos soportados: {e}")
        modelos = []
        soportados = set()

    if soportados and EMBED_MODEL not in soportados:
        logger.warning(
            f"[FastEmbed] Modelo '{EMBED_MODEL}' NO soportado. "
            f"Haciendo fallback a '{DEFAULT_EMBED_MODEL}'."
        )
        # Si quieres, aquí también puedes imprimir algunos modelos recomendados
        modelos_multilingues = [
            m for m in modelos
            if "multilingual" in m.get("description", "").lower()
            or "paraphrase-multilingual" in m.get("model", "")
        ]
        if modelos_multilingues:
            logger.info("[FastEmbed] Modelos multilingües disponibles:")
            for m in modelos_multilingues:
                logger.info(f"  - {m['model']} (dim={m['dim']})")

        model_name = DEFAULT_EMBED_MODEL
    else:
        model_name = EMBED_MODEL or DEFAULT_EMBED_MODEL

    print(f"🧠 Cargando modelo FastEmbed: {model_name}")
    return TextEmbedding(model_name=model_name)
