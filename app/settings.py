from pydantic import BaseModel
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Web
    PORT: int = int(os.getenv("PORT", "10000"))

    # WhatsApp Cloud API
    WA_ACCESS_TOKEN: str = ""
    WA_PHONE_NUMBER_ID: str = ""
    WA_VERIFY_TOKEN: str = ""
    WA_API_VERSION: str = "v21.0"

    # Groq (LLM)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "gemma2-9b-it"

    # Hugging Face (Embeddings vía API)
    HF_API_TOKEN: str = ""
    HF_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_API_URL: str = "https://api-inference.huggingface.co/pipeline/feature-extraction/"

    # Qdrant Cloud
    QDRANT_URL: str = ""   # ej: https://XXXXX.us-east-1-0.aws.cloud.qdrant.io
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "ccp_docs"
    QDRANT_VECTOR_SIZE: int = 384
    QDRANT_DISTANCE: str = "Cosine"

def get_settings() -> Settings:
    return Settings()
