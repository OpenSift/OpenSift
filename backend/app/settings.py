from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: Optional[str] = None  # ✅ optional now
    gen_model: str = "gpt-4.1-mini"
    embed_model: str = "text-embedding-3-small"

    chroma_dir: str = "./.chroma"
    collection_name: str = "opensift"

settings = Settings()