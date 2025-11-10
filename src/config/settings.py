# src/config/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path

# Get project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-5-nano"
    
    # SQL Server
    sql_server_host: str
    sql_server_port: int = 1433
    sql_server_database: str
    sql_server_username: str
    sql_server_password: str
    
    # PostgreSQL (Cosmos DB)
    postgres_host: str
    postgres_port: int = 5432
    postgres_database: str
    postgres_username: str
    postgres_password: str
    postgres_sslmode: str = "require"
    
    # Pipeline config
    batch_size: int = 100
    embedding_dimension: int = 1536
    
    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        case_sensitive = False
