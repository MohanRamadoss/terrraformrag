from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    model_name: str = Field(..., min_length=1)
    cloud_provider: str = Field(default="aws")
    search_type: str = Field(default="keyword")

    @validator('query')
    def validate_query(cls, v):
        # Prevent common injection patterns
        forbidden = [';', '&&', '||', '`', '$', '>', '<']
        if any(char in v for char in forbidden):
            raise ValueError("Query contains invalid characters")
        return v

class TerraformCode(BaseModel):
    code: str
    provider: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    validated: bool = False

class AppConfig(BaseModel):
    ollama_api_url: str = Field(..., env='OLLAMA_API_URL')
    priority_models: List[str] = ["granite-code:20b", "deepseek-r1:14b", "codestral:latest"]
    request_timeout: int = Field(default=30)
    max_retries: int = Field(default=3)
    cache_ttl: int = Field(default=3600)
