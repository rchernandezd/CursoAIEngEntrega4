"""Settings inmutables leídos del entorno (.env), con validaciones de la consigna."""
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("rag_pinecone")

_INDEX_NAME_RE = re.compile(r"^[a-z0-9-]{1,45}$")   # regla de nombres de Pinecone


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str = field(repr=False)        # nunca aparece en logs ni en repr()
    openai_api_key: str = field(repr=False)          # solo se usa si embedding_provider == "openai"
    index_name: str
    namespace: str
    cloud: str
    region: str
    metric: str
    embedding_provider: str                          # "huggingface" (default) | "openai"
    embedding_model: str
    data_dir: Path
    catalog_path: Path
    chunk_size: int
    chunk_overlap: int
    upsert_batch_size: int
    top_k: int
    candidate_k: int
    bm25_weight: float
    vector_weight: float
    rrf_c: int
    golden_set_path: Path
    reports_dir: Path

    def __post_init__(self):
        if not _INDEX_NAME_RE.match(self.index_name):
            raise ValueError("INDEX_NAME: minúsculas, números y '-', máx. 45 caracteres")
        if not self.namespace.strip():
            raise ValueError("PINECONE_NAMESPACE es obligatorio (no usar el namespace por defecto)")
        if self.metric != "cosine":
            raise ValueError("PINECONE_METRIC debe ser 'cosine' (E5 normalizado y OpenAI están optimizados para coseno)")
        if self.embedding_provider not in ("huggingface", "openai"):
            raise ValueError("EMBEDDING_PROVIDER debe ser 'huggingface' u 'openai'")
        if not 500 <= self.chunk_size <= 800:
            raise ValueError("CHUNK_SIZE debe estar entre 500 y 800 tokens (consigna)")
        if not 50 <= self.chunk_overlap <= self.chunk_size // 4:
            raise ValueError("CHUNK_OVERLAP debe estar entre 50 y CHUNK_SIZE/4")
        if not 100 <= self.upsert_batch_size <= 200:
            raise ValueError("UPSERT_BATCH_SIZE debe estar entre 100 y 200 (punto dulce de la consigna)")
        if not 1 <= self.top_k <= 10:
            raise ValueError("TOP_K debe estar entre 1 y 10")
        if not self.top_k <= self.candidate_k <= 50:
            raise ValueError("CANDIDATE_K debe ser >= TOP_K y <= 50")
        if self.bm25_weight < 0 or self.vector_weight < 0 or abs(self.bm25_weight + self.vector_weight - 1) > 1e-6:
            raise ValueError("BM25_WEIGHT y VECTOR_WEIGHT deben ser >= 0 y sumar 1")
        if self.rrf_c < 1:
            raise ValueError("RRF_C debe ser >= 1")

    @property
    def weights(self) -> tuple[float, float]:
        return (self.bm25_weight, self.vector_weight)

    @property
    def embedding_model_id(self) -> str:
        """Identificador '<provider>:<model>' que se guarda en la metadata de cada vector."""
        return f"{self.embedding_provider}:{self.embedding_model}"

    def snapshot(self) -> dict:
        """Configuración relevante para reportes (sin API keys)."""
        return {
            "index_name": self.index_name,
            "namespace": self.namespace,
            "cloud": self.cloud,
            "region": self.region,
            "metric": self.metric,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "weights": list(self.weights),
            "rrf_c": self.rrf_c,
        }

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        load_dotenv()
        provider = overrides.get("embedding_provider") or os.getenv("EMBEDDING_PROVIDER", "huggingface")
        default_model = "text-embedding-3-small" if provider == "openai" else "intfloat/multilingual-e5-small"
        default_index = "techcorp-rag-oai-1536" if provider == "openai" else "techcorp-rag-e5"
        index_name = os.getenv("INDEX_NAME")
        if not index_name and os.getenv("PINECONE_INDEX_NAME"):
            logger.warning("PINECONE_INDEX_NAME está deprecada: usá INDEX_NAME en .env")
            index_name = os.getenv("PINECONE_INDEX_NAME")
        values = dict(
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            index_name=index_name or default_index,
            namespace=os.getenv("PINECONE_NAMESPACE", "techcorp-dev"),
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
            metric=os.getenv("PINECONE_METRIC", "cosine"),
            embedding_provider=provider,
            embedding_model=os.getenv("EMBEDDING_MODEL", default_model),
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            catalog_path=Path(os.getenv("CATALOG_PATH", "data/catalog.json")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "75")),
            upsert_batch_size=int(os.getenv("UPSERT_BATCH_SIZE", "100")),
            top_k=int(os.getenv("TOP_K", "5")),
            candidate_k=int(os.getenv("CANDIDATE_K", "10")),
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.5")),
            vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.5")),
            rrf_c=int(os.getenv("RRF_C", "60")),
            golden_set_path=Path(os.getenv("GOLDEN_SET_PATH", "eval/golden_set.json")),
            reports_dir=Path(os.getenv("REPORTS_DIR", "reports")),
        )
        values.update(overrides)
        return cls(**values)
