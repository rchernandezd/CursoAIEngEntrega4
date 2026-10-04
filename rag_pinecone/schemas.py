"""Contratos de datos (anti Schema Drift): un único esquema define los nombres de campos."""
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PIPELINE_VERSION = "1.0"   # subir si cambia la limpieza/chunking → fuerza reindexado

Department = Literal["rrhh", "it", "seguridad", "finanzas"]
DocType = Literal["politica", "procedimiento", "manual", "guia", "documentacion_tecnica"]

PINECONE_METADATA_MAX_BYTES = 40 * 1024
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class CatalogEntry(BaseModel):
    """Metadatos de negocio de un documento (una entrada por archivo en data/catalog.json)."""
    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=2)
    department: Department
    doc_type: DocType
    author: str
    version: str = Field(pattern=r"^\d+\.\d+$")
    effective_year: int = Field(ge=2000, le=2100)
    language: Literal["es"] = "es"
    tags: list[str] = Field(min_length=1)

    @field_validator("tags")
    @classmethod
    def _tags_normalizados(cls, tags: list[str]) -> list[str]:
        malos = [t for t in tags if not _TAG_RE.match(t)]
        if malos:
            raise ValueError(f"tags deben ser minúsculas y sin espacios: {malos}")
        return tags


class ChunkMetadata(BaseModel):
    """ÚNICO esquema de metadata que se escribe en Pinecone. Ningún otro módulo inventa claves."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    # identidad
    chunk_id: str
    doc_id: str
    source: str
    # contenido
    title: str
    section: str
    text: str
    # negocio (desde catalog.json)
    category: str
    department: str
    doc_type: str
    author: str
    version: str
    effective_year: int
    language: str
    tags: list[str]
    # chunking / linaje
    chunk_index: int
    n_chunks: int
    n_tokens: int
    n_chars: int
    start_index: int
    content_hash: str
    chunk_hash: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    pipeline_version: str
    ingested_at: str

    def to_pinecone(self) -> dict:
        """model_dump() validando límites de Pinecone: sin None, valores escalares o list[str],
        JSON < 40 KB por vector."""
        data = self.model_dump()
        for key, value in data.items():
            if value is None:
                raise ValueError(f"metadata '{key}' es None (Pinecone no admite nulos)")
            if isinstance(value, list):
                if not all(isinstance(v, str) for v in value):
                    raise ValueError(f"metadata '{key}' debe ser list[str]")
            elif not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"metadata '{key}' tiene un tipo no soportado: {type(value).__name__}")
        size = len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        if size >= PINECONE_METADATA_MAX_BYTES:
            raise ValueError(f"metadata de {self.chunk_id} ocupa {size} bytes (límite Pinecone 40 KB)")
        return data


FILTERABLE_FIELDS = frozenset({
    "doc_id", "source", "section", "category", "department", "doc_type",
    "author", "version", "effective_year", "language", "tags",
})
LIST_FIELDS = frozenset({"tags"})


class ResultadoRecuperacion(BaseModel):
    rank: int
    chunk_id: str
    fuente: str
    categoria: str
    titulo: str
    seccion: str
    departamento: str
    contenido: str
    rank_bm25: int | None = None          # solo en modo diagnóstico
    rank_vectorial: int | None = None     # solo en modo diagnóstico


class CasoGolden(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^q\d{2}$")
    pregunta: str = Field(min_length=5)
    documentos_esperados: list[str] = Field(min_length=1)
    evidencias: list[str] = Field(default_factory=list)
    tipo: Literal["semantica", "lexica", "multi_doc", "filtrada"]
    filtros: dict | None = None
    nucleo: bool = False      # True = parte del benchmark de 5 preguntas exigido por la consigna

    @model_validator(mode="after")
    def _coherencia(self):
        if (self.tipo == "filtrada") != bool(self.filtros):
            raise ValueError(f"{self.id}: tipo 'filtrada' si y solo si tiene filtros")
        if (self.tipo == "multi_doc") != (len(self.documentos_esperados) >= 2):
            raise ValueError(f"{self.id}: tipo 'multi_doc' si y solo si tiene >= 2 documentos esperados")
        if len(set(self.documentos_esperados)) != len(self.documentos_esperados):
            raise ValueError(f"{self.id}: documentos_esperados con duplicados")
        if self.nucleo and (len(self.documentos_esperados) != 1 or self.filtros):
            raise ValueError(f"{self.id}: un caso 'nucleo' tiene exactamente 1 documento esperado y sin filtros")
        return self
