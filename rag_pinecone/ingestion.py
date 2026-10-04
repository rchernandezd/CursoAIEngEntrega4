"""Pipeline de ingesta masiva: chunking → ChunkMetadata → embeddings → batch upsert, con ciclo de vida
por documento (sin_cambios / nuevo / actualizado / eliminado) para no reindexar lo que no cambió."""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from langchain_core.documents import Document

from .embeddings import assert_chunk_fits, build_token_counter, get_embedding_dimension
from .exceptions import EmbeddingModelMismatchError
from .pinecone_index import (
    assert_namespace_compatible,
    delete_ids,
    delete_namespace,
    ensure_index,
    fetch_metadata,
    list_ids,
    namespace_vector_count,
    wait_for_count,
)
from .schemas import PIPELINE_VERSION, ChunkMetadata
from .text_processing import load_catalog, load_documents, split_documents

logger = logging.getLogger("rag_pinecone")

ID_SEP = "#"
_LINEAGE_KEYS = ("content_hash", "embedding_model", "chunk_size", "chunk_overlap", "pipeline_version")


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}{ID_SEP}{chunk_index:04d}"


def doc_id_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split(ID_SEP, 1)[0]


def is_transient(exc: Exception) -> bool:
    """429 / 5xx / timeouts / errores de conexión."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    if status is not None:
        return status == 429 or 500 <= status < 600
    text = str(exc).lower()
    return any(s in text for s in ("429", "too many requests", "timed out", "timeout", "503", "502", "504"))


@dataclass
class DocIngestReport:
    source: str
    doc_id: str
    status: Literal["nuevo", "actualizado", "sin_cambios", "eliminado"]
    n_chunks: int
    max_tokens: int
    upserted: int = 0
    deleted: int = 0


@dataclass
class IngestReport:
    docs: list[DocIngestReport]
    total_upserted: int
    total_deleted: int
    namespace_count: int
    elapsed_s: float
    dry_run: bool
    orphans: list[str] = field(default_factory=list)   # doc_ids en el índice sin archivo (sin --prune)


class IngestionPipeline:
    def __init__(self, settings, index, embeddings, *, length_function=None, now=None,
                 sleep=time.sleep, retry_delays=(1, 2, 4)):
        self.settings = settings
        self.index = index
        self.embeddings = embeddings
        self._custom_length = length_function is not None
        self.length_function = length_function or build_token_counter(settings)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep
        self._retry_delays = tuple(retry_delays)

    @classmethod
    def from_client(cls, settings, pc, embeddings, **kwargs) -> "IngestionPipeline":
        """Valida el índice existente (dimensión derivada del modelo, métrica). Nunca crea índices."""
        dimension = get_embedding_dimension(embeddings)
        index, _ = ensure_index(pc, settings, dimension, create=False)
        return cls(settings, index, embeddings, **kwargs)

    # --- metadata -------------------------------------------------------------------------------
    def create_metadata(self, chunk: Document, *, chunk_index: int, n_chunks: int,
                        ingested_at: str | None = None) -> ChunkMetadata:
        """Construye la metadata validada (Pydantic) de un chunk. ÚNICO lugar donde se arma metadata."""
        m = chunk.metadata
        s = self.settings
        return ChunkMetadata(
            chunk_id=make_chunk_id(m["doc_id"], chunk_index),
            doc_id=m["doc_id"],
            source=m["source"],
            title=m["title"],
            section=m.get("section", ""),
            text=chunk.page_content,
            category=m["category"],
            department=m["department"],
            doc_type=m["doc_type"],
            author=m["author"],
            version=m["version"],
            effective_year=m["effective_year"],
            language=m["language"],
            tags=list(m["tags"]),
            chunk_index=chunk_index,
            n_chunks=n_chunks,
            n_tokens=m.get("n_tokens", self.length_function(chunk.page_content)),
            n_chars=len(chunk.page_content),
            start_index=m.get("start_index", 0),
            content_hash=m["content_hash"],
            chunk_hash=m["chunk_hash"],
            embedding_model=s.embedding_model_id,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            pipeline_version=PIPELINE_VERSION,
            ingested_at=ingested_at or self._now().isoformat(timespec="seconds"),
        )

    # --- upsert ---------------------------------------------------------------------------------
    def _upsert_with_retry(self, vectors: list[dict]) -> None:
        attempts = len(self._retry_delays)
        for attempt in range(attempts):
            try:
                self.index.upsert(vectors=vectors, namespace=self.settings.namespace)
                return
            except Exception as exc:
                if attempt == attempts - 1 or not is_transient(exc):
                    raise
                delay = self._retry_delays[attempt]
                logger.warning("Upsert falló (%s); reintento %d/%d en %ss", exc, attempt + 1, attempts - 1, delay)
                self._sleep(delay)

    def process_and_upsert_batches(self, items: list[ChunkMetadata]) -> int:
        bs = self.settings.upsert_batch_size
        total = 0
        n_batches = (len(items) + bs - 1) // bs
        for b, start in enumerate(range(0, len(items), bs), 1):
            batch = items[start:start + bs]
            t0 = time.perf_counter()
            values = self.embeddings.embed_documents([m.text for m in batch])
            vectors = [{"id": m.chunk_id, "values": list(v), "metadata": m.to_pinecone()}
                       for m, v in zip(batch, values)]
            self._upsert_with_retry(vectors)
            total += len(vectors)
            logger.info("Lote %d/%d: %d vectores en %.0f ms", b, n_batches, len(vectors),
                        (time.perf_counter() - t0) * 1000)
        return total

    # --- ciclo de vida --------------------------------------------------------------------------
    def _is_unchanged(self, existing: list[str], first_chunk: Document) -> bool:
        meta = fetch_metadata(self.index, self.settings.namespace, [sorted(existing)[0]])
        stored = next(iter(meta.values()), {})
        expected = {
            "content_hash": first_chunk.metadata["content_hash"],
            "embedding_model": self.settings.embedding_model_id,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "pipeline_version": PIPELINE_VERSION,
        }
        same = all(_eq(stored.get(k), expected[k]) for k in _LINEAGE_KEYS)
        return same and _eq(stored.get("n_chunks"), len(existing))

    def run(self, *, force: bool = False, prune: bool = False, dry_run: bool = False) -> IngestReport:
        t0 = time.perf_counter()
        s = self.settings
        ns = s.namespace
        if not self._custom_length:
            assert_chunk_fits(s)
        catalog = load_catalog(s.catalog_path)
        docs = load_documents(s.data_dir, catalog)
        chunks = split_documents(docs, s, length_function=self.length_function)
        by_doc: dict[str, list[Document]] = {}
        for c in chunks:
            by_doc.setdefault(c.metadata["doc_id"], []).append(c)

        initial_count = namespace_vector_count(self.index, ns)
        try:
            assert_namespace_compatible(self.index, s, require_vectors=False)
        except EmbeddingModelMismatchError:
            if not force:
                raise
            logger.warning("Namespace '%s' con otro modelo de embeddings: --force lo vacía y reindexa todo", ns)
            if not dry_run:
                delete_namespace(self.index, ns)
                initial_count = 0

        reports: list[DocIngestReport] = []
        to_upsert: list[ChunkMetadata] = []
        total_deleted = 0
        ingested_at = self._now().isoformat(timespec="seconds")
        for doc in docs:
            doc_id = doc.metadata["doc_id"]
            doc_chunks = by_doc.get(doc_id, [])
            max_tokens = max((c.metadata["n_tokens"] for c in doc_chunks), default=0)
            existing = list_ids(self.index, ns, prefix=f"{doc_id}{ID_SEP}")
            if existing and not force and doc_chunks and self._is_unchanged(existing, doc_chunks[0]):
                reports.append(DocIngestReport(doc.metadata["source"], doc_id, "sin_cambios",
                                               len(doc_chunks), max_tokens))
                continue
            deleted = 0
            if existing and not dry_run:
                deleted = delete_ids(self.index, ns, existing)
            elif existing:
                deleted = len(existing)
            total_deleted += deleted
            for c in doc_chunks:
                to_upsert.append(self.create_metadata(
                    c, chunk_index=c.metadata["chunk_index"], n_chunks=c.metadata["n_chunks"],
                    ingested_at=ingested_at))
            reports.append(DocIngestReport(doc.metadata["source"], doc_id,
                                           "actualizado" if existing else "nuevo",
                                           len(doc_chunks), max_tokens, upserted=len(doc_chunks),
                                           deleted=deleted))

        # Huérfanos: vectores de documentos que ya no están en data/
        known = {d.metadata["doc_id"] for d in docs}
        orphan_ids: dict[str, list[str]] = {}
        for vid in list_ids(self.index, ns):
            did = doc_id_from_chunk_id(vid)
            if did not in known:
                orphan_ids.setdefault(did, []).append(vid)
        orphans = sorted(orphan_ids)
        if orphan_ids:
            if prune:
                for did, ids in sorted(orphan_ids.items()):
                    n = len(ids) if dry_run else delete_ids(self.index, ns, ids)
                    total_deleted += n
                    reports.append(DocIngestReport(did, did, "eliminado", 0, 0, deleted=n))
                orphans = []
            else:
                logger.warning("Documentos en el índice que ya no están en data/: %s "
                               "(usá python ingest.py --prune para borrarlos)", orphans)

        total_upserted = 0
        if dry_run:
            total_upserted = len(to_upsert)
            final_count = initial_count
        else:
            if to_upsert:
                total_upserted = self.process_and_upsert_batches(to_upsert)
            expected = initial_count - total_deleted + total_upserted
            if total_upserted or total_deleted:
                wait_for_count(self.index, ns, expected)
            final_count = namespace_vector_count(self.index, ns)

        report = IngestReport(reports, total_upserted, total_deleted, final_count,
                              time.perf_counter() - t0, dry_run, orphans)
        counts: dict[str, int] = {}
        for r in reports:
            counts[r.status] = counts.get(r.status, 0) + 1
        logger.info("Ingesta %s: %s | subidos=%d borrados=%d | vectores en '%s'=%d | %.1f s",
                    "(dry-run)" if dry_run else "completa", counts, total_upserted, total_deleted,
                    ns, final_count, report.elapsed_s)
        return report


def _eq(a, b) -> bool:
    """Compara tolerando que Pinecone devuelva números como float (500.0 == 500)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b
