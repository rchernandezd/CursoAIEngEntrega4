"""RAGSystem: recuperador híbrido BM25 + Pinecone fusionado con EnsembleRetriever (RRF ponderada)."""
import json
import logging

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from .embeddings import build_embeddings, get_embedding_dimension
from .filters import matches_filter, normalize_filter
from .pinecone_index import assert_namespace_compatible, ensure_index, get_pinecone_client
from .retrievers import build_bm25_retriever, build_vector_retriever, get_vectorstore, load_corpus_from_pinecone
from .schemas import ResultadoRecuperacion

logger = logging.getLogger("rag_pinecone")


class RAGSystem:
    """Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) fusionados con
    EnsembleRetriever (Reciprocal Rank Fusion ponderada). Nunca suma scores crudos."""

    MODOS = ("hibrido", "bm25", "vectorial")

    def __init__(self, corpus: list[Document], vectorstore, *, top_k: int = 5, candidate_k: int = 10,
                 weights: tuple[float, float] = (0.5, 0.5), rrf_c: int = 60, namespace: str | None = None):
        if not corpus:
            raise ValueError("El corpus está vacío")
        if candidate_k < top_k:
            raise ValueError("candidate_k debe ser >= top_k")
        self.corpus = corpus
        self._vs = vectorstore
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.weights = tuple(weights)
        self.rrf_c = rrf_c
        self.namespace = namespace
        self._bm25_cache: dict[str, object] = {"": build_bm25_retriever(corpus, candidate_k)}

    @classmethod
    def from_settings(cls, settings, *, embeddings=None, pc=None) -> "RAGSystem":
        embeddings = embeddings or build_embeddings(settings)
        pc = pc or get_pinecone_client(settings)
        index, _ = ensure_index(pc, settings, get_embedding_dimension(embeddings), create=False)
        assert_namespace_compatible(index, settings, require_vectors=True)
        corpus = load_corpus_from_pinecone(index, settings.namespace)
        vectorstore = get_vectorstore(settings, index, embeddings)
        logger.info("RAGSystem listo: índice=%s namespace=%s chunks=%d pesos(bm25,vec)=%s top_k=%d candidate_k=%d",
                    settings.index_name, settings.namespace, len(corpus), settings.weights,
                    settings.top_k, settings.candidate_k)
        return cls(corpus, vectorstore, top_k=settings.top_k, candidate_k=settings.candidate_k,
                   weights=settings.weights, rrf_c=settings.rrf_c, namespace=settings.namespace)

    def with_weights(self, weights: tuple[float, float]) -> "RAGSystem":
        """Copia con otros pesos que reutiliza corpus, vectorstore y BM25 ya construidos (barrido)."""
        clone = object.__new__(RAGSystem)
        clone.__dict__.update(self.__dict__)
        clone.weights = tuple(weights)
        return clone

    # --- retrievers -----------------------------------------------------------------------------
    def _bm25_for(self, filtros):
        """BM25 sobre todo el corpus o sobre el subconjunto que cumple el filtro (cacheado).
        Subconjunto vacío → None."""
        norm = normalize_filter(filtros)
        key = json.dumps(norm, sort_keys=True, ensure_ascii=False) if norm else ""
        if key not in self._bm25_cache:
            subset = [d for d in self.corpus if matches_filter(d.metadata, norm)]
            self._bm25_cache[key] = build_bm25_retriever(subset, self.candidate_k) if subset else None
        return self._bm25_cache[key]

    def _vector_retriever(self, filtros):
        return build_vector_retriever(self._vs, self.candidate_k, filtros, self.namespace)

    def _ensemble(self, filtros, bm25=None) -> EnsembleRetriever:
        bm25 = bm25 or self._bm25_for(filtros)
        return EnsembleRetriever(
            retrievers=[bm25, self._vector_retriever(filtros)],
            weights=list(self.weights),
            c=self.rrf_c,
            id_key="chunk_id",      # dedupe por id, no por page_content
        )

    # --- API pública ----------------------------------------------------------------------------
    def recuperar(self, query: str, *, k: int | None = None, filtros: dict | None = None,
                  modo: str = "hibrido") -> list[Document]:
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacía")
        if modo not in self.MODOS:
            raise ValueError(f"modo inválido '{modo}'. Opciones: {self.MODOS}")
        k = k or self.top_k
        bm25 = self._bm25_for(filtros)
        if bm25 is None:            # ningún chunk cumple el filtro
            return []
        if modo == "bm25":
            docs = bm25.invoke(query)
        elif modo == "vectorial":
            docs = self._vector_retriever(filtros).invoke(query)
        else:
            docs = self._ensemble(filtros, bm25).invoke(query)
        return _dedupe(docs)[:k]

    def obtener_top_k(self, query: str, *, k: int | None = None, filtros: dict | None = None,
                      modo: str = "hibrido") -> list[dict]:
        docs = self.recuperar(query, k=k, filtros=filtros, modo=modo)
        return [_to_result(i, d).model_dump() for i, d in enumerate(docs, 1)]

    def diagnosticar(self, query: str, *, k: int | None = None, filtros: dict | None = None) -> list[dict]:
        """Como obtener_top_k(modo='hibrido') + rank_bm25 y rank_vectorial de cada resultado."""
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacía")
        bm25 = self._bm25_for(filtros)
        if bm25 is None:
            return []
        rank_b = {_cid(d): i for i, d in enumerate(bm25.invoke(query), 1)}
        rank_v = {_cid(d): i for i, d in enumerate(self._vector_retriever(filtros).invoke(query), 1)}
        docs = _dedupe(self._ensemble(filtros, bm25).invoke(query))[:k or self.top_k]
        out = []
        for i, d in enumerate(docs, 1):
            r = _to_result(i, d)
            r.rank_bm25 = rank_b.get(_cid(d))
            r.rank_vectorial = rank_v.get(_cid(d))
            out.append(r.model_dump())
        return out


def _cid(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.id or doc.page_content


def _dedupe(docs: list[Document]) -> list[Document]:
    seen, out = set(), []
    for d in docs:
        cid = _cid(d)
        if cid not in seen:
            seen.add(cid)
            out.append(d)
    return out


def _to_result(rank: int, d: Document) -> ResultadoRecuperacion:
    m = d.metadata
    return ResultadoRecuperacion(
        rank=rank,
        chunk_id=_cid(d),
        fuente=m.get("source", ""),
        categoria=m.get("category", ""),
        titulo=m.get("title", ""),
        seccion=m.get("section", ""),
        departamento=m.get("department", ""),
        contenido=d.page_content,
    )
