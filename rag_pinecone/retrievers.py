"""Retrievers: corpus léxico reconstruido desde Pinecone, BM25 con preprocesamiento en español y vectorial."""
import re

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .exceptions import EmptyNamespaceError
from .filters import matches_filter, to_pinecone_filter
from .pinecone_index import fetch_metadata, list_ids
from .text_processing import strip_accents

STOPWORDS_ES: frozenset[str] = frozenset("""
a al algo algun alguna algunas alguno algunos ante antes asi aun cada como con contra cual cuales cualquier
cuando cuanta cuantas cuanto cuantos de del desde donde dos e el ella ellas ello ellos en entre era es esa
esas ese eso esos esta estan estas este esto estos fue fueron ha hacia han hasta hay la las le les lo los mas
me mi mis mucho muy ni no nos o otra otro para pero poco por porque puede pueden puedo que quien se segun sea
ser si sin sobre son su sus tambien tan tanto te tener tengo tiene tienen todo todos tu tus un una unas uno
unos y ya yo debo hacer hace hago
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def bm25_preprocess(text: str) -> list[str]:
    """Sin tildes → minúsculas → tokens alfanuméricos (con '-', '_' o '.' internos) → sin stopwords.
    '(2FA)'→'2fa', 'VPN-809'→'vpn-809', 'días'→'dias', 'vpn2.techcorp.internal' queda como un token."""
    norm = strip_accents(text).lower()
    return [t for t in _TOKEN_RE.findall(norm) if t not in STOPWORDS_ES]


def load_corpus_from_pinecone(index, namespace: str) -> list[Document]:
    """Fuente única de verdad: list_ids + fetch en lotes → Document(id=chunk_id, page_content=text)."""
    ids = list_ids(index, namespace)
    if not ids:
        raise EmptyNamespaceError(f"El namespace '{namespace}' no tiene vectores. Ejecutá: python ingest.py")
    metas = fetch_metadata(index, namespace, ids)
    corpus = []
    for vid, meta in metas.items():
        meta = dict(meta)
        text = meta.pop("text", "")
        meta.setdefault("chunk_id", vid)
        for k in ("chunk_index", "n_chunks", "n_tokens", "n_chars", "start_index", "effective_year",
                  "chunk_size", "chunk_overlap"):
            if isinstance(meta.get(k), float) and meta[k].is_integer():
                meta[k] = int(meta[k])           # Pinecone devuelve números como float
        corpus.append(Document(id=vid, page_content=text, metadata=meta))
    corpus.sort(key=lambda d: (d.metadata.get("source", ""), d.metadata.get("chunk_index", 0)))
    return corpus


def build_bm25_retriever(corpus: list[Document], k: int) -> BM25Retriever:
    return BM25Retriever.from_documents(corpus, preprocess_func=bm25_preprocess, k=k)


def get_vectorstore(settings, index, embeddings):
    from langchain_pinecone import PineconeVectorStore
    return PineconeVectorStore(index=index, embedding=embeddings, text_key="text", namespace=settings.namespace)


def build_vector_retriever(vectorstore, k: int, filtros: dict | None = None, namespace: str | None = None):
    """Pinecone: filtro en la sintaxis de Pinecone + namespace explícito.
    Otros VectorStore (InMemoryVectorStore en tests): filtro como predicado Python equivalente."""
    try:
        from langchain_pinecone import PineconeVectorStore
        is_pinecone = isinstance(vectorstore, PineconeVectorStore)
    except ImportError:  # pragma: no cover
        is_pinecone = False
    search_kwargs: dict = {"k": k}
    if is_pinecone:
        if namespace:
            search_kwargs["namespace"] = namespace
        pf = to_pinecone_filter(filtros)
        if pf:
            search_kwargs["filter"] = pf
    elif filtros:
        search_kwargs["filter"] = lambda doc: matches_filter(doc.metadata, filtros)
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
