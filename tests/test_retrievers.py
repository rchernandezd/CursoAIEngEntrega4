import pytest
from langchain_core.documents import Document

from conftest import FakePineconeIndex, SpyEmbeddings, make_settings, word_count
from rag_pinecone.exceptions import EmptyNamespaceError
from rag_pinecone.ingestion import IngestionPipeline
from rag_pinecone.retrievers import bm25_preprocess, build_bm25_retriever, load_corpus_from_pinecone


def test_bm25_preprocess():
    assert bm25_preprocess("(2FA)") == ["2fa"]
    assert "dias" in bm25_preprocess("días")
    assert bm25_preprocess("Error VPN-809") == ["error", "vpn-809"]
    assert bm25_preprocess("SEV3") == ["sev3"]
    assert bm25_preprocess("usar vpn2.techcorp.internal ahora") == ["usar", "vpn2.techcorp.internal", "ahora"]
    assert bm25_preprocess("¿Qué es el de la para los?") == []


def test_bm25_encuentra_sigla_entre_parentesis():
    docs = [Document(page_content=t, metadata={"chunk_id": str(i)}) for i, t in enumerate([
        "Usar autenticación de dos factores (2FA) en el correo.",
        "Las vacaciones se piden con anticipación.",
        "El teletrabajo es híbrido.",
    ])]
    top = build_bm25_retriever(docs, k=1).invoke("¿Dónde es obligatorio el 2FA?")
    assert top[0].metadata["chunk_id"] == "0"


def test_load_corpus_from_pinecone(corpus_dir):
    s = make_settings(corpus_dir)
    index = FakePineconeIndex()
    IngestionPipeline(s, index, SpyEmbeddings(size=384), length_function=word_count).run()
    corpus = load_corpus_from_pinecone(index, s.namespace)
    assert [d.metadata["source"] for d in corpus] == ["guia_vpn.md", "politica_seguridad.txt",
                                                      "politica_vacaciones.txt"]
    d = corpus[0]
    assert "VPN-809" in d.page_content and "text" not in d.metadata
    assert d.id == d.metadata["chunk_id"] == "guia_vpn#0000"
    assert d.metadata["department"] == "rrhh"


def test_load_corpus_namespace_vacio():
    with pytest.raises(EmptyNamespaceError, match="ingest.py"):
        load_corpus_from_pinecone(FakePineconeIndex(), "vacio")
