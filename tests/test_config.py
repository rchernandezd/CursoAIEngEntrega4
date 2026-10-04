import pytest

from conftest import make_settings


def test_defaults_validos():
    s = make_settings()
    assert s.weights == (0.5, 0.5)
    assert s.embedding_model_id == "huggingface:intfloat/multilingual-e5-small"


@pytest.mark.parametrize("override", [
    {"chunk_size": 400}, {"chunk_size": 900}, {"chunk_overlap": 40},
    {"upsert_batch_size": 50}, {"upsert_batch_size": 250},
    {"top_k": 5, "candidate_k": 4},
    {"bm25_weight": 0.6, "vector_weight": 0.6},
    {"metric": "euclidean"}, {"namespace": ""}, {"namespace": "   "},
    {"index_name": "Techcorp-RAG"}, {"embedding_provider": "cohere"}, {"rrf_c": 0},
])
def test_rechaza_valores_invalidos(override):
    with pytest.raises(ValueError):
        make_settings(**override)


def test_repr_no_expone_api_keys():
    s = make_settings(pinecone_api_key="pcsk_SECRETO_123", openai_api_key="sk-OTRO_SECRETO")
    assert "SECRETO" not in repr(s)
    assert "SECRETO" not in str(s.snapshot())


def test_index_name_alias(monkeypatch):
    from rag_pinecone.config import Settings
    monkeypatch.setattr("rag_pinecone.config.load_dotenv", lambda: None)
    monkeypatch.delenv("INDEX_NAME", raising=False)
    monkeypatch.setenv("PINECONE_INDEX_NAME", "indice-viejo")
    assert Settings.from_env().index_name == "indice-viejo"
    monkeypatch.setenv("INDEX_NAME", "indice-nuevo")
    assert Settings.from_env().index_name == "indice-nuevo"


def test_default_index_por_proveedor(monkeypatch):
    from rag_pinecone.config import Settings
    monkeypatch.setattr("rag_pinecone.config.load_dotenv", lambda: None)
    for var in ("INDEX_NAME", "PINECONE_INDEX_NAME", "EMBEDDING_MODEL", "EMBEDDING_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    assert Settings.from_env().index_name == "techcorp-rag-e5"
    s = Settings.from_env(embedding_provider="openai")
    assert s.index_name == "techcorp-rag-oai-1536"
    assert s.embedding_model == "text-embedding-3-small"
