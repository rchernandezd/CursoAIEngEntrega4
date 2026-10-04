import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from pinecone import ServerlessSpec

import setup_index
from conftest import FakePineconeClient, FakePineconeIndex, make_settings
from rag_pinecone.embeddings import build_embeddings
from rag_pinecone.exceptions import MissingApiKeyError
from rag_pinecone.pinecone_index import get_pinecone_client


@pytest.fixture
def emb():
    return DeterministicFakeEmbedding(size=384)


def test_crea_indice_serverless_si_no_existe(emb, capsys):
    pc = FakePineconeClient()
    rc = setup_index.main([], settings=make_settings(), pc=pc, embeddings=emb)
    assert rc == 0
    assert len(pc.create_calls) == 1
    call = pc.create_calls[0]
    assert call["dimension"] == 384 and call["metric"] == "cosine" and call["name"] == "test-index"
    assert isinstance(call["spec"], ServerlessSpec)
    spec = call["spec"].to_dict() if hasattr(call["spec"], "to_dict") else vars(call["spec"])
    assert "aws" in str(spec) and "us-east-1" in str(spec)
    out = capsys.readouterr().out
    assert "creado" in out and "python ingest.py" in out


def test_idempotente_si_existe_compatible(emb, capsys):
    pc = FakePineconeClient(existing={"test-index": {"dimension": 384, "metric": "cosine",
                                                     "index": FakePineconeIndex(384)}})
    assert setup_index.main([], settings=make_settings(), pc=pc, embeddings=emb) == 0
    assert pc.create_calls == []
    assert "ya existe" in capsys.readouterr().out


def test_dimension_incompatible_exit_1(emb, capsys):
    pc = FakePineconeClient(existing={"test-index": {"dimension": 768, "metric": "cosine",
                                                     "index": FakePineconeIndex(768)}})
    assert setup_index.main([], settings=make_settings(), pc=pc, embeddings=emb) == 1
    err = capsys.readouterr().err
    assert "768" in err and "384" in err


def test_metrica_incompatible_exit_1(emb, capsys):
    pc = FakePineconeClient(existing={"test-index": {"dimension": 384, "metric": "euclidean",
                                                     "index": FakePineconeIndex(384)}})
    assert setup_index.main([], settings=make_settings(), pc=pc, embeddings=emb) == 1
    assert "euclidean" in capsys.readouterr().err


def test_check_nunca_crea(emb, capsys):
    pc = FakePineconeClient()
    assert setup_index.main(["--check"], settings=make_settings(), pc=pc, embeddings=emb) == 1
    assert pc.create_calls == []
    assert "setup_index.py" in capsys.readouterr().err


def test_api_key_vacia():
    with pytest.raises(MissingApiKeyError, match="PINECONE_API_KEY"):
        get_pinecone_client(make_settings(pinecone_api_key=""))


def test_openai_sin_key_no_hace_red(monkeypatch):
    import socket

    def no_network(*a, **k):
        raise AssertionError("no debe haber llamadas de red")
    monkeypatch.setattr(socket, "create_connection", no_network)
    s = make_settings(embedding_provider="openai", embedding_model="text-embedding-3-small",
                      index_name="techcorp-rag-oai-1536", openai_api_key="")
    with pytest.raises(MissingApiKeyError, match="OPENAI_API_KEY"):
        build_embeddings(s)
