import logging
from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document

from conftest import (
    SMALL_CORPUS,
    FakePineconeClient,
    FakePineconeIndex,
    SpyEmbeddings,
    long_markdown,
    make_settings,
    word_count,
    write_corpus,
)
from rag_pinecone.exceptions import (
    EmbeddingModelMismatchError,
    IndexDimensionMismatchError,
    IndexMetricMismatchError,
    IndexNotFoundError,
)
from rag_pinecone.ingestion import IngestionPipeline
from rag_pinecone.schemas import ChunkMetadata

FIXED_NOW = lambda: datetime(2026, 10, 4, tzinfo=timezone.utc)  # noqa: E731


def _pipeline(settings, index=None, emb=None, **kw):
    return IngestionPipeline(settings, index or FakePineconeIndex(), emb or SpyEmbeddings(size=384),
                             length_function=word_count, now=FIXED_NOW, sleep=lambda s: None, **kw)


def _ids(index, ns="test-ns"):
    return sorted(index.data.get(ns, {}))


def test_primera_corrida_todo_nuevo(corpus_dir):
    s = make_settings(corpus_dir)
    p = _pipeline(s)
    rep = p.run()
    assert {d.status for d in rep.docs} == {"nuevo"}
    assert rep.namespace_count == rep.total_upserted == 3
    assert _ids(p.index) == ["guia_vpn#0000", "politica_seguridad#0000", "politica_vacaciones#0000"]
    meta = p.index.data["test-ns"]["guia_vpn#0000"][1]
    assert set(meta) == set(ChunkMetadata.model_fields)
    assert "VPN-809" in meta["text"] and meta["source"] == "guia_vpn.md"
    assert meta["embedding_model"] == "huggingface:intfloat/multilingual-e5-small"


def test_segunda_corrida_sin_cambios_no_embebe(corpus_dir):
    s = make_settings(corpus_dir)
    index, emb = FakePineconeIndex(), SpyEmbeddings(size=384)
    _pipeline(s, index, emb).run()
    calls_before, upserts_before = emb.embed_documents_calls, len(index.upsert_calls)
    rep = _pipeline(s, index, emb).run()
    assert {d.status for d in rep.docs} == {"sin_cambios"}
    assert rep.total_upserted == 0 and rep.total_deleted == 0
    assert emb.embed_documents_calls == calls_before and len(index.upsert_calls) == upserts_before
    assert rep.namespace_count == 3


def test_modificar_un_archivo_solo_reindexa_ese(corpus_dir):
    s = make_settings(corpus_dir)
    index = FakePineconeIndex()
    _pipeline(s, index).run()
    (corpus_dir / "data" / "guia_vpn.md").write_text(SMALL_CORPUS["guia_vpn.md"] + "\n\nNuevo párrafo.",
                                                     encoding="utf-8")
    rep = _pipeline(s, index).run()
    status = {d.source: d.status for d in rep.docs}
    assert status == {"guia_vpn.md": "actualizado", "politica_seguridad.txt": "sin_cambios",
                      "politica_vacaciones.txt": "sin_cambios"}
    assert rep.total_upserted == 1


def test_documento_que_achica_no_deja_huerfanos(tmp_path):
    write_corpus(tmp_path / "data", {"manual.md": long_markdown("Manual", 3, 450)})
    s = make_settings(tmp_path)
    index = FakePineconeIndex()
    _pipeline(s, index).run()
    assert len(_ids(index)) >= 3
    (tmp_path / "data" / "manual.md").write_text("# Manual\n\nAhora es corto.", encoding="utf-8")
    rep = _pipeline(s, index).run()
    assert rep.docs[0].status == "actualizado"
    assert _ids(index) == ["manual#0000"]


def test_prune_y_huerfanos(corpus_dir, caplog):
    s = make_settings(corpus_dir)
    index = FakePineconeIndex()
    _pipeline(s, index).run()
    (corpus_dir / "data" / "guia_vpn.md").unlink()
    write_corpus(corpus_dir / "data", {k: v for k, v in SMALL_CORPUS.items() if k != "guia_vpn.md"})
    with caplog.at_level(logging.WARNING, logger="rag_pinecone"):
        rep = _pipeline(s, index).run()
    assert "guia_vpn#0000" in _ids(index) and rep.orphans == ["guia_vpn"]
    assert "--prune" in caplog.text
    rep = _pipeline(s, index).run(prune=True)
    assert "guia_vpn#0000" not in _ids(index)
    assert any(d.status == "eliminado" and d.doc_id == "guia_vpn" for d in rep.docs)


def test_force_reindexa_todo(corpus_dir):
    s = make_settings(corpus_dir)
    index = FakePineconeIndex()
    _pipeline(s, index).run()
    rep = _pipeline(s, index).run(force=True)
    assert {d.status for d in rep.docs} == {"actualizado"}
    assert rep.total_upserted == 3 and rep.namespace_count == 3


def _synthetic(n: int, s) -> list[ChunkMetadata]:
    p = _pipeline(s)
    items = []
    for i in range(n):
        doc = Document(page_content=f"chunk {i}", metadata={
            "doc_id": f"d{i}", "source": f"d{i}.md", "title": "T", "section": "", "category": "cc",
            "department": "it", "doc_type": "guia", "author": "a", "version": "1.0", "effective_year": 2026,
            "language": "es", "tags": ["x"], "content_hash": "h", "chunk_hash": f"h{i}", "start_index": 0})
        items.append(p.create_metadata(doc, chunk_index=0, n_chunks=1))
    return items


def test_lotes_100_100_50(corpus_dir):
    s = make_settings(corpus_dir, upsert_batch_size=100)
    index = FakePineconeIndex()
    p = _pipeline(s, index)
    assert p.process_and_upsert_batches(_synthetic(250, s)) == 250
    assert index.upsert_calls == [100, 100, 50]


def test_reintento_ante_error_transitorio(corpus_dir):
    s = make_settings(corpus_dir)
    index = FakePineconeIndex(fail_first_upsert=True)
    rep = _pipeline(s, index).run()
    assert rep.total_upserted == 3 and index.upsert_calls == [3]


def test_error_no_transitorio_no_se_reintenta(corpus_dir):
    s = make_settings(corpus_dir)
    index = FakePineconeIndex()

    def boom(vectors, namespace):
        raise ValueError("400 bad request")
    index.upsert = boom
    with pytest.raises(ValueError):
        _pipeline(s, index).run()


@pytest.mark.parametrize("dim,metric,exc", [(768, "cosine", IndexDimensionMismatchError),
                                            (384, "euclidean", IndexMetricMismatchError)])
def test_indice_incompatible(corpus_dir, dim, metric, exc):
    s = make_settings(corpus_dir)
    pc = FakePineconeClient(existing={"test-index": {"dimension": dim, "metric": metric,
                                                     "index": FakePineconeIndex(dim)}})
    with pytest.raises(exc):
        IngestionPipeline.from_client(s, pc, SpyEmbeddings(size=384), length_function=word_count)


def test_indice_inexistente_no_se_crea(corpus_dir):
    pc = FakePineconeClient()
    with pytest.raises(IndexNotFoundError, match="setup_index.py"):
        IngestionPipeline.from_client(make_settings(corpus_dir), pc, SpyEmbeddings(size=384),
                                      length_function=word_count)
    assert pc.create_calls == []


def test_namespace_con_otro_modelo(corpus_dir):
    s = make_settings(corpus_dir)
    index = FakePineconeIndex()
    _pipeline(s, index).run()
    for vid, (vals, meta) in list(index.data["test-ns"].items()):
        index.data["test-ns"][vid] = (vals, {**meta, "embedding_model": "huggingface:otro-modelo"})
    with pytest.raises(EmbeddingModelMismatchError, match="--force"):
        _pipeline(s, index).run()
    rep = _pipeline(s, index).run(force=True)
    assert rep.namespace_count == 3
    assert {m["embedding_model"] for _, m in index.data["test-ns"].values()} == {s.embedding_model_id}


def test_dry_run_no_escribe(corpus_dir):
    s = make_settings(corpus_dir)
    index = FakePineconeIndex()
    emb = SpyEmbeddings(size=384)
    rep = _pipeline(s, index, emb).run(dry_run=True)
    assert rep.dry_run and rep.total_upserted == 3
    assert index.upsert_calls == [] and index.delete_calls == [] and emb.embed_documents_calls == 0
    _pipeline(s, index).run()
    (corpus_dir / "data" / "guia_vpn.md").write_text("# Guía VPN\n\nCambiado.", encoding="utf-8")
    rep = _pipeline(s, index).run(dry_run=True, prune=True)
    assert index.delete_calls == []
    assert {d.source: d.status for d in rep.docs}["guia_vpn.md"] == "actualizado"
