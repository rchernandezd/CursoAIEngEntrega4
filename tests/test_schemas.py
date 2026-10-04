import json

import pytest
from pydantic import ValidationError

from rag_pinecone.evaluation.runner import load_golden_set
from rag_pinecone.schemas import CasoGolden, CatalogEntry, ChunkMetadata


def _meta(**over):
    base = dict(
        chunk_id="doc#0000", doc_id="doc", source="doc.md", title="Doc", section="", text="hola",
        category="c", department="it", doc_type="guia", author="a", version="1.0", effective_year=2025,
        language="es", tags=["x"], chunk_index=0, n_chunks=1, n_tokens=1, n_chars=4, start_index=0,
        content_hash="h", chunk_hash="h", embedding_model="huggingface:m", chunk_size=500, chunk_overlap=75,
        pipeline_version="1.0", ingested_at="2026-01-01T00:00:00+00:00",
    )
    base.update(over)
    return base


def test_chunk_metadata_rechaza_claves_extra():
    with pytest.raises(ValidationError):
        ChunkMetadata(**_meta(categoria="x"))


def test_chunk_metadata_rechaza_none():
    with pytest.raises(ValidationError):
        ChunkMetadata(**_meta(section=None))


def test_to_pinecone_incluye_text_y_respeta_40kb():
    m = ChunkMetadata(**_meta())
    d = m.to_pinecone()
    assert d["text"] == "hola" and set(d) == set(ChunkMetadata.model_fields)
    with pytest.raises(ValueError, match="40 KB"):
        ChunkMetadata(**_meta(text="a" * 41_000)).to_pinecone()


def test_catalog_entry_validaciones():
    ok = dict(category="vpn", department="it", doc_type="guia", author="IT", version="1.2",
              effective_year=2026, tags=["vpn"])
    CatalogEntry(**ok)
    with pytest.raises(ValidationError):
        CatalogEntry(**{**ok, "department": "legal"})
    with pytest.raises(ValidationError):
        CatalogEntry(**{**ok, "version": "v2"})
    with pytest.raises(ValidationError):
        CatalogEntry(**{**ok, "tags": ["Con Espacios"]})


def _caso(**over):
    base = dict(id="q01", pregunta="¿Pregunta de prueba?", documentos_esperados=["a.md"], tipo="semantica")
    base.update(over)
    return base


@pytest.mark.parametrize("over", [
    {"tipo": "filtrada"},                                                  # filtrada sin filtros
    {"filtros": {"department": "it"}},                                     # filtros sin tipo filtrada
    {"tipo": "multi_doc"},                                                 # multi_doc con 1 doc
    {"documentos_esperados": ["a.md", "b.md"]},                            # 2 docs sin multi_doc
    {"tipo": "multi_doc", "documentos_esperados": ["a.md", "a.md"]},      # duplicados
    {"tipo": "multi_doc", "documentos_esperados": ["a.md", "b.md"], "nucleo": True},
    {"tipo": "filtrada", "filtros": {"department": "it"}, "nucleo": True},
    {"id": "Q1"},
])
def test_caso_golden_incoherente(over):
    with pytest.raises(ValidationError):
        CasoGolden(**_caso(**over))


def _golden(tmp_path, casos):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(casos), encoding="utf-8")
    return p


def test_load_golden_set_documento_inexistente(tmp_path):
    casos = [_caso(id=f"q0{i}", nucleo=True) for i in range(1, 6)]
    casos[0]["documentos_esperados"] = ["no_existe.md"]
    with pytest.raises(ValueError, match="no_existe.md"):
        load_golden_set(_golden(tmp_path, casos), {"a.md"})


@pytest.mark.parametrize("n_nucleo", [4, 6])
def test_load_golden_set_exige_5_nucleo(tmp_path, n_nucleo):
    casos = [_caso(id=f"q{i:02d}", nucleo=i <= n_nucleo) for i in range(1, 8)]
    with pytest.raises(ValueError, match="exactamente 5"):
        load_golden_set(_golden(tmp_path, casos), {"a.md"})


def test_golden_set_real_es_valido():
    from conftest import REPO
    from rag_pinecone.text_processing import load_catalog
    casos = load_golden_set(REPO / "eval" / "golden_set.json", set(load_catalog(REPO / "data" / "catalog.json")))
    assert len(casos) == 26
    assert [c.id for c in casos if c.nucleo] == ["q01", "q08", "q10", "q23", "q24"]
