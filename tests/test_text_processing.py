import logging
import unicodedata

import pytest
from langchain_core.documents import Document

from conftest import long_markdown, make_settings, word_count, write_corpus
from rag_pinecone.exceptions import CatalogError
from rag_pinecone.text_processing import (
    clean_text,
    load_catalog,
    load_documents,
    slugify_doc_id,
    split_documents,
)


def test_clean_text_nfc_bom_y_espacios():
    raw = "﻿Política  de​   vacaciones \r\n\r\n\r\n\r\nSegundo\tpárrafo   "
    out = clean_text(raw)
    assert out == "Política de vacaciones\n\nSegundo párrafo"
    assert unicodedata.is_normalized("NFC", out)


def test_clean_text_desenvuelve_parrafos_pero_respeta_bloques():
    raw = ("# Título\nSigue el texto\ncortado a mano\n\n- item uno\n- item dos\n1. paso\n2. paso\n"
           "> cita\n| a | b |\n| 1 | 2 |\n\n## Sección\nlínea uno\nlínea dos")
    out = clean_text(raw)
    assert "# Título\nSigue el texto cortado a mano" in out
    assert "- item uno\n- item dos\n1. paso\n2. paso\n> cita\n| a | b |\n| 1 | 2 |" in out
    assert "## Sección\nlínea uno línea dos" in out


def test_clean_text_preserva_codigo():
    raw = "Texto\n\n```python\ndef f():\n    return 1\n\n\n\nx = f()\n```\nfin"
    out = clean_text(raw)
    assert "def f():\n    return 1" in out


@pytest.mark.parametrize("raw", [
    "a\nb\nc\n\n\n\nd", "# T\nx\ny\n- l\n  cont\n```\n  code\n```\nz", "﻿  hola \r\n mundo  ",
])
def test_clean_text_idempotente(raw):
    assert clean_text(clean_text(raw)) == clean_text(raw)


def test_slugify():
    assert slugify_doc_id("Política Vacaciones.txt") == "politica_vacaciones"
    assert slugify_doc_id("sdk-referencia_cliente.md") == "sdk-referencia_cliente"


def _settings(tmp_path):
    return make_settings(tmp_path)


def test_chunking_tamano_overlap_y_secciones(tmp_path):
    md = long_markdown("Manual - TechCorp", n_sections=3, words_per_section=450)
    write_corpus(tmp_path / "data", {"manual.md": md, "corto.txt": "Política corta\n\nUn solo párrafo."})
    s = _settings(tmp_path)
    docs = load_documents(s.data_dir, load_catalog(s.catalog_path))
    chunks = split_documents(docs, s, length_function=word_count)
    md_chunks = [c for c in chunks if c.metadata["source"] == "manual.md"]
    assert len(md_chunks) >= 3
    assert all(word_count(c.page_content) <= s.chunk_size for c in chunks)
    # secciones correctas y start_index coherente
    for c in md_chunks:
        assert c.metadata["section"].startswith("Seccion")
        doc_text = docs[[d.metadata["source"] for d in docs].index("manual.md")].page_content
        assert doc_text[c.metadata["start_index"]:].startswith(c.page_content[:30])
    assert {c.metadata["section"] for c in md_chunks} == {"Seccion 1", "Seccion 2", "Seccion 3"}
    # índices por documento
    assert [c.metadata["chunk_index"] for c in md_chunks] == list(range(len(md_chunks)))
    assert all(c.metadata["n_chunks"] == len(md_chunks) for c in md_chunks)
    txt = [c for c in chunks if c.metadata["source"] == "corto.txt"]
    assert len(txt) == 1 and txt[0].metadata["section"] == ""


def test_chunks_consecutivos_comparten_texto(tmp_path):
    words = [f"w{i}" for i in range(1400)]
    text = "Doc\n\n" + " ".join(words)            # un solo párrafo → el splitter corta por espacios con overlap
    write_corpus(tmp_path / "data", {"largo.txt": text})
    s = _settings(tmp_path)
    chunks = split_documents(load_documents(s.data_dir, load_catalog(s.catalog_path)), s,
                             length_function=word_count)
    body = [c for c in chunks if c.page_content.startswith("w")]   # el título queda en su propio chunk
    assert len(body) >= 3
    for a, b in zip(body, body[1:]):
        assert set(a.page_content.split()[-20:]) & set(b.page_content.split()[:100])


def test_dedupe_chunks_identicos(tmp_path, caplog):
    write_corpus(tmp_path / "data", {"a.txt": "Texto repetido\n\nIgual.", "b.txt": "texto  REPETIDO\n\nigual."})
    s = _settings(tmp_path)
    with caplog.at_level(logging.WARNING, logger="rag_pinecone"):
        chunks = split_documents(load_documents(s.data_dir, load_catalog(s.catalog_path)), s,
                                 length_function=word_count)
    assert len(chunks) == 1
    assert "duplicado" in caplog.text


def test_catalogo_estricto(tmp_path):
    write_corpus(tmp_path / "data", {"a.txt": "A\n\nx"})
    (tmp_path / "data" / "extra.txt").write_text("sin catálogo", encoding="utf-8")
    s = _settings(tmp_path)
    with pytest.raises(CatalogError, match="extra.txt"):
        load_documents(s.data_dir, load_catalog(s.catalog_path))
    (tmp_path / "data" / "extra.txt").unlink()
    (tmp_path / "data" / "a.txt").unlink()
    (tmp_path / "data" / "b.txt").write_text("B\n\ny", encoding="utf-8")
    with pytest.raises(CatalogError, match="a.txt"):
        load_documents(s.data_dir, load_catalog(s.catalog_path))


def test_data_vacio(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "data", {})


def test_metadata_del_documento(corpus_dir):
    s = _settings(corpus_dir)
    docs = load_documents(s.data_dir, load_catalog(s.catalog_path))
    d = next(d for d in docs if d.metadata["source"] == "guia_vpn.md")
    assert d.metadata["doc_id"] == "guia_vpn"
    assert d.metadata["title"] == "Guía VPN - TechCorp"
    assert len(d.metadata["content_hash"]) == 16
    assert isinstance(d, Document)
