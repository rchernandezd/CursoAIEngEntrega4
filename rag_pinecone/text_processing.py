"""Carga, limpieza, chunking por tokens del embedder, sección y dedupe."""
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .exceptions import CatalogError
from .schemas import CatalogEntry

logger = logging.getLogger("rag_pinecone")

SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_BLOCK_START = re.compile(r"^(#|-|\*|>|\||\d+\.|```)")
_H2 = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_for_match(text: str) -> str:
    """Minúsculas, sin tildes, espacios (incl. saltos de línea) colapsados."""
    return re.sub(r"\s+", " ", strip_accents(text).lower()).strip()


def clean_text(text: str) -> str:
    """Normaliza Unicode y espacios, y desenvuelve saltos de línea duros dentro de párrafos.
    Idempotente. Respeta encabezados, listas, citas, tablas y bloques de código."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    out: list[str] = []
    in_code = False
    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(stripped)
            continue
        if in_code:                      # el código conserva su indentación
            out.append(raw.rstrip())
            continue
        line = re.sub(r"[ \t]+", " ", raw).strip()
        stripped = line
        prev = out[-1] if out else ""
        prev_s = prev.strip()
        can_join = (
            out
            and prev_s
            and stripped
            and not prev_s.startswith("#")
            and not prev_s.startswith("```")
            and not prev_s.startswith("|")
            and not _BLOCK_START.match(stripped)
        )
        if can_join:
            out[-1] = f"{prev} {stripped}"
        else:
            out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify_doc_id(filename: str) -> str:
    stem = Path(filename).stem
    slug = strip_accents(stem).lower()
    return re.sub(r"[^a-z0-9_-]", "_", slug)


def extract_title(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line.strip().lstrip("#").strip()
    return ""


def load_catalog(path: Path) -> dict[str, CatalogEntry]:
    if not Path(path).exists():
        raise CatalogError(f"No existe el catálogo {path}. Crealo según el README (data/catalog.json).")
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    try:
        return {name: CatalogEntry(**entry) for name, entry in raw.items()}
    except Exception as exc:  # pydantic.ValidationError
        raise CatalogError(f"catalog.json inválido: {exc}") from exc


def list_data_files(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir)
    return sorted(data_dir.glob("*.txt")) + sorted(data_dir.glob("*.md"))


def load_documents(data_dir: Path, catalog: dict[str, CatalogEntry]) -> list[Document]:
    files = list_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No hay archivos .txt/.md en {data_dir}. Revisá DATA_DIR en .env.")
    names = {f.name for f in files}
    sin_catalogo = sorted(names - set(catalog))
    sin_archivo = sorted(set(catalog) - names)
    if sin_catalogo or sin_archivo:
        raise CatalogError(
            f"data/ y catalog.json no coinciden. Archivos sin entrada en el catálogo: {sin_catalogo}. "
            f"Entradas del catálogo sin archivo: {sin_archivo}. Corregí data/catalog.json."
        )
    docs = []
    for path in files:
        text = clean_text(path.read_text(encoding="utf-8-sig"))
        if not text:
            logger.warning("Archivo vacío tras limpiar, se omite: %s", path.name)
            continue
        entry = catalog[path.name]
        metadata = {
            "source": path.name,
            "doc_id": slugify_doc_id(path.name),
            "title": extract_title(text),
            "content_hash": sha16(text),
            **entry.model_dump(),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def _section_for(chunk_text: str, doc_text: str, start_index: int) -> str:
    m = _H2.search(chunk_text)
    if m:
        return m.group(1).strip()
    section = ""
    for h in _H2.finditer(doc_text):
        if h.start() <= start_index:
            section = h.group(1).strip()
        else:
            break
    return section


def split_documents(docs: list[Document], settings, length_function=None) -> list[Document]:
    """Chunking por tokens del embedder, Markdown-aware, con sección, dedupe e índices por documento."""
    if length_function is None:
        from .embeddings import build_token_counter
        length_function = build_token_counter(settings)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=length_function,
        separators=SEPARATORS,
        add_start_index=True,
    )
    seen: dict[str, str] = {}
    result: list[Document] = []
    for doc in docs:
        doc_chunks = []
        for chunk in splitter.split_documents([doc]):
            text = chunk.page_content.strip()
            if not text:
                continue
            chash = sha16(normalize_for_match(text))
            origin = f"{doc.metadata['source']}@{chunk.metadata.get('start_index', 0)}"
            if chash in seen:
                logger.warning("Chunk duplicado descartado: %s (idéntico a %s)", origin, seen[chash])
                continue
            seen[chash] = origin
            start = chunk.metadata.get("start_index", 0)
            meta = dict(chunk.metadata)
            meta.update(
                section=_section_for(text, doc.page_content, max(start, 0)),
                chunk_hash=chash,
                n_tokens=length_function(text),
                n_chars=len(text),
                start_index=max(start, 0),
            )
            doc_chunks.append(Document(page_content=text, metadata=meta))
        for i, c in enumerate(doc_chunks):
            c.metadata["chunk_index"] = i
            c.metadata["n_chunks"] = len(doc_chunks)
        result.extend(doc_chunks)
    return result
