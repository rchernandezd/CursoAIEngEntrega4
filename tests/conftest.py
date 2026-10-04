import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_pinecone.config import Settings
from rag_pinecone.filters import matches_filter

REPO = Path(__file__).resolve().parent.parent

word_count = lambda text: len(text.split())   # noqa: E731 — length_function offline (sin tokenizer)


class TransientError(Exception):
    status = 503


class FakePineconeIndex:
    """Superficie de pinecone.Index usada por el proyecto, en memoria."""

    def __init__(self, dimension: int = 384, *, fail_first_upsert: bool = False):
        self.dimension = dimension
        self.data: dict[str, dict[str, tuple[list[float], dict]]] = {}
        self.upsert_calls: list[int] = []
        self.delete_calls: list[dict] = []
        self.fail_first_upsert = fail_first_upsert

    def upsert(self, vectors, namespace):
        if self.fail_first_upsert:
            self.fail_first_upsert = False
            raise TransientError("503 Service Unavailable")
        self.upsert_calls.append(len(vectors))
        ns = self.data.setdefault(namespace, {})
        for v in vectors:
            assert len(v["values"]) == self.dimension
            ns[v["id"]] = (list(v["values"]), dict(v["metadata"]))
        return {"upserted_count": len(vectors)}

    def list(self, prefix=None, namespace="", limit=None):
        ids = sorted(i for i in self.data.get(namespace, {}) if prefix is None or i.startswith(prefix))
        page = limit or 100
        for i in range(0, len(ids), page):
            yield ids[i:i + page]

    def fetch(self, ids, namespace=""):
        ns = self.data.get(namespace, {})
        return SimpleNamespace(vectors={i: SimpleNamespace(id=i, metadata=dict(ns[i][1]))
                                        for i in ids if i in ns})

    def delete(self, ids=None, delete_all=False, namespace=""):
        self.delete_calls.append({"ids": ids, "delete_all": delete_all, "namespace": namespace})
        ns = self.data.get(namespace, {})
        if delete_all:
            ns.clear()
        for i in ids or []:
            ns.pop(i, None)

    def describe_index_stats(self, **_):
        nss = {ns: {"vector_count": len(v)} for ns, v in self.data.items() if v}
        return {"namespaces": nss, "dimension": self.dimension,
                "total_vector_count": sum(x["vector_count"] for x in nss.values())}

    def query(self, vector, top_k, namespace="", filter=None, include_metadata=True, **_):
        def cos(a, b):
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
            return sum(x * y for x, y in zip(a, b)) / (na * nb or 1)
        items = [(vid, cos(vector, vals), meta) for vid, (vals, meta) in self.data.get(namespace, {}).items()
                 if matches_filter(meta, filter)]
        items.sort(key=lambda t: -t[1])
        return {"matches": [{"id": i, "score": s, "metadata": m} for i, s, m in items[:top_k]]}


class FakePineconeClient:
    def __init__(self, *, existing: dict | None = None):
        """existing = {name: {"dimension": 384, "metric": "cosine", "index": FakePineconeIndex}}"""
        self.indexes = dict(existing or {})
        self.create_calls: list[dict] = []

    def has_index(self, name):
        return name in self.indexes

    def describe_index(self, name):
        info = self.indexes[name]
        return {"name": name, "dimension": info["dimension"], "metric": info["metric"],
                "status": {"ready": True}, "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}}}

    def create_index(self, name, dimension, metric, spec):
        self.create_calls.append({"name": name, "dimension": dimension, "metric": metric, "spec": spec})
        self.indexes[name] = {"dimension": dimension, "metric": metric, "index": FakePineconeIndex(dimension)}

    def Index(self, name):  # noqa: N802 — misma API que pinecone.Pinecone
        return self.indexes[name]["index"]


class SpyEmbeddings(DeterministicFakeEmbedding):
    """Embeddings deterministas que cuentan llamadas (para verificar que no se re-embebe)."""
    embed_documents_calls: int = 0
    embedded_texts: int = 0

    def embed_documents(self, texts):
        self.embed_documents_calls += 1
        self.embedded_texts += len(texts)
        return super().embed_documents(texts)


def make_settings(tmp_path: Path | None = None, **overrides) -> Settings:
    base = tmp_path or REPO
    values = dict(
        pinecone_api_key="", openai_api_key="", index_name="test-index", namespace="test-ns",
        cloud="aws", region="us-east-1", metric="cosine", embedding_provider="huggingface",
        embedding_model="intfloat/multilingual-e5-small",
        data_dir=base / "data", catalog_path=base / "data" / "catalog.json",
        chunk_size=500, chunk_overlap=75, upsert_batch_size=100, top_k=5, candidate_k=10,
        bm25_weight=0.5, vector_weight=0.5, rrf_c=60,
        golden_set_path=base / "eval" / "golden_set.json", reports_dir=base / "reports",
    )
    values.update(overrides)
    return Settings.from_env(**values)


CATALOG_BASE = {"department": "rrhh", "doc_type": "politica", "author": "RRHH", "version": "1.0",
                "effective_year": 2025}


def write_corpus(data_dir: Path, docs: dict[str, str], catalog: dict | None = None) -> None:
    """Escribe documentos y un catalog.json coherente (category por documento)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        (data_dir / name).write_text(text, encoding="utf-8")
    if catalog is None:
        catalog = {name: {**CATALOG_BASE, "category": "cat " + Path(name).stem.replace("_", " "),
                          "tags": [Path(name).stem.split("_")[0]]} for name in docs}
    (data_dir / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")


def long_markdown(title: str, n_sections: int = 3, words_per_section: int = 450, seed: str = "x") -> str:
    parts = [f"# {title}", ""]
    for s in range(n_sections):
        parts.append(f"## Seccion {s + 1}")
        parts.append("")
        words = [f"{seed}{s}p{i}" for i in range(words_per_section)]
        # párrafos de 50 palabras
        for i in range(0, len(words), 50):
            parts.append(" ".join(words[i:i + 50]) + ".")
            parts.append("")
    return "\n".join(parts)


SMALL_CORPUS = {
    "politica_vacaciones.txt": "Política de Vacaciones - TechCorp\n\nLos empleados con 5 años o más acceden a "
                               "21 días corridos de vacaciones por año.",
    "politica_seguridad.txt": "Política de Seguridad - TechCorp\n\nSe exige autenticación de dos factores (2FA) "
                              "en el correo. El perfil MDM es obligatorio para BYOD.",
    "guia_vpn.md": "# Guía VPN - TechCorp\n\n## Errores\n\nEl error VPN-809 se resuelve activando TCP 443.\n\n"
                   "## Soporte\n\nLlamar al interno 4357.",
}


@pytest.fixture
def corpus_dir(tmp_path):
    write_corpus(tmp_path / "data", SMALL_CORPUS)
    return tmp_path


@pytest.fixture
def fake_embeddings():
    return SpyEmbeddings(size=384)
