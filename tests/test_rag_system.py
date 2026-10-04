import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore

from rag_pinecone.filters import matches_filter
from rag_pinecone.rag_system import RAGSystem


def _doc(cid, text, dept="it", source=None):
    return Document(id=cid, page_content=text, metadata={
        "chunk_id": cid, "source": source or f"{cid.split('#')[0]}.md", "category": f"cat-{dept}",
        "title": "T", "section": "S", "department": dept, "tags": [dept]})


CORPUS = [
    _doc("vpn#0000", "El error VPN-809 se resuelve con TCP 443 en TC-Connect.", "it"),
    _doc("vpn#0001", "La Mesa de Ayuda atiende en el interno 4357.", "it"),
    _doc("gastos#0000", "Los viajes internacionales se solicitan con 15 días de anticipación.", "finanzas"),
    _doc("gastos#0001", "El viático internacional es USD 120 por día.", "finanzas"),
    _doc("vac#0000", "Las vacaciones se solicitan con 30 días de anticipación.", "rrhh"),
    _doc("vac#0001", "Con 5 años de antigüedad corresponden 21 días.", "rrhh"),
    _doc("inc#0000", "Un SEV1 tiene primera respuesta en 15 minutos.", "seguridad"),
    _doc("inc#0001", "El postmortem se hace dentro de 5 días hábiles.", "seguridad"),
]


@pytest.fixture
def rag():
    vs = InMemoryVectorStore(DeterministicFakeEmbedding(size=64))
    vs.add_documents(CORPUS, ids=[d.id for d in CORPUS])
    return RAGSystem(CORPUS, vs, top_k=5, candidate_k=8)


@pytest.mark.parametrize("modo", RAGSystem.MODOS)
def test_recuperar_k_y_sin_duplicados(rag, modo):
    docs = rag.recuperar("¿Qué hago con el error VPN-809?", modo=modo)
    ids = [d.metadata["chunk_id"] for d in docs]
    assert 0 < len(docs) <= 5 and len(ids) == len(set(ids))
    assert len(rag.recuperar("anticipación", k=2, modo=modo)) <= 2


def test_bm25_trae_termino_exacto_primero(rag):
    assert rag.recuperar("VPN-809", modo="bm25")[0].metadata["chunk_id"] == "vpn#0000"


class StubRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return list(self.docs)


class StubRAG(RAGSystem):
    """RAGSystem con retrievers de orden fijo para verificar la fusión RRF."""
    bm25_order: list[Document] = []
    vec_order: list[Document] = []

    def _bm25_for(self, filtros):
        return StubRetriever(docs=self.bm25_order)

    def _vector_retriever(self, filtros):
        return StubRetriever(docs=self.vec_order)


def _stub(weights, bm25_ids, vec_ids):
    by_id = {d.id: d for d in CORPUS}
    r = StubRAG(CORPUS, vectorstore=None, top_k=4, candidate_k=8, weights=weights)
    r.bm25_order = [by_id[i] for i in bm25_ids]
    r.vec_order = [by_id[i] for i in vec_ids]
    return r


BM25_IDS = ["vac#0000", "vpn#0000", "gastos#0000", "inc#0000"]
VEC_IDS = ["vpn#0001", "vpn#0000", "inc#0001", "gastos#0001"]


def _ids(docs):
    return [d.metadata["chunk_id"] for d in docs]


def test_rrf_doc_en_ambos_queda_primero():
    docs = _stub((0.5, 0.5), BM25_IDS, VEC_IDS).recuperar("q")
    assert _ids(docs)[0] == "vpn#0000"            # rank 2 en ambos > rank 1 en uno solo


def test_pesos_extremos_reproducen_cada_retriever():
    assert _ids(_stub((1.0, 0.0), BM25_IDS, VEC_IDS).recuperar("q", k=4)) == BM25_IDS
    assert _ids(_stub((0.0, 1.0), BM25_IDS, VEC_IDS).recuperar("q", k=4)) == VEC_IDS


def test_rrf_no_suma_scores_crudos():
    # Con pesos iguales, el orden depende solo de los rankings: vac#0000 (rank1 bm25) y vpn#0001
    # (rank1 vectorial) empatan en 0.5/61 y quedan detrás de vpn#0000 (0.5/62 * 2).
    docs = _stub((0.5, 0.5), BM25_IDS, VEC_IDS).recuperar("q", k=3)
    assert set(_ids(docs)[1:]) == {"vac#0000", "vpn#0001"}


@pytest.mark.parametrize("modo", RAGSystem.MODOS)
def test_filtros_en_ambos_lados(rag, modo):
    filtros = {"department": "finanzas"}
    docs = rag.recuperar("¿Con cuántos días de anticipación hay que solicitarlo?", filtros=filtros, modo=modo)
    assert docs and all(matches_filter(d.metadata, filtros) for d in docs)
    assert rag.recuperar("algo", filtros={"department": "inexistente"}, modo=modo) == []


def test_obtener_top_k_claves_notebook(rag):
    res = rag.obtener_top_k("VPN-809")
    assert res[0]["rank"] == 1
    for key in ("contenido", "fuente", "categoria", "rank", "chunk_id", "titulo", "seccion", "departamento"):
        assert key in res[0]


def test_diagnosticar_informa_ranks():
    r = _stub((0.5, 0.5), BM25_IDS, VEC_IDS)
    diag = r.diagnosticar("q", k=4)
    first = diag[0]
    assert first["chunk_id"] == "vpn#0000" and first["rank_bm25"] == 2 and first["rank_vectorial"] == 2
    solo_bm25 = next(d for d in diag if d["chunk_id"] == "vac#0000")
    assert solo_bm25["rank_bm25"] == 1 and solo_bm25["rank_vectorial"] is None


def test_validaciones(rag):
    with pytest.raises(ValueError):
        rag.recuperar("   ")
    with pytest.raises(ValueError):
        rag.recuperar("hola", modo="magia")


def test_with_weights_reutiliza_corpus(rag):
    other = rag.with_weights((0.7, 0.3))
    assert other.weights == (0.7, 0.3) and rag.weights == (0.5, 0.5)
    assert other.corpus is rag.corpus and other._vs is rag._vs
