"""Integración con Pinecone real y E5 real (pytest -m integration). Usa un namespace efímero
test-<uuid8> en el índice configurado (INDEX_NAME), que se borra en el teardown aunque el test falle."""
import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def env():
    from rag_pinecone.config import Settings
    from rag_pinecone.embeddings import build_embeddings
    from rag_pinecone.pinecone_index import delete_namespace, get_pinecone_client

    base = Settings.from_env()
    if not base.pinecone_api_key:
        pytest.skip("PINECONE_API_KEY no definida")
    settings = Settings.from_env(namespace=f"test-{uuid.uuid4().hex[:8]}")
    pc = get_pinecone_client(settings)
    embeddings = build_embeddings(settings)
    state = {"settings": settings, "pc": pc, "embeddings": embeddings, "index": None}
    yield state
    if state["index"] is not None:
        delete_namespace(state["index"], settings.namespace)


def test_setup_index_idempotente(env, capsys):
    import setup_index
    s, pc, emb = env["settings"], env["pc"], env["embeddings"]
    assert setup_index.main([], settings=s, pc=pc, embeddings=emb) == 0
    assert setup_index.main([], settings=s, pc=pc, embeddings=emb) == 0
    assert "ya existe" in capsys.readouterr().out
    env["index"] = pc.Index(s.index_name)


def test_prefijos_e5(env):
    emb = env["embeddings"]
    assert emb.embed_query("x") != emb.embed_documents(["x"])[0]


def test_ingesta_completa_e_idempotente(env):
    from rag_pinecone.ingestion import IngestionPipeline
    s, pc, emb = env["settings"], env["pc"], env["embeddings"]
    p = IngestionPipeline.from_client(s, pc, emb)
    env["index"] = p.index
    rep = p.run()
    assert {d.status for d in rep.docs} == {"nuevo"}
    assert rep.namespace_count == rep.total_upserted > 0
    assert max(d.max_tokens for d in rep.docs) <= s.chunk_size
    env["n_vectores"] = rep.namespace_count
    rep2 = IngestionPipeline.from_client(s, pc, emb).run()
    assert rep2.total_upserted == 0 and {d.status for d in rep2.docs} == {"sin_cambios"}
    assert rep2.namespace_count == rep.namespace_count


def test_rag_system_real(env):
    from rag_pinecone.evaluation.runner import chunks_por_fuente_de, evaluar, load_golden_set
    from rag_pinecone.rag_system import RAGSystem
    from rag_pinecone.text_processing import load_catalog
    s = env["settings"]
    rag = RAGSystem.from_settings(s, embeddings=env["embeddings"], pc=env["pc"])
    assert len(rag.corpus) == env["n_vectores"]

    top = rag.recuperar("¿Qué significa el error VPN-809?", modo="bm25")
    assert top[0].metadata["source"] == "guia_acceso_remoto_vpn.md"

    filtrados = rag.recuperar("¿Con cuántos días de anticipación hay que solicitarlo?",
                              filtros={"department": "finanzas"})
    assert filtrados and all(d.metadata["department"] == "finanzas" for d in filtrados)

    golden = load_golden_set(s.golden_set_path, set(load_catalog(s.catalog_path)))
    cpf = chunks_por_fuente_de(rag.corpus)
    rep = evaluar(rag, golden, k=5, modo="hibrido", chunks_por_fuente=cpf)
    assert rep.promedios["hit_rate"] >= 0.9
    nuc = evaluar(rag, golden, k=5, modo="hibrido", chunks_por_fuente=cpf, solo_nucleo=True)
    assert nuc.promedios["recall"] == 1.0
