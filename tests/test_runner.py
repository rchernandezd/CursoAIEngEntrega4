import pytest
from langchain_core.documents import Document

import evaluate
from conftest import make_settings
from rag_pinecone.evaluation.runner import (
    barrido_pesos,
    comparar_modos,
    evaluar,
    render_consola,
    render_markdown,
)
from rag_pinecone.schemas import CasoGolden


def _d(source, text="texto"):
    return Document(page_content=text, metadata={"source": source, "chunk_id": f"{source}#0"})


class StubRAG:
    """Respuestas fijas por (pregunta, modo); con filtros solo devuelve docs de 'f.md'."""
    weights = (0.5, 0.5)

    def __init__(self, respuestas):
        self.respuestas = respuestas
        self.corpus = [_d(s) for s in ("a.md", "b.md", "c.md", "f.md")] + [_d("b.md")]

    def recuperar(self, query, *, k=5, filtros=None, modo="hibrido"):
        docs = self.respuestas[(query, modo)]
        if filtros:
            docs = [d for d in docs if d.metadata["source"] == "f.md"]
        return docs[:k]

    def with_weights(self, w):
        clone = StubRAG(self.respuestas)
        clone.weights = w
        return clone


def _golden():
    casos = [CasoGolden(id=f"q0{i}", pregunta=f"pregunta {i}", documentos_esperados=["a.md"], tipo="semantica",
                        evidencias=["texto"], nucleo=True) for i in range(1, 6)]
    casos.append(CasoGolden(id="q06", pregunta="pregunta 6", documentos_esperados=["a.md", "b.md"],
                            tipo="multi_doc"))
    casos.append(CasoGolden(id="q07", pregunta="pregunta 7", documentos_esperados=["f.md"], tipo="filtrada",
                            filtros={"department": "finanzas"}))
    return casos


def _respuestas():
    r = {}
    for modo in ("hibrido", "bm25", "vectorial"):
        for i in range(1, 6):
            r[(f"pregunta {i}", modo)] = [_d("a.md"), _d("c.md")] if modo != "bm25" or i < 4 else [_d("c.md")]
        r[("pregunta 6", modo)] = [_d("c.md"), _d("a.md"), _d("c.md")]
        r[("pregunta 7", modo)] = [_d("c.md"), _d("f.md")]
    return r


CPF = {"a.md": 1, "b.md": 2, "c.md": 1, "f.md": 1}


def test_evaluar_promedios_y_por_tipo():
    rep = evaluar(StubRAG(_respuestas()), _golden(), k=5, modo="hibrido", chunks_por_fuente=CPF)
    assert len(rep.detalle) == 7
    nuc = rep.por_tipo["semantica"]
    assert nuc["recall"] == 1.0 and nuc["precision"] == pytest.approx(0.2) and nuc["mrr"] == 1.0
    assert nuc["precision_normalizada"] == pytest.approx(1.0)       # techo 0.2
    md = rep.por_tipo["multi_doc"]
    assert md["recall"] == pytest.approx(0.5) and md["mrr"] == pytest.approx(0.5)
    f = rep.por_tipo["filtrada"]
    assert f["recall"] == 1.0 and f["mrr"] == 1.0                   # el filtro deja solo f.md
    assert rep.promedios["recall"] == pytest.approx((5 + 0.5 + 1) / 7)


def test_solo_nucleo_evalua_5():
    rep = evaluar(StubRAG(_respuestas()), _golden(), k=5, modo="hibrido", chunks_por_fuente=CPF, solo_nucleo=True)
    assert [f["id"] for f in rep.detalle] == ["q01", "q02", "q03", "q04", "q05"]


def test_render_consola_orden_de_bloques():
    rag = StubRAG(_respuestas())
    comp = comparar_modos(rag, _golden(), k=5, chunks_por_fuente=CPF)
    out = render_consola(comp, barrido_pesos(rag, _golden(), k=5, chunks_por_fuente=CPF))
    i_bench = out.index("BENCHMARK CONSIGNA")
    i_res = out.index("RESUMEN")
    assert i_bench < i_res < out.index("BARRIDO") < out.index("Conclusión breve")
    bench = out[i_bench:i_res]
    assert "Recall@5" in bench and "Precision@5" in bench and "RECALL@5 PROMEDIO" in bench
    assert out.rstrip().splitlines()[-1].startswith("- ")
    assert "bm25" in out[i_res:] and "vectorial" in out[i_res:] and "hibrido" in out[i_res:]


def test_render_markdown_tabla_y_config():
    comp = comparar_modos(StubRAG(_respuestas()), _golden(), k=5, chunks_por_fuente=CPF)
    md = render_markdown(comp, None, {"index_name": "idx", "namespace": "ns"}, "2026-10-04T00:00:00")
    assert "| modo | P@5 |" in md and "| hibrido |" in md
    assert "`idx`" in md and "Lectura de resultados" in md


def test_evaluate_cli_consola_y_strict(tmp_path, capsys):
    s = make_settings(tmp_path)
    rag_ok = StubRAG(_respuestas())
    assert evaluate.main(["--no-report"], settings=s, rag=rag_ok, golden=_golden()) == 0
    out = capsys.readouterr().out
    assert out.index("BENCHMARK CONSIGNA") < out.index("RESUMEN") < out.index("Conclusión breve")
    assert evaluate.main(["--strict", "--no-report"], settings=s, rag=rag_ok, golden=_golden()) == 0
    capsys.readouterr()
    # un caso núcleo sin el documento esperado → recall núcleo < 1 → --strict falla
    malas = _respuestas() | {("pregunta 1", "hibrido"): [_d("c.md")]}
    assert evaluate.main(["--strict", "--no-report"], settings=s, rag=StubRAG(malas), golden=_golden()) == 1
    assert "núcleo" in capsys.readouterr().err


def test_evaluate_cli_escribe_reportes(tmp_path):
    s = make_settings(tmp_path)
    assert evaluate.main([], settings=s, rag=StubRAG(_respuestas()), golden=_golden()) == 0
    assert (tmp_path / "reports" / "eval_report.md").exists()
    assert (tmp_path / "reports" / "eval_results.json").exists()
