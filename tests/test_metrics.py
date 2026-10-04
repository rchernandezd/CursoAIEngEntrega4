import math

import pytest

from rag_pinecone.evaluation import metrics as M


def test_precision_divide_por_k():
    assert M.precision_at_k(["a", "x", "a", "y", "z"], {"a"}, 5) == pytest.approx(0.4)
    assert M.precision_at_k(["a", "x", "y"], {"a"}, 5) == pytest.approx(0.2)     # solo 3 recuperados
    assert M.precision_at_k([], {"a"}, 5) == 0.0


def test_recall_multi_doc():
    assert M.recall_at_k(["a", "x", "y", "z", "w"], {"a", "b"}, 5) == pytest.approx(0.5)
    assert M.recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    assert M.recall_at_k(["x", "y", "z", "w", "v", "a"], {"a"}, 5) == 0.0      # fuera del top-k


def test_f1_y_hit_rate():
    assert M.f1_at_k(["x"], {"a"}, 5) == 0.0
    p, r = 0.2, 0.5
    assert M.f1_at_k(["a", "x", "y", "z", "w"], {"a", "b"}, 5) == pytest.approx(2 * p * r / (p + r))
    assert M.hit_rate_at_k(["x", "a"], {"a"}, 5) == 1.0 and M.hit_rate_at_k(["x"], {"a"}, 5) == 0.0


def test_mrr():
    assert M.reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert M.reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg():
    assert M.ndcg_at_k(["a", "a", "x"], {"a"}, 5, {"a": 2}) == pytest.approx(1.0)
    assert M.ndcg_at_k(["x", "a"], {"a"}, 5, {"a": 1}) == pytest.approx(1 / math.log2(3))
    assert M.ndcg_at_k(["x"], {"a"}, 5, {"a": 1}) == 0.0


def test_evidence_recall_normaliza_y_exige_fuente_relevante():
    chunks = [("a.md", "Primera respuesta en 15\nminutos para un SEV1."),
              ("b.md", "Contiene USD 120 pero b no es relevante.")]
    assert M.evidence_recall(chunks, {"a.md"}, ["15 minutos", "sev1"]) == 1.0
    assert M.evidence_recall(chunks, {"a.md"}, ["USD 120"]) == 0.0
    assert M.evidence_recall([("a.md", "Política de más días")], {"a.md"}, ["mas  DIAS"]) == 1.0


def test_evidence_recall_limites_de_palabra():
    assert M.evidence_recall([("a", "la sesión dura 12 horas")], {"a"}, ["2 horas"]) == 0.0
    assert M.evidence_recall([("a", "dentro de 2 horas.")], {"a"}, ["2 horas"]) == 1.0


def test_precision_ceiling():
    assert M.precision_ceiling({"a"}, {"a": 1, "b": 4}, 5) == pytest.approx(0.2)
    assert M.precision_ceiling({"a", "b"}, {"a": 1, "b": 4}, 5) == 1.0
    assert M.precision_ceiling({"b"}, {"b": 9}, 5) == 1.0
