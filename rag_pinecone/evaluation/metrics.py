"""Métricas de recuperación (funciones puras). La unidad recuperada es el chunk; la relevancia se define a
nivel documento: `retrieved` es la lista ordenada de `source` de los chunks del top-k."""
import math
import re
from collections.abc import Iterable, Sequence

from ..text_processing import normalize_for_match


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """#{i < k : retrieved[i] ∈ relevant} / k — se divide por k aunque se recuperen menos."""
    rel = set(relevant)
    return sum(1 for s in retrieved[:k] if s in rel) / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """|set(retrieved[:k]) ∩ relevant| / |relevant| — no binario con preguntas multi-documento."""
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(retrieved[:k]) & rel) / len(rel)


def f1_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    p, r = precision_at_k(retrieved, rel, k), recall_at_k(retrieved, rel, k)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    return 1.0 if any(s in rel for s in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, s in enumerate(retrieved, 1):
        if s in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int,
              chunks_por_fuente: dict[str, int] | None = None) -> float:
    """Ganancia binaria por chunk relevante. IDCG con el máximo alcanzable: min(k, chunks relevantes en el índice)
    (si no se conoce el conteo de chunks, se asume que cualquier posición puede ser relevante)."""
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(retrieved[:k]) if s in rel)
    if chunks_por_fuente is None:
        ideal = k
    else:
        ideal = min(k, sum(chunks_por_fuente.get(s, 0) for s in rel))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def _contains(haystack_norm: str, needle: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(normalize_for_match(needle)) + r"(?!\w)"
    return re.search(pattern, haystack_norm) is not None


def evidence_recall(chunks: Sequence[tuple[str, str]], relevant: Iterable[str], evidencias: Sequence[str]) -> float:
    """Fracción de evidencias halladas en chunks recuperados cuya fuente es relevante.
    `chunks` = [(source, texto)]. Texto normalizado (sin tildes, minúsculas, espacios colapsados) y con
    límites de palabra ('2 horas' no matchea dentro de '12 horas')."""
    if not evidencias:
        return 1.0
    rel = set(relevant)
    texto = " \n ".join(normalize_for_match(t) for s, t in chunks if s in rel)
    return sum(1 for e in evidencias if _contains(texto, e)) / len(evidencias)


def precision_ceiling(relevant: Iterable[str], chunks_por_fuente: dict[str, int], k: int) -> float:
    """Techo alcanzable de Precision@k: min(k, Σ chunks de los docs relevantes) / k."""
    total = sum(chunks_por_fuente.get(s, 0) for s in set(relevant))
    return min(k, total) / k
