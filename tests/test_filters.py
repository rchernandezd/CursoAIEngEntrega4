import itertools

import pytest

from rag_pinecone.exceptions import UnknownFilterFieldError
from rag_pinecone.filters import matches_filter, normalize_filter, parse_cli_filters, to_pinecone_filter


def test_traduccion_a_pinecone():
    assert to_pinecone_filter({"department": "finanzas", "effective_year": {"$gte": 2025}}) == {
        "department": {"$eq": "finanzas"}, "effective_year": {"$gte": 2025}}
    assert to_pinecone_filter({"source": ["a.md", "b.md"]}) == {"source": {"$in": ["a.md", "b.md"]}}
    assert to_pinecone_filter({"tags": "vpn"}) == {"tags": {"$in": ["vpn"]}}
    assert to_pinecone_filter({"tags": {"$eq": "vpn"}}) == {"tags": {"$in": ["vpn"]}}
    assert to_pinecone_filter({"tags": {"$ne": "vpn"}}) == {"tags": {"$nin": ["vpn"]}}
    assert to_pinecone_filter({}) is None and to_pinecone_filter(None) is None


def test_campo_desconocido_falla_en_voz_alta():
    with pytest.raises(UnknownFilterFieldError, match="department"):
        normalize_filter({"departamento": "finanzas"})


def test_operador_no_soportado():
    with pytest.raises(ValueError):
        normalize_filter({"effective_year": {"$regex": "20"}})


def _reference(meta: dict, flt: dict) -> bool:
    """Implementación de referencia de la semántica de Pinecone (AND entre campos y operadores)."""
    for key, cond in flt.items():
        val = meta.get(key)
        for op, exp in cond.items():
            if isinstance(val, list):
                inter = bool(set(val) & set(exp))
                ok = inter if op == "$in" else not inter
            elif op == "$eq":
                ok = val == exp
            elif op == "$ne":
                ok = val != exp
            elif op == "$in":
                ok = val in exp
            elif op == "$nin":
                ok = val not in exp
            else:
                ok = val is not None and {"$gt": val > exp, "$gte": val >= exp,
                                          "$lt": val < exp, "$lte": val <= exp}[op]
            if not ok:
                return False
    return True


METAS = [
    {"department": d, "effective_year": y, "tags": t, "source": s}
    for d, y, t, s in itertools.product(["it", "finanzas", "rrhh"], [2023, 2025, 2026],
                                        [["vpn", "2fa"], ["viajes"]], ["a.md", "b.txt"])
]
FILTROS = [
    {"department": "finanzas"},
    {"department": ["it", "rrhh"]},
    {"effective_year": {"$gte": 2025}},
    {"effective_year": {"$gt": 2023, "$lt": 2026}},
    {"tags": "vpn"},
    {"tags": ["viajes", "2fa"]},
    {"tags": {"$nin": ["vpn"]}},
    {"department": {"$ne": "it"}, "source": "a.md"},
    {"source": {"$nin": ["a.md"]}, "effective_year": {"$lte": 2025}},
]


@pytest.mark.parametrize("flt", FILTROS)
def test_matches_filter_equivale_a_referencia(flt):
    norm = normalize_filter(flt)
    for meta in METAS:
        assert matches_filter(meta, flt) == _reference(meta, norm), (meta, flt)


def test_parse_cli_filters():
    assert parse_cli_filters(["department=finanzas"]) == {"department": {"$eq": "finanzas"}}
    assert parse_cli_filters(["effective_year>=2025"]) == {"effective_year": {"$gte": 2025}}
    assert parse_cli_filters(["tags=vpn,2fa"]) == {"tags": {"$in": ["vpn", "2fa"]}}
    assert parse_cli_filters(["department!=it", "effective_year<2026"]) == {
        "department": {"$ne": "it"}, "effective_year": {"$lt": 2026}}
    assert parse_cli_filters(["effective_year>2023", "effective_year<=2026"]) == {
        "effective_year": {"$gt": 2023, "$lte": 2026}}
    with pytest.raises(ValueError):
        parse_cli_filters(["sin_operador"])
