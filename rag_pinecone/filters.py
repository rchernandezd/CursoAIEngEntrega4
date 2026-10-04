"""Filtros por metadata ("hard filter"): un formato de entrada, misma semántica en Pinecone y en Python."""
import re

from .exceptions import UnknownFilterFieldError
from .schemas import FILTERABLE_FIELDS, LIST_FIELDS

SUPPORTED_OPS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}


def normalize_filter(filtros: dict | None) -> dict | None:
    """Devuelve {campo: {op: valor}} validado.
    - Claves ⊂ FILTERABLE_FIELDS, si no UnknownFilterFieldError.
    - escalar → {"$eq": v}; lista → {"$in": v}; dict → se validan operadores.
    - Campo lista ("tags"): escalar → {"$in": [v]}.
    - {} o None → None."""
    if not filtros:
        return None
    if not isinstance(filtros, dict):
        raise TypeError("los filtros deben ser un dict")
    unknown = sorted(set(filtros) - FILTERABLE_FIELDS)
    if unknown:
        raise UnknownFilterFieldError(
            f"Campos de filtro desconocidos: {unknown}. Campos válidos: {sorted(FILTERABLE_FIELDS)}"
        )
    normalized: dict[str, dict] = {}
    for key, value in filtros.items():
        if isinstance(value, dict):
            bad = sorted(set(value) - SUPPORTED_OPS)
            if bad or not value:
                raise ValueError(f"Operadores no soportados en '{key}': {bad}. Válidos: {sorted(SUPPORTED_OPS)}")
            cond = {}
            for op, v in value.items():
                if key in LIST_FIELDS:
                    if op not in ("$eq", "$ne", "$in", "$nin"):
                        raise ValueError(f"'{key}' es una lista: solo admite $eq, $ne, $in, $nin")
                    op = {"$eq": "$in", "$ne": "$nin"}.get(op, op)
                if op in ("$in", "$nin"):
                    if not isinstance(v, (list, tuple)):
                        v = [v]
                    cond[op] = list(v)
                else:
                    cond[op] = v
            normalized[key] = cond
        elif isinstance(value, (list, tuple)):
            normalized[key] = {"$in": list(value)}
        elif key in LIST_FIELDS:
            normalized[key] = {"$in": [value]}
        else:
            normalized[key] = {"$eq": value}
    return normalized


def to_pinecone_filter(filtros: dict | None) -> dict | None:
    """Traducción a la sintaxis de Pinecone (AND implícito entre campos)."""
    return normalize_filter(filtros)


def _cmp(op: str, actual, expected) -> bool:
    if isinstance(actual, list):           # campo lista: semántica de Pinecone sobre arrays
        if op in ("$eq", "$in"):
            exp = expected if isinstance(expected, list) else [expected]
            return bool(set(actual) & set(exp))
        if op in ("$ne", "$nin"):
            exp = expected if isinstance(expected, list) else [expected]
            return not (set(actual) & set(exp))
        return False
    try:
        if op == "$eq":
            return actual == expected
        if op == "$ne":
            return actual != expected
        if op == "$in":
            return actual in expected
        if op == "$nin":
            return actual not in expected
        if actual is None:
            return False
        if op == "$gt":
            return actual > expected
        if op == "$gte":
            return actual >= expected
        if op == "$lt":
            return actual < expected
        if op == "$lte":
            return actual <= expected
    except TypeError:
        return False
    raise ValueError(f"operador no soportado: {op}")


def matches_filter(metadata: dict, filtros: dict | None) -> bool:
    """Predicado Python equivalente al filtro de Pinecone (para BM25 y para tests)."""
    norm = normalize_filter(filtros)
    if not norm:
        return True
    for key, cond in norm.items():
        if key not in metadata:
            if any(op not in ("$ne", "$nin") for op in cond):
                return False
            continue
        actual = metadata[key]
        if not all(_cmp(op, actual, expected) for op, expected in cond.items()):
            return False
    return True


_CLI_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(!=|>=|<=|=|>|<)\s*(.*?)\s*$")
_CLI_OPS = {"=": "$eq", "!=": "$ne", ">=": "$gte", "<=": "$lte", ">": "$gt", "<": "$lt"}


def _cast(v: str):
    return int(v) if re.fullmatch(r"-?\d+", v) else v


def parse_cli_filters(items: list[str] | None) -> dict:
    """["department=finanzas", "effective_year>=2025", "tags=vpn,2fa"] → dict normalizable."""
    result: dict = {}
    for item in items or []:
        m = _CLI_RE.match(item)
        if not m:
            raise ValueError(f"Filtro inválido '{item}'. Formato: campo=valor, campo>=2025, campo=a,b")
        key, op, raw = m.groups()
        values = [_cast(v.strip()) for v in raw.split(",") if v.strip()]
        if not values:
            raise ValueError(f"Filtro sin valor: '{item}'")
        if len(values) > 1:
            if op not in ("=", "!="):
                raise ValueError(f"Las listas solo admiten '=' o '!=': '{item}'")
            cond = {"$in" if op == "=" else "$nin": values}
        else:
            cond = {_CLI_OPS[op]: values[0]}
        result.setdefault(key, {}).update(cond)
    return result
