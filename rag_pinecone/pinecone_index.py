"""Infraestructura cloud: cliente Pinecone, creación/validación del índice serverless y helpers por namespace."""
import logging
import time

from .exceptions import (
    EmbeddingModelMismatchError,
    EmptyNamespaceError,
    IndexDimensionMismatchError,
    IndexMetricMismatchError,
    IndexNotFoundError,
    MissingApiKeyError,
    RAGConfigError,
)

logger = logging.getLogger("rag_pinecone")


def _get(obj, key, default=None):
    """Lee un atributo o clave: el SDK devuelve objetos o dicts según la versión."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, None)
    if value is None:
        try:
            value = obj[key]
        except (KeyError, TypeError, IndexError):
            value = default
    return value


def get_pinecone_client(settings):
    """MissingApiKeyError si settings.pinecone_api_key está vacío. Nunca loguear la key."""
    if not settings.pinecone_api_key:
        raise MissingApiKeyError(
            "Falta PINECONE_API_KEY. Creá .env desde .env.example (copy .env.example .env) y completala "
            "con la key de app.pinecone.io (API Keys → Create API key)."
        )
    from pinecone import Pinecone
    return Pinecone(api_key=settings.pinecone_api_key)


def index_exists(pc, name: str) -> bool:
    if hasattr(pc, "has_index"):
        return bool(pc.has_index(name))
    return name in pc.list_indexes().names()


def describe(pc, name: str) -> dict:
    """Resumen de describe_index: dimension, metric, ready, cloud, region."""
    desc = pc.describe_index(name)
    spec = _get(desc, "spec")
    serverless = _get(spec, "serverless")
    status = _get(desc, "status")
    return {
        "dimension": _get(desc, "dimension"),
        "metric": _get(desc, "metric"),
        "ready": bool(_get(status, "ready", False)),
        "cloud": _get(serverless, "cloud"),
        "region": _get(serverless, "region"),
    }


def validate_index(pc, settings, dimension: int) -> dict:
    info = describe(pc, settings.index_name)
    if info["dimension"] is not None and int(info["dimension"]) != dimension:
        raise IndexDimensionMismatchError(
            f"El índice '{settings.index_name}' tiene dimensión {info['dimension']} pero el modelo "
            f"{settings.embedding_model_id} produce vectores de {dimension}. Usá otro INDEX_NAME en .env "
            f"(ej. '{settings.index_name}-{dimension}') y ejecutá python setup_index.py, o borrá el índice "
            f"desde la consola de Pinecone."
        )
    metric = str(info["metric"]).lower() if info["metric"] is not None else None
    if metric is not None and metric.split(".")[-1] != settings.metric:
        raise IndexMetricMismatchError(
            f"El índice '{settings.index_name}' usa la métrica '{info['metric']}' y se requiere "
            f"'{settings.metric}'. Usá otro INDEX_NAME en .env y ejecutá python setup_index.py."
        )
    return info


def ensure_index(pc, settings, dimension: int, *, create: bool = False):
    """Devuelve (index, created). Valida dimensión y métrica si existe; crea solo si create=True."""
    created = False
    if index_exists(pc, settings.index_name):
        validate_index(pc, settings, dimension)
    elif create:
        from pinecone import ServerlessSpec
        logger.info("Creando índice serverless '%s' (dim %d, %s, %s/%s)", settings.index_name, dimension,
                    settings.metric, settings.cloud, settings.region)
        try:
            pc.create_index(
                name=settings.index_name,
                dimension=dimension,
                metric=settings.metric,
                spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "quota" in msg or "forbidden" in msg or ("max" in msg and "index" in msg):
                raise RAGConfigError(
                    f"No se pudo crear '{settings.index_name}': se alcanzó el límite de índices del plan. "
                    f"Borrá un índice que no uses (por ejemplo 'techcorp-rag-hibrido' del notebook) desde la "
                    f"consola de Pinecone y volvé a ejecutar python setup_index.py. Detalle: {exc}"
                ) from exc
            raise
        wait_until_ready(pc, settings.index_name)
        created = True
    else:
        raise IndexNotFoundError(
            f"El índice '{settings.index_name}' no existe. Ejecutá: python setup_index.py"
        )
    return pc.Index(settings.index_name), created


def wait_until_ready(pc, name: str, timeout_s: float = 120, poll_s: float = 2) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        if describe(pc, name)["ready"]:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"El índice '{name}' no quedó listo en {timeout_s:.0f} s")
        time.sleep(poll_s)


def list_ids(index, namespace: str, prefix: str | None = None) -> list[str]:
    """Aplana las páginas de index.list(prefix=..., namespace=...) (solo serverless)."""
    kwargs = {"namespace": namespace}
    if prefix:
        kwargs["prefix"] = prefix
    ids: list[str] = []
    for page in index.list(**kwargs):
        ids.extend(page if isinstance(page, (list, tuple)) else [page])
    return ids


def fetch_metadata(index, namespace: str, ids: list[str], batch: int = 100) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(ids), batch):
        resp = index.fetch(ids=ids[i:i + batch], namespace=namespace)
        vectors = _get(resp, "vectors", {}) or {}
        for vid, vec in vectors.items():
            out[vid] = dict(_get(vec, "metadata", {}) or {})
    return out


def delete_ids(index, namespace: str, ids: list[str], batch: int = 1000) -> int:
    for i in range(0, len(ids), batch):
        index.delete(ids=ids[i:i + batch], namespace=namespace)
    return len(ids)


def delete_namespace(index, namespace: str) -> None:
    if namespace_vector_count(index, namespace):
        index.delete(delete_all=True, namespace=namespace)


def index_stats(index) -> dict:
    stats = index.describe_index_stats()
    namespaces = _get(stats, "namespaces", {}) or {}
    return {
        "dimension": _get(stats, "dimension"),
        "total_vector_count": _get(stats, "total_vector_count"),
        "namespaces": {ns: int(_get(v, "vector_count", 0) or 0) for ns, v in dict(namespaces).items()},
    }


def namespace_vector_count(index, namespace: str) -> int:
    return index_stats(index)["namespaces"].get(namespace, 0)


def wait_for_count(index, namespace: str, expected: int, timeout_s: float = 60, poll_s: float = 1) -> bool:
    """Consistencia eventual: espera a que el conteo del namespace sea == expected."""
    deadline = time.monotonic() + timeout_s
    while True:
        count = namespace_vector_count(index, namespace)
        if count == expected:
            return True
        if time.monotonic() > deadline:
            logger.warning("El namespace '%s' reporta %d vectores (se esperaban %d) tras %.0f s",
                           namespace, count, expected, timeout_s)
            return False
        time.sleep(poll_s)


def assert_namespace_compatible(index, settings, *, require_vectors: bool) -> None:
    """Compara metadata['embedding_model'] de un vector del namespace con el modelo configurado."""
    ns = settings.namespace
    first = None
    for page in index.list(namespace=ns, limit=1):
        items = page if isinstance(page, (list, tuple)) else [page]
        if items:
            first = items[0]
            break
    if first is None:
        if require_vectors:
            raise EmptyNamespaceError(
                f"El namespace '{ns}' del índice '{settings.index_name}' no tiene vectores. Ejecutá: python ingest.py"
            )
        return
    meta = fetch_metadata(index, ns, [first]).get(first, {})
    found = meta.get("embedding_model")
    if found != settings.embedding_model_id:
        raise EmbeddingModelMismatchError(
            f"El namespace '{ns}' fue indexado con '{found}' y la configuración actual usa "
            f"'{settings.embedding_model_id}'. Ejecutá: python ingest.py --force (reindexa con el modelo actual)."
        )
