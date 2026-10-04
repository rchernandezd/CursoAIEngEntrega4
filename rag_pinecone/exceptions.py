"""Excepciones del módulo. Todos los mensajes indican el comando a ejecutar para resolverlos."""


class RAGConfigError(RuntimeError):
    """Error de configuración o de estado del índice (los CLIs lo muestran sin traceback)."""


class MissingApiKeyError(RAGConfigError):
    """Falta PINECONE_API_KEY (u OPENAI_API_KEY si EMBEDDING_PROVIDER=openai)."""


class CatalogError(RAGConfigError):
    """Archivo de data/ sin entrada en catalog.json, o entrada del catálogo sin archivo."""


class IndexNotFoundError(RAGConfigError):
    """El índice no existe; se crea con setup_index.py."""


class IndexDimensionMismatchError(RAGConfigError):
    """La dimensión del índice no coincide con la del modelo de embeddings."""


class IndexMetricMismatchError(RAGConfigError):
    """La métrica del índice no coincide con la configurada."""


class EmbeddingModelMismatchError(RAGConfigError):
    """El namespace fue indexado con otro modelo de embeddings."""


class EmptyNamespaceError(RAGConfigError):
    """El namespace no tiene vectores; se cargan con ingest.py."""


class UnknownFilterFieldError(ValueError):
    """Filtro con una clave que no existe en el esquema ChunkMetadata."""


class ChunkTooLongError(RAGConfigError):
    """chunk_size no entra en el max_seq_length del embedder (habría truncación silenciosa)."""
