"""Única fuente de embeddings: modelo, dimensión real y tokenizer del MISMO embedder."""
from functools import lru_cache

from .exceptions import ChunkTooLongError, MissingApiKeyError

_E5_PREFIXES = ("intfloat/multilingual-e5", "intfloat/e5")
E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "
_OPENAI_MAX_TOKENS = 8191


def is_e5(model_name: str) -> bool:
    return model_name.startswith(_E5_PREFIXES)


@lru_cache(maxsize=4)
def get_embeddings(model_name: str):
    """Construye embeddings HuggingFace normalizados (con prefijos E5 si corresponde)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    encode_kwargs = {"normalize_embeddings": True}
    query_encode_kwargs = {"normalize_embeddings": True}
    if is_e5(model_name):
        encode_kwargs["prompt"] = E5_PASSAGE_PREFIX
        query_encode_kwargs["prompt"] = E5_QUERY_PREFIX
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs=encode_kwargs,
            query_encode_kwargs=query_encode_kwargs,
        )
    except TypeError:
        # Versión instalada sin soporte de query_encode_kwargs.
        if not is_e5(model_name):
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"},
                                         encode_kwargs={"normalize_embeddings": True})

        class _E5Embeddings(HuggingFaceEmbeddings):
            def embed_query(self, text: str):
                return super().embed_documents([f"{E5_QUERY_PREFIX}{text}"])[0]

            def embed_documents(self, texts):
                return super().embed_documents([f"{E5_PASSAGE_PREFIX}{t}" for t in texts])

        return _E5Embeddings(model_name=model_name, model_kwargs={"device": "cpu"},
                             encode_kwargs={"normalize_embeddings": True})


def get_embedding_dimension(embeddings) -> int:
    """Dimensión REAL del modelo (nunca hardcodeada)."""
    return len(embeddings.embed_query("dimension probe"))


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def get_token_counter(model_name: str):
    """Cuenta tokens con el tokenizer del MISMO modelo de embeddings (sin tokens especiales)."""
    tok = get_tokenizer(model_name)
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def get_max_seq_length(model_name: str) -> int:
    n = get_tokenizer(model_name).model_max_length
    return 512 if n is None or n > 100_000 else int(n)


# --- Fachada por proveedor (todos los módulos usan estas funciones) ---------------------------

def build_embeddings(settings):
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise MissingApiKeyError("EMBEDDING_PROVIDER=openai requiere OPENAI_API_KEY en .env")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return get_embeddings(settings.embedding_model)


def build_token_counter(settings):
    if settings.embedding_provider == "openai":
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(settings.embedding_model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    return get_token_counter(settings.embedding_model)


def max_seq_length_for(settings) -> int:
    if settings.embedding_provider == "openai":
        return _OPENAI_MAX_TOKENS
    return get_max_seq_length(settings.embedding_model)


def prefix_tokens_for(settings) -> int:
    if settings.embedding_provider == "huggingface" and is_e5(settings.embedding_model):
        return get_token_counter(settings.embedding_model)(E5_PASSAGE_PREFIX)
    return 0


def special_tokens_for(settings) -> int:
    return 0 if settings.embedding_provider == "openai" else 2   # [CLS]/<s> y [SEP]/</s>


def assert_chunk_fits(settings) -> None:
    """chunk_size + prefijo + tokens especiales <= max_seq_length; si no, ChunkTooLongError."""
    max_len = max_seq_length_for(settings)
    needed = settings.chunk_size + prefix_tokens_for(settings) + special_tokens_for(settings)
    if needed > max_len:
        raise ChunkTooLongError(
            f"CHUNK_SIZE={settings.chunk_size} + prefijo + tokens especiales = {needed} tokens supera el "
            f"max_seq_length={max_len} de {settings.embedding_model}: el embedder truncaría cada chunk en "
            f"silencio. Bajá CHUNK_SIZE a 500 en .env o usá un modelo con más contexto."
        )
