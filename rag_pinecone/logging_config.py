import logging
import os
import sys

_NOISY = ("pinecone", "urllib3", "httpx", "httpcore", "sentence_transformers", "transformers", "filelock",
          "huggingface_hub")


def setup_console() -> None:
    """UTF-8 en stdout/stderr (la consola de Windows redirigida usa cp1252 y rompe con tildes/emojis)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configura el logger 'rag_pinecone' a stdout. Nunca loguear API keys ni vectores."""
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logger = logging.getLogger("rag_pinecone")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger
