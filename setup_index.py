"""Inicialización del índice Pinecone Serverless: verifica si existe y lo crea si es necesario.

    python setup_index.py            # verifica/crea INDEX_NAME (idempotente)
    python setup_index.py --check    # solo verifica; nunca crea (exit 1 si no existe o no es compatible)
"""
import argparse
import logging
import sys
import time

from rag_pinecone.config import Settings
from rag_pinecone.embeddings import build_embeddings, get_embedding_dimension
from rag_pinecone.exceptions import RAGConfigError
from rag_pinecone.logging_config import setup_console, setup_logging
from rag_pinecone.pinecone_index import ensure_index, get_pinecone_client, index_exists, namespace_vector_count

logger = logging.getLogger("rag_pinecone")


def main(argv=None, *, settings=None, pc=None, embeddings=None) -> int:
    setup_console()
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="solo verifica, nunca crea el índice")
    args = parser.parse_args(argv)
    try:
        settings = settings or Settings.from_env()
        pc = pc or get_pinecone_client(settings)
        embeddings = embeddings or build_embeddings(settings)
        dimension = get_embedding_dimension(embeddings)
        existed = index_exists(pc, settings.index_name)
        t0 = time.perf_counter()
        index, created = ensure_index(pc, settings, dimension, create=not args.check)
        spec = f"Serverless {settings.cloud}/{settings.region}, dim {dimension}, {settings.metric}"
        if created:
            print(f"🆕 Índice '{settings.index_name}' creado ({spec}) — listo en {time.perf_counter() - t0:.1f} s")
        elif existed:
            print(f"♻️ Índice '{settings.index_name}' ya existe — dimensión {dimension} ✓, métrica "
                  f"{settings.metric} ✓ — no se vuelve a crear")
        count = namespace_vector_count(index, settings.namespace)
        siguiente = "python ingest.py" if count == 0 else "python ingest.py (sube solo lo nuevo/cambiado)"
        print(f"   Namespace '{settings.namespace}': {count} vectores → siguiente paso: {siguiente}")
        return 0
    except (RAGConfigError, ValueError) as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("setup_index falló")
        print(f"❌ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
