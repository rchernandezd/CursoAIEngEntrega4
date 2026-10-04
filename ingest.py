"""Ingesta de data/ a Pinecone (requiere el índice creado con setup_index.py).

    python ingest.py                         # sube solo lo nuevo/cambiado
    python ingest.py --dry-run               # muestra qué haría, sin escribir
    python ingest.py --force                 # reindexa todos los documentos
    python ingest.py --prune                 # además borra vectores de documentos que ya no están en data/
    python ingest.py --stats                 # solo imprime describe_index_stats() del índice/namespace
    python ingest.py --reset-namespace --yes # borra TODOS los vectores del namespace
"""
import argparse
import logging
import sys

from rag_pinecone.config import Settings
from rag_pinecone.exceptions import RAGConfigError
from rag_pinecone.logging_config import setup_console, setup_logging
from rag_pinecone.pinecone_index import describe, delete_namespace, get_pinecone_client, index_exists, index_stats

logger = logging.getLogger("rag_pinecone")


def _index_line(pc, settings, count: int) -> str:
    info = describe(pc, settings.index_name)
    return (f"Índice: {settings.index_name} (dim {info['dimension']}, {info['metric']}, "
            f"{info['cloud'] or settings.cloud}/{info['region'] or settings.region}) | "
            f"Namespace: {settings.namespace} | Vectores: {count}")


def print_report(report) -> None:
    rows = [("archivo", "estado", "chunks", "tokens máx", "subidos", "borrados")]
    rows += [(d.source, d.status, str(d.n_chunks), str(d.max_tokens), str(d.upserted), str(d.deleted))
             for d in report.docs]
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for i, r in enumerate(rows):
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))
        if i == 0:
            print("-+-".join("-" * w for w in widths))
    titulo = "DRY-RUN (no se escribió nada): " if report.dry_run else ""
    print(f"\n{titulo}subidos={report.total_upserted} borrados={report.total_deleted} "
          f"tiempo={report.elapsed_s:.1f} s")


def main(argv=None, *, settings=None, pc=None, embeddings=None) -> int:
    setup_console()
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--reset-namespace", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirma --reset-namespace")
    args = parser.parse_args(argv)
    try:
        settings = settings or Settings.from_env()
        pc = pc or get_pinecone_client(settings)
        if args.stats or args.reset_namespace:
            if not index_exists(pc, settings.index_name):
                raise RAGConfigError(f"El índice '{settings.index_name}' no existe. Ejecutá: python setup_index.py")
            index = pc.Index(settings.index_name)
            if args.reset_namespace:
                if not args.yes:
                    print("--reset-namespace borra TODOS los vectores del namespace: agregá --yes para confirmar.",
                          file=sys.stderr)
                    return 2
                delete_namespace(index, settings.namespace)
                print(f"🗑️ Namespace '{settings.namespace}' vaciado.")
            stats = index_stats(index)
            info = describe(pc, settings.index_name)
            print(f"dimension: {stats['dimension']} | metric: {info['metric']} | "
                  f"total_vector_count: {stats['total_vector_count']}")
            for ns, n in sorted(stats["namespaces"].items()):
                mark = "  ← configurado" if ns == settings.namespace else ""
                print(f"  namespace '{ns}': vector_count={n}{mark}")
            return 0

        from rag_pinecone.embeddings import build_embeddings
        from rag_pinecone.ingestion import IngestionPipeline

        embeddings = embeddings or build_embeddings(settings)
        pipeline = IngestionPipeline.from_client(settings, pc, embeddings)
        report = pipeline.run(force=args.force, prune=args.prune, dry_run=args.dry_run)
        print_report(report)
        print(_index_line(pc, settings, report.namespace_count))
        return 0
    except (RAGConfigError, FileNotFoundError, ValueError) as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("ingest falló")
        print(f"❌ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
