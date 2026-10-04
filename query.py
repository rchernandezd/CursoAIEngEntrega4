"""Consulta puntual o interactiva sobre el RAGSystem híbrido.

    python query.py "¿Qué significa el error VPN-809?"
    python query.py "¿Con cuántos días de anticipación hay que solicitarlo?" --filtro department=finanzas
    python query.py "..." --modo bm25 --k 3
    python query.py "..." --diagnostico      # muestra rank_bm25 / rank_vectorial
    python query.py                          # modo interactivo; 'salir' termina
"""
import argparse
import logging
import sys

from rag_pinecone.config import Settings
from rag_pinecone.exceptions import RAGConfigError
from rag_pinecone.filters import normalize_filter, parse_cli_filters
from rag_pinecone.logging_config import setup_console, setup_logging

logger = logging.getLogger("rag_pinecone")


def _print_results(results: list[dict], diagnostico: bool) -> None:
    if not results:
        print("(sin resultados: ningún chunk cumple los filtros)")
    for r in results:
        extra = ""
        if diagnostico:
            extra = f"  [bm25 #{r['rank_bm25'] or '-'} | vectorial #{r['rank_vectorial'] or '-'}]"
        print(f"{r['rank']}. [{r['fuente']} | {r['categoria']} | {r['seccion'] or '-'}] {r['chunk_id']}{extra}")
        texto = " ".join(r["contenido"].split())
        print(f"   {texto[:200]}{'...' if len(texto) > 200 else ''}")


def main(argv=None, *, rag=None) -> int:
    setup_console()
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pregunta", nargs="?")
    parser.add_argument("--filtro", action="append", default=[], help="campo=valor, campo>=2025, tags=vpn,2fa")
    parser.add_argument("--modo", choices=("hibrido", "bm25", "vectorial"), default="hibrido")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--diagnostico", action="store_true")
    args = parser.parse_args(argv)
    try:
        filtros = parse_cli_filters(args.filtro) or None
        normalize_filter(filtros)          # falla en voz alta ante un campo mal escrito
        if rag is None:
            from rag_pinecone.rag_system import RAGSystem
            rag = RAGSystem.from_settings(Settings.from_env())

        def responder(q: str) -> None:
            if args.diagnostico:
                _print_results(rag.diagnosticar(q, k=args.k, filtros=filtros), True)
            else:
                _print_results(rag.obtener_top_k(q, k=args.k, filtros=filtros, modo=args.modo), False)

        if args.pregunta:
            responder(args.pregunta)
            return 0
        print("Modo interactivo (escribí 'salir' para terminar).")
        while True:
            try:
                q = input("\n❓ Pregunta: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if q.lower() in ("salir", "exit", "quit"):
                return 0
            if q:
                responder(q)
    except (RAGConfigError, ValueError) as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("query falló")
        print(f"❌ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
