"""Evaluación del recuperador: Recall@k y Precision@k (+ F1, Hit Rate, MRR, nDCG, evidencias) sobre el
benchmark de 5 preguntas de la consigna y el Golden Set ampliado, comparando bm25 / vectorial / híbrido.

    python evaluate.py                      # k=5: benchmark núcleo + Golden Set ampliado, 3 modos
    python evaluate.py --solo-nucleo        # solo las 5 preguntas de la consigna
    python evaluate.py --k 3
    python evaluate.py --modos hibrido
    python evaluate.py --sweep-pesos        # agrega barrido de pesos BM25/vectorial
    python evaluate.py --strict             # exit code 1 si el híbrido no cumple umbrales
    python evaluate.py --golden eval/golden_set.json
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from rag_pinecone.config import Settings
from rag_pinecone.evaluation.runner import (
    barrido_pesos,
    check_strict,
    chunks_por_fuente_de,
    comparar_modos,
    evaluar_caso,
    load_golden_set,
    render_consola,
    render_markdown,
    to_json,
)
from rag_pinecone.exceptions import RAGConfigError
from rag_pinecone.logging_config import setup_console, setup_logging
from rag_pinecone.text_processing import load_catalog

logger = logging.getLogger("rag_pinecone")
MODOS = ("bm25", "vectorial", "hibrido")
LECTURA_PATH = Path("eval/lectura_resultados.md")


def main(argv=None, *, settings=None, rag=None, golden=None) -> int:
    setup_console()
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--modos", nargs="+", choices=MODOS, default=list(MODOS))
    parser.add_argument("--solo-nucleo", action="store_true")
    parser.add_argument("--sweep-pesos", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--no-report", action="store_true", help="no escribe reports/")
    args = parser.parse_args(argv)
    try:
        settings = settings or Settings.from_env()
        k = args.k or settings.top_k
        if golden is None:
            known = set(load_catalog(settings.catalog_path))
            golden = load_golden_set(args.golden or settings.golden_set_path, known)
        if rag is None:
            from rag_pinecone.rag_system import RAGSystem
            rag = RAGSystem.from_settings(settings)
        cpf = chunks_por_fuente_de(rag.corpus)
        modos = [m for m in MODOS if m in args.modos]

        comparacion = comparar_modos(rag, golden, k=k, chunks_por_fuente=cpf, modos=modos,
                                     solo_nucleo=args.solo_nucleo)
        sin_filtro = []
        if "hibrido" in comparacion and not args.solo_nucleo:
            con = {f["id"]: f for f in comparacion["hibrido"].detalle}
            for caso in golden:
                if caso.filtros:
                    fila = evaluar_caso(rag, caso, k=k, modo="hibrido", chunks_por_fuente=cpf, usar_filtros=False)
                    fila["recall_con"] = con[caso.id]["recall"]
                    sin_filtro.append(fila)
        barrido = None
        if args.sweep_pesos and not args.solo_nucleo:
            barrido = barrido_pesos(rag, golden, k=k, chunks_por_fuente=cpf)

        print(render_consola(comparacion, barrido, sin_filtro))

        if not args.no_report:
            ts = datetime.now().astimezone().isoformat(timespec="seconds")
            snap = settings.snapshot() | {"k": k, "golden_set": str(args.golden or settings.golden_set_path)}
            settings.reports_dir.mkdir(parents=True, exist_ok=True)
            md_path = settings.reports_dir / "eval_report.md"
            lectura = None
            if LECTURA_PATH.exists():   # análisis escrito a mano a partir de los números reales
                lectura = [l[2:].strip() for l in LECTURA_PATH.read_text(encoding="utf-8").splitlines()
                           if l.startswith("- ")]
            md_path.write_text(render_markdown(comparacion, barrido, snap, ts, sin_filtro=sin_filtro,
                                               lectura=lectura), encoding="utf-8")
            (settings.reports_dir / "eval_results.json").write_text(to_json(comparacion, barrido, snap, ts),
                                                                    encoding="utf-8")
            print(f"\n📝 Reporte: {md_path}")

        if args.strict:
            if "hibrido" not in comparacion:
                print("--strict requiere el modo hibrido", file=sys.stderr)
                return 1
            fallos = check_strict(comparacion["hibrido"])
            if fallos:
                print("❌ --strict: no se cumplen los umbrales: " + "; ".join(fallos), file=sys.stderr)
                return 1
            print("✅ --strict: el híbrido cumple todos los umbrales.")
        return 0
    except (RAGConfigError, ValueError, FileNotFoundError) as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("evaluate falló")
        print(f"❌ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
