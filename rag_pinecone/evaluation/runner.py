"""Evaluación sobre el Golden Set: benchmark de la consigna (5 casos núcleo), comparación de modos,
barrido de pesos y render del reporte (consola y Markdown)."""
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..schemas import CasoGolden
from . import metrics as M

METRIC_KEYS = ("precision", "precision_normalizada", "recall", "f1", "hit_rate", "mrr", "ndcg", "evidence_recall")
METRIC_LABELS = {
    "precision": "P@{k}", "precision_normalizada": "P@{k} norm", "recall": "R@{k}", "f1": "F1@{k}",
    "hit_rate": "Hit@{k}", "mrr": "MRR", "ndcg": "nDCG@{k}", "evidence_recall": "Evid.",
}
N_NUCLEO = 5
UMBRALES_STRICT = {"nucleo_recall": 1.0, "hit_rate": 0.95, "recall": 0.90, "mrr": 0.70}


def load_golden_set(path: Path, known_sources: set[str]) -> list[CasoGolden]:
    """Valida esquema, que cada documento esperado exista y que haya exactamente 5 casos núcleo."""
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    casos = [CasoGolden(**c) for c in raw]
    ids = [c.id for c in casos]
    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        raise ValueError(f"Golden Set con ids duplicados: {dup}")
    desconocidos = sorted({d for c in casos for d in c.documentos_esperados} - set(known_sources))
    if desconocidos:
        raise ValueError(f"Golden Set referencia documentos que no están en el catálogo: {desconocidos}")
    n_nucleo = sum(c.nucleo for c in casos)
    if n_nucleo != N_NUCLEO:
        raise ValueError(f"El Golden Set debe tener exactamente {N_NUCLEO} casos 'nucleo' (tiene {n_nucleo})")
    return casos


@dataclass
class EvalReport:
    modo: str
    k: int
    pesos: tuple[float, float] | None
    detalle: list[dict]
    promedios: dict[str, float]
    por_tipo: dict[str, dict[str, float]] = field(default_factory=dict)


def _promedios(filas: list[dict]) -> dict[str, float]:
    if not filas:
        return {m: 0.0 for m in METRIC_KEYS}
    return {m: sum(f[m] for f in filas) / len(filas) for m in METRIC_KEYS}


def evaluar_caso(rag_system, caso: CasoGolden, *, k: int, modo: str, chunks_por_fuente: dict[str, int],
                 usar_filtros: bool = True) -> dict:
    docs = rag_system.recuperar(caso.pregunta, k=k, modo=modo, filtros=caso.filtros if usar_filtros else None)
    fuentes = [d.metadata.get("source", "") for d in docs]
    rel = caso.documentos_esperados
    techo = M.precision_ceiling(rel, chunks_por_fuente, k)
    p = M.precision_at_k(fuentes, rel, k)
    return {
        "id": caso.id,
        "pregunta": caso.pregunta,
        "tipo": caso.tipo,
        "nucleo": caso.nucleo,
        "filtros": caso.filtros if usar_filtros else None,
        "esperados": rel,
        "recuperados": fuentes,
        "chunk_ids": [d.metadata.get("chunk_id", "") for d in docs],
        "precision": p,
        "techo": techo,
        "precision_normalizada": p / techo if techo else 0.0,
        "recall": M.recall_at_k(fuentes, rel, k),
        "f1": M.f1_at_k(fuentes, rel, k),
        "hit_rate": M.hit_rate_at_k(fuentes, rel, k),
        "mrr": M.reciprocal_rank(fuentes, rel),
        "ndcg": M.ndcg_at_k(fuentes, rel, k, chunks_por_fuente),
        "evidence_recall": M.evidence_recall([(d.metadata.get("source", ""), d.page_content) for d in docs],
                                             rel, caso.evidencias),
    }


def evaluar(rag_system, golden_set: list[CasoGolden], *, k: int = 5, modo: str = "hibrido",
            chunks_por_fuente: dict[str, int], solo_nucleo: bool = False) -> EvalReport:
    casos = [c for c in golden_set if c.nucleo] if solo_nucleo else list(golden_set)
    filas = [evaluar_caso(rag_system, c, k=k, modo=modo, chunks_por_fuente=chunks_por_fuente) for c in casos]
    grupos: dict[str, list[dict]] = defaultdict(list)
    for f in filas:
        grupos[f["tipo"]].append(f)
    pesos = tuple(rag_system.weights) if modo == "hibrido" else None
    return EvalReport(modo, k, pesos, filas, _promedios(filas), {t: _promedios(g) for t, g in sorted(grupos.items())})


def comparar_modos(rag_system, golden_set, *, k: int, chunks_por_fuente: dict[str, int],
                   modos=("bm25", "vectorial", "hibrido"), solo_nucleo: bool = False) -> dict[str, EvalReport]:
    return {m: evaluar(rag_system, golden_set, k=k, modo=m, chunks_por_fuente=chunks_por_fuente,
                       solo_nucleo=solo_nucleo) for m in modos}


def barrido_pesos(rag_system, golden_set, *, k: int, chunks_por_fuente: dict[str, int],
                  pesos_bm25=(0.3, 0.4, 0.5, 0.6, 0.7)) -> list[EvalReport]:
    """Un RAGSystem por combinación, reutilizando corpus, BM25 y vectorstore (sin re-fetch)."""
    out = []
    for w in pesos_bm25:
        variante = rag_system.with_weights((round(w, 2), round(1 - w, 2)))
        out.append(evaluar(variante, golden_set, k=k, modo="hibrido", chunks_por_fuente=chunks_por_fuente))
    return out


def mejor_combinacion(barrido: list[EvalReport]) -> EvalReport:
    return max(barrido, key=lambda r: (round(r.promedios["mrr"], 4), round(r.promedios["recall"], 4),
                                       round(r.promedios["precision"], 4)))


def chunks_por_fuente_de(corpus) -> dict[str, int]:
    return dict(Counter(d.metadata.get("source", "") for d in corpus))


# --- umbrales -----------------------------------------------------------------------------------

def nucleo_report(report: EvalReport) -> EvalReport:
    filas = [f for f in report.detalle if f["nucleo"]]
    return EvalReport(report.modo, report.k, report.pesos, filas, _promedios(filas))


def check_strict(hibrido: EvalReport) -> list[str]:
    """Devuelve la lista de umbrales incumplidos (vacía si todo OK)."""
    fallos = []
    nuc = nucleo_report(hibrido).promedios
    if nuc["recall"] < UMBRALES_STRICT["nucleo_recall"]:
        fallos.append(f"benchmark núcleo recall={nuc['recall']:.2f} < {UMBRALES_STRICT['nucleo_recall']}")
    for m in ("hit_rate", "recall", "mrr"):
        if hibrido.promedios[m] < UMBRALES_STRICT[m]:
            fallos.append(f"Golden Set {m}={hibrido.promedios[m]:.2f} < {UMBRALES_STRICT[m]}")
    return fallos


# --- conclusión breve (reglas deterministas) ---------------------------------------------------

def conclusion_breve(comparacion: dict[str, EvalReport]) -> list[str]:
    lineas = []
    h = comparacion.get("hibrido")
    indiv = {m: r for m, r in comparacion.items() if m != "hibrido"}
    if h and indiv:
        mejor_m, mejor = max(indiv.items(), key=lambda kv: (kv[1].promedios["recall"], kv[1].promedios["mrr"]))
        hr, br = h.promedios["recall"], mejor.promedios["recall"]
        verbo = "iguala o supera" if hr >= br - 1e-9 else "queda por debajo de"
        lineas.append(f"El híbrido {verbo} al mejor retriever individual ({mejor_m}) en Recall@{h.k} "
                      f"({hr:.2f} vs {br:.2f}); MRR {h.promedios['mrr']:.2f} vs {mejor.promedios['mrr']:.2f}.")
    b, v = comparacion.get("bm25"), comparacion.get("vectorial")
    if b and v:
        for tipo, nombre in (("lexica", "léxicas"), ("semantica", "semánticas")):
            mb = b.por_tipo.get(tipo, {}).get("mrr")
            mv = v.por_tipo.get(tipo, {}).get("mrr")
            if mb is None or mv is None:
                continue
            ganador = "gana BM25" if mb > mv else ("gana el vectorial" if mv > mb else "empate")
            lineas.append(f"Preguntas {nombre}: {ganador} (MRR bm25 {mb:.2f} vs vectorial {mv:.2f}).")
    if h:
        fallos = [f for f in h.detalle if f["recall"] < 1.0]
        if fallos:
            desc = []
            for f in fallos[:4]:
                faltan = sorted(set(f["esperados"]) - set(f["recuperados"]))
                top = f["recuperados"][0] if f["recuperados"] else "nada"
                desc.append(f"{f['id']} (falta {', '.join(faltan)}; top-1: {top})")
            extra = f" y {len(fallos) - 4} más" if len(fallos) > 4 else ""
            lineas.append(f"Fallos del híbrido (recall<1): {'; '.join(desc)}{extra}.")
        else:
            lineas.append("El híbrido recupera todos los documentos esperados en todos los casos (recall=1).")
    return lineas


# --- render -------------------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _trunc(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def render_benchmark(report: EvalReport) -> str:
    nuc = nucleo_report(report)
    k = report.k
    out = [f"{'=' * 20} BENCHMARK CONSIGNA ({len(nuc.detalle)} preguntas, k={k}, {report.modo}) {'=' * 20}"]
    for f in nuc.detalle:
        icon = "✅" if f["recall"] >= 1.0 else "❌"
        out.append(f"{icon} {f['id']} {_trunc(f['pregunta'])}")
        out.append(f"   Esperado: {', '.join(f['esperados'])} | Top-{k}: {', '.join(f['recuperados'])}")
        out.append(f"   Recall@{k}: {_pct(f['recall'])} | Precision@{k}: {_pct(f['precision'])} "
                   f"(techo {_pct(f['techo'])})")
    p = nuc.promedios
    techo = sum(f["techo"] for f in nuc.detalle) / max(len(nuc.detalle), 1)
    out.append(f"📊 RECALL@{k} PROMEDIO:    {_pct(p['recall'])}")
    out.append(f"📊 PRECISION@{k} PROMEDIO: {_pct(p['precision'])}  (techo promedio {_pct(techo)} → "
               f"precisión normalizada {_pct(p['precision_normalizada'])})")
    return "\n".join(out)


def _table_rows(items: list[tuple[str, dict[str, float]]], k: int) -> list[str]:
    head = f"{'':<12}" + "".join(f"{METRIC_LABELS[m].format(k=k):<11}" for m in METRIC_KEYS)
    rows = [head]
    for name, prom in items:
        rows.append(f"{name:<12}" + "".join(f"{prom[m]:<11.2f}" for m in METRIC_KEYS))
    return rows


def render_consola(comparacion: dict[str, EvalReport], barrido: list[EvalReport] | None = None,
                   sin_filtro: list[dict] | None = None) -> str:
    any_rep = next(iter(comparacion.values()))
    k = any_rep.k
    bloques = []
    if "hibrido" in comparacion:
        bloques.append(render_benchmark(comparacion["hibrido"]))
    n = len(any_rep.detalle)
    t = [f"{'=' * 32} RESUMEN (k={k}, {n} casos) {'=' * 32}"]
    t += _table_rows([(m, r.promedios) for m, r in comparacion.items()], k)
    rep_tipo = comparacion.get("hibrido", any_rep)
    t.append("")
    t.append(f"Por tipo de pregunta ({rep_tipo.modo}):")
    t += _table_rows(list(rep_tipo.por_tipo.items()), k)
    bloques.append("\n".join(t))
    if sin_filtro:
        s = ["Casos filtrados evaluados SIN filtro (valor del hard filter, híbrido):"]
        for f in sin_filtro:
            s.append(f"  {f['id']}: R@{k} con filtro {f['recall_con']:.2f} → sin filtro {f['recall']:.2f} | "
                     f"top-1 sin filtro: {f['recuperados'][0] if f['recuperados'] else '-'}")
        bloques.append("\n".join(s))
    if barrido:
        b = [f"{'=' * 30} BARRIDO DE PESOS (bm25, vectorial) {'=' * 30}"]
        b += _table_rows([(f"{r.pesos[0]:.1f}/{r.pesos[1]:.1f}", r.promedios) for r in barrido], k)
        best = mejor_combinacion(barrido)
        b.append(f"Mejor combinación por (MRR, Recall@{k}, Precision@{k}): bm25={best.pesos[0]:.1f} "
                 f"vectorial={best.pesos[1]:.1f}")
        bloques.append("\n".join(b))
    bloques.append("Conclusión breve:\n" + "\n".join(f"- {l}" for l in conclusion_breve(comparacion)))
    return "\n\n".join(bloques)


def _md_table(items: list[tuple[str, dict[str, float]]], k: int, first: str = "modo") -> list[str]:
    cols = [METRIC_LABELS[m].format(k=k) for m in METRIC_KEYS]
    out = [f"| {first} | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, prom in items:
        out.append(f"| {name} | " + " | ".join(f"{prom[m]:.2f}" for m in METRIC_KEYS) + " |")
    return out


def render_markdown(comparacion: dict[str, EvalReport], barrido: list[EvalReport] | None,
                    settings_snapshot: dict, timestamp: str, *, sin_filtro: list[dict] | None = None,
                    lectura: list[str] | None = None) -> str:
    any_rep = next(iter(comparacion.values()))
    k = any_rep.k
    md = ["# Reporte de evaluación — RAG híbrido con Pinecone", "",
          f"Generado: {timestamp} por `python evaluate.py`.", "", "## Configuración", ""]
    md += [f"- **{key}**: `{val}`" for key, val in settings_snapshot.items()]
    md += ["", "## Consola", "", "```text", render_consola(comparacion, barrido, sin_filtro), "```", ""]
    md += [f"## Comparación de modos (k={k}, {len(any_rep.detalle)} casos)", ""]
    md += _md_table([(m, r.promedios) for m, r in comparacion.items()], k)
    if "hibrido" in comparacion:
        h = comparacion["hibrido"]
        md += ["", "## Híbrido por tipo de pregunta", ""]
        md += _md_table(list(h.por_tipo.items()), k, first="tipo")
        md += ["", "## Detalle por caso (híbrido)", "",
               f"| id | tipo | núcleo | esperados | top-{k} | P@{k} | techo | R@{k} | RR | Evid. |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for f in h.detalle:
            md.append(f"| {f['id']} | {f['tipo']} | {'sí' if f['nucleo'] else ''} | {', '.join(f['esperados'])} | "
                      f"{', '.join(f['recuperados'])} | {f['precision']:.2f} | {f['techo']:.2f} | "
                      f"{f['recall']:.2f} | {f['mrr']:.2f} | {f['evidence_recall']:.2f} |")
    if barrido:
        md += ["", "## Barrido de pesos (bm25/vectorial)", ""]
        md += _md_table([(f"{r.pesos[0]:.1f}/{r.pesos[1]:.1f}", r.promedios) for r in barrido], k, first="pesos")
    md += ["", "## Lectura de resultados", ""]
    md += [f"- {l}" for l in (lectura or conclusion_breve(comparacion))]
    md += ["", "### Notas sobre las métricas", "",
           f"- **Precision@{k}** divide siempre por k. Un documento de 1 chunk como única respuesta tiene un techo "
           f"de {1 / k:.2f}; por eso se informa el techo y la precisión normalizada (precisión / techo).",
           f"- **Recall@{k}** vale 0 o 1 en preguntas de un documento (benchmark núcleo) y es fraccionario en las "
           f"multi-documento.",
           "- **Evid.** mide si los chunks recuperados de documentos relevantes contienen el dato esperado."]
    return "\n".join(md) + "\n"


def to_json(comparacion: dict[str, EvalReport], barrido: list[EvalReport] | None, settings_snapshot: dict,
            timestamp: str) -> str:
    data = {"timestamp": timestamp, "settings": settings_snapshot,
            "comparacion": {m: asdict(r) for m, r in comparacion.items()},
            "barrido": [asdict(r) for r in barrido or []]}
    return json.dumps(data, ensure_ascii=False, indent=2)
