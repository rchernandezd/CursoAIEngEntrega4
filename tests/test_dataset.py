"""Chequeo del dataset real (offline): catálogo coherente, evidencias del Golden Set presentes en sus
documentos esperados y términos 'prohibidos' ausentes (PRD §8.3–§8.5)."""
import hashlib
import re

import pytest

from conftest import REPO
from rag_pinecone.evaluation.runner import load_golden_set
from rag_pinecone.text_processing import clean_text, load_catalog, load_documents, normalize_for_match

DATA = REPO / "data"

SHA256_ENTREGA3 = {
    "politica_vacaciones.txt": "66afee037a90365665eaf792b67b2daeaa7902847c2bb958331cd3c43cc6dd15",
    "politica_teletrabajo.txt": "07bbc3d9c99c221872e2a23b5417f0ab9b6970f33571096d45ef0612e1cda6b5",
    "politica_seguridad_informatica.txt": "75aa9c16d8b2d3d2eb10eab6c0c74a49baab4bc5b1aeec328dd9b5d300a08e44",
    "onboarding_nuevos_empleados.txt": "e44ae3b1c9f2b705d40c6ea8e11b2a6e9e605833463c2f7c2b8a22e24adcf1f6",
    "manual_beneficios_y_licencias.md": "a1d4850cba5127cb7e5d442d36f716e6b53bbbe0fb5a82d944d5ca32efbf0f45",
}

LITERALES = {
    "guia_acceso_remoto_vpn.md": ["TC-Connect", "versión 5.2", "VPN-401", "VPN-503", "vpn2.techcorp.internal",
                                  "VPN-809", "TCP 443", "interno 4357", "lunes a viernes de 8 a 20 hs",
                                  "12 horas", "dispositivos personales", "MDM", "split tunneling", "2FA"],
    "procedimiento_gestion_incidentes.md": ["SEV1", "SEV2", "SEV3", "SEV4", "15 minutos", "1 hora",
                                            "4 horas hábiles", "2 días hábiles", "#incidentes-seguridad",
                                            "Incident Commander", "postmortem", "5 días hábiles", "24 horas"],
    "politica_gastos_y_viajes.md": ["15 días de anticipación", "GastosTC", "10 días hábiles", "USD 60",
                                    "USD 120", "clase económica", "menos de 6 horas", "más de 8 horas",
                                    "tarjeta corporativa"],
    "sdk_instalacion_configuracion.md": ["pip install techcorp-sdk", "Python 3.10", "versión 3.4",
                                         "import tcsdk", "TCSDK_API_TOKEN", "tcsdk.toml", "TCSDK_ENV",
                                         "30 segundos"],
    "sdk_referencia_cliente.md": ["TCClient", "TCClient.from_env()", "get_employee(employee_id)",
                                  "list_requests", "page_size", "máximo de 200", "next_cursor", "AsyncTCClient"],
    "sdk_errores_y_reintentos.md": ["TCError", "TCAuthError", "HTTP 401", "TCRateLimitError", "HTTP 429",
                                    "Retry-After", "TCServerError", "RetryPolicy", "max_attempts",
                                    "backoff_factor", "3 intentos", "600 solicitudes por minuto"],
}

PROHIBIDOS = {
    "guia_acceso_remoto_vpn.md": ["SEV", "USD", "GastosTC", "postmortem", "15 días de anticipación", "20 Mbps"],
    "procedimiento_gestion_incidentes.md": ["VPN-", "USD", "GastosTC", "15 días de anticipación", "4357", "MDM",
                                            "2 horas"],
    "politica_gastos_y_viajes.md": ["VPN", "SEV", "postmortem", "5 días hábiles", "licencia parental"],
    "sdk_instalacion_configuracion.md": ["TCAuthError", "TCRateLimitError", "page_size", "VPN", "SEV", "GastosTC"],
    "sdk_referencia_cliente.md": ["TCSDK_API_TOKEN", "pip install", "TCRateLimitError", "Retry-After", "VPN"],
    "sdk_errores_y_reintentos.md": ["TCSDK_API_TOKEN", "page_size", "pip install", "VPN-", "SEV"],
}


def _clean(name: str) -> str:
    return clean_text((DATA / name).read_text(encoding="utf-8-sig"))


def _contains_prefix(haystack: str, needle: str) -> bool:
    """Coincidencia que empieza en límite de palabra: 'SEV' detecta 'SEV1'; '2 horas' no matchea '12 horas'."""
    return re.search(r"(?<!\w)" + re.escape(needle), haystack) is not None


@pytest.mark.parametrize("name,sha", SHA256_ENTREGA3.items())
def test_documentos_entrega3_sin_modificar(name, sha):
    assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == sha


def test_catalogo_cubre_data():
    catalog = load_catalog(DATA / "catalog.json")
    docs = load_documents(DATA, catalog)
    assert len(docs) == 11
    assert sum(1 for e in catalog.values() if e.doc_type == "documentacion_tecnica") == 3


@pytest.mark.parametrize("name", LITERALES)
def test_literales_presentes(name):
    text = " ".join(_clean(name).split())
    faltan = [lit for lit in LITERALES[name] if lit not in text]
    assert not faltan, f"{name}: faltan literales {faltan}"


@pytest.mark.parametrize("name", PROHIBIDOS)
def test_prohibidos_ausentes(name):
    text = " ".join(_clean(name).split())
    presentes = [p for p in PROHIBIDOS[name] if _contains_prefix(text, p)]
    assert not presentes, f"{name}: contiene términos prohibidos {presentes}"


@pytest.mark.parametrize("name", [n for n in LITERALES if n.startswith("sdk_")])
def test_sdk_tiene_bloques_python(name):
    assert (DATA / name).read_text(encoding="utf-8").count("```python") >= 2


def test_evidencias_en_documentos_esperados():
    catalog = load_catalog(DATA / "catalog.json")
    casos = load_golden_set(REPO / "eval" / "golden_set.json", set(catalog))
    norm = {name: normalize_for_match(_clean(name)) for name in catalog}
    faltantes = []
    for c in casos:
        for ev in c.evidencias:
            pat = r"(?<!\w)" + re.escape(normalize_for_match(ev)) + r"(?!\w)"
            if not any(re.search(pat, norm[d]) for d in c.documentos_esperados):
                faltantes.append((c.id, ev))
    assert not faltantes, f"Evidencias que no aparecen en sus documentos esperados: {faltantes}"
