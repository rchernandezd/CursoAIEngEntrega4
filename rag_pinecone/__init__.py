"""Módulo de recuperación escalable: ingesta a Pinecone Serverless + recuperador híbrido BM25/vectorial."""
from .config import Settings

__all__ = ["Settings", "RAGSystem", "IngestionPipeline", "evaluar"]


def __getattr__(name):
    # Import perezoso: evita cargar LangChain/torch al importar solo Settings.
    if name == "RAGSystem":
        from .rag_system import RAGSystem
        return RAGSystem
    if name == "IngestionPipeline":
        from .ingestion import IngestionPipeline
        return IngestionPipeline
    if name == "evaluar":
        from .evaluation.runner import evaluar
        return evaluar
    raise AttributeError(name)
