"""Compatibility import; the production workflow lives in repo_agent.py."""
from pathlib import Path
from repo_agent import RepoState
from repo_agent import build_graph as _build_graph
from retrieval import Retriever


def build_graph(documents, embeddings, model, bm25=None, *, root=None, **options):
    retriever = Retriever(documents, embeddings, model, root=Path(root or Path.cwd()).resolve())
    return _build_graph(retriever, **options)
