"""Compatibility entry point for the Level 9/10 retrieval API."""
from retrieval import Retriever


def retrieve_candidates(question, documents, embeddings, model, bm25=None, k=5, rewritten_question=None):
    # Query rewriting is explicit; retrieval itself needs no LLM call.
    rewritten = rewritten_question or question
    retriever = Retriever(documents, embeddings, model)
    return rewritten, retriever.search(question, k=k, rewritten=rewritten)
