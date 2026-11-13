"""Legacy imports delegate to the shared retrieval implementation."""
from retrieval import Retriever, reciprocal_rank_fusion


def hybrid_search(question, documents, embeddings, model, bm25=None, k=5, candidate_k=40):
    return Retriever(documents, embeddings, model).search(question, k, candidate_k=candidate_k)
