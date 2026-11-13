from retrieval import Retriever


def semantic_search(question, documents, embeddings, model, k=3):
    return Retriever(documents, embeddings, model).search(question, k, strategy="vector")
