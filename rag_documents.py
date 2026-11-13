from langchain_core.documents import Document


def chunk_to_document(chunk: dict) -> Document:
    metadata = {
        k: v for k, v in chunk.items()
        if k != "content"
    }
    return Document(page_content=chunk["content"], metadata=metadata)
