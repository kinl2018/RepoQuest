from pathlib import Path
from langchain_core.documents import Document
from code_chunker import chunk_repo


def chunk_to_document(chunk: dict) -> Document:
    metadata = {
        k: v for k, v in chunk.items()
        if k != "content"
    }

    return Document(
        page_content=chunk["content"],
        metadata=metadata
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    chunks, errors = chunk_repo(root)

    documents = [
        chunk_to_document(chunk)
        for chunk in chunks
    ]

    print(f"CodeChunk 数量：{len(chunks)}")
    print(f"Document 数量：{len(documents)}")
    print(f"解析失败文件数：{len(errors)}")

    if documents:
        doc = documents[0]

        print("\n===== Metadata =====")
        print(doc.metadata)

        print("\n===== Page Content =====")
        print(doc.page_content)

    assert len(chunks) == len(documents)

    for chunk, doc in zip(chunks, documents):
        assert doc.page_content == chunk["content"]
        assert doc.metadata["file"] == chunk["file"]
        assert doc.metadata["start_line"] == chunk["start_line"]

    print("\n全部验证通过！")
