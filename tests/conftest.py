from pathlib import Path
import numpy as np
import pytest
from langchain_core.documents import Document
from code_chunker import chunk_repo
from rag_documents import chunk_to_document
from retrieval import Retriever


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "lib.py").write_text('def leaf():\n    return 7\n\ndef middle():\n    return leaf()\n', encoding="utf-8")
    (tmp_path / "app.py").write_text('from lib import middle as run\n\ndef entry():\n    return run()\n', encoding="utf-8")
    (tmp_path / "other.py").write_text('class Worker:\n    def leaf(self):\n        return 9\n    def work(self):\n        return self.leaf()\n', encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_lib.py").write_text('from lib import leaf\n\ndef test_leaf():\n    assert leaf() == 7\n', encoding="utf-8")
    return tmp_path


@pytest.fixture
def retriever(sample_repo):
    chunks, errors = chunk_repo(sample_repo)
    assert not errors
    return Retriever([chunk_to_document(c) for c in chunks], root=sample_repo)


def make_doc(name, file="lib.py", start=1, content=None, kind="function"):
    content = content or f"def {name}():\n    return 7"
    return Document(page_content=content, metadata={"file": file, "name": name, "kind": kind,
                    "start_line": start, "end_line": start+len(content.splitlines())-1})


class FixedModel:
    def encode(self, text, **kwargs):
        return np.array([1.0, 0.0], dtype=np.float32)
