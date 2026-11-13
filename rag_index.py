"""JSON + NumPy index persistence with checksums and source/model provenance."""
import hashlib
import json
import os
from pathlib import Path
from contextlib import contextmanager
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from scanner import scan_python_files


def source_fingerprint(root, *, include_tests=False, include_demos=False):
    digest = hashlib.sha256()
    for file in scan_python_files(root, include_tests=include_tests, include_demos=include_demos):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def _index_staging(index_dir):
    # Python 3.13 TemporaryDirectory creates an owner-only Windows DACL.
    # os.replace preserves that DACL, locking other repository users out.
    # A normal Windows directory inherits the index directory's access rules.
    staging = index_dir / f".staging-{uuid4().hex}"
    staging.mkdir(mode=0o777 if os.name == "nt" else 0o700)
    try:
        yield staging
    finally:
        for name in ("embeddings.npy", "documents.json", "manifest.json"):
            (staging / name).unlink(missing_ok=True)
        staging.rmdir()


def save_index(documents, embeddings, index_dir: Path, *, root=None, model_name=None,
               include_tests=False, include_demos=False):
    index_dir = Path(index_dir)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or len(documents) != len(matrix) or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("Document 与 Embedding 数量、维度或数值无效")
    index_dir.mkdir(parents=True, exist_ok=True)
    # Unique staging files; commit manifest last. A mixed interrupted write fails checksum validation.
    with _index_staging(index_dir) as staging:
        np.save(staging / "embeddings.npy", matrix)
        data = [{"page_content": d.page_content, "metadata": d.metadata} for d in documents]
        (staging / "documents.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        manifest = {"version": 2, "model_name": model_name, "root": str(Path(root).resolve()) if root else None,
                    "include_tests": include_tests, "include_demos": include_demos,
                    "source_fingerprint": source_fingerprint(root, include_tests=include_tests, include_demos=include_demos) if root else None,
                    "checksums": {name: _sha(staging / name) for name in ["documents.json", "embeddings.npy"]}}
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        for name in ["documents.json", "embeddings.npy", "manifest.json"]:
            os.replace(staging / name, index_dir / name)


def read_manifest(index_dir):
    path = Path(index_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_index(index_dir: Path):
    index_dir = Path(index_dir)
    manifest = read_manifest(index_dir)
    for name, expected in manifest.get("checksums", {}).items():
        if name not in {"documents.json", "embeddings.npy"} or _sha(index_dir / name) != expected:
            raise ValueError("索引校验失败，请重建索引")
    matrix = np.load(index_dir / "embeddings.npy", allow_pickle=False)
    data = json.loads((index_dir / "documents.json").read_text(encoding="utf-8"))
    documents = [Document(**item) for item in data]
    if matrix.ndim != 2 or len(documents) != len(matrix) or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("Document 与 Embedding 数量、维度或数值无效")
    return documents, matrix


def assert_current_index(root, index_dir, model_name=None):
    manifest = read_manifest(index_dir)
    if not manifest.get("source_fingerprint"):
        raise ValueError("旧索引没有源码与模型信息，请先重建索引")
    if Path(manifest["root"]).resolve() != Path(root).resolve():
        raise ValueError("索引属于其他仓库，请重建索引")
    fingerprint = source_fingerprint(root, include_tests=manifest["include_tests"], include_demos=manifest["include_demos"])
    if manifest["source_fingerprint"] != fingerprint:
        raise ValueError("源码已变化，请重建索引后再使用向量检索")
    if model_name and manifest.get("model_name") != model_name:
        raise ValueError("Embedding 模型与索引不一致，请重建索引")


def get_index_metadata(index_dir: Path) -> dict:
    index_dir = Path(index_dir)
    matrix = np.load(index_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2:
        raise ValueError("Embedding 矩阵必须是二维数组")
    manifest = read_manifest(index_dir)
    return {"embedding_count": int(matrix.shape[0]), "embedding_dimension": int(matrix.shape[1]),
            "model_name": manifest.get("model_name"), "manifest_version": manifest.get("version", 1)}
