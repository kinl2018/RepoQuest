import numpy as np
import pytest
from embedding_runtime import IsolatedEmbeddingModel


def worker_model(tmp_path, code, timeout=10):
    script = tmp_path / "worker.py"
    script.write_text(code, encoding="utf-8")
    return IsolatedEmbeddingModel("fake", worker_path=script, log_dir=tmp_path, timeout=timeout)


def test_embedding_worker_reuses_process_and_handles_strings_and_batches(tmp_path):
    model = worker_model(tmp_path, '''import json,sys
for line in sys.stdin:
    data = json.loads(line)
    texts = data["texts"]
    print(json.dumps({"vectors": [1.,0.] if isinstance(texts,str) else [[1.,0.] for _ in texts]}),flush=True)
''')
    try:
        assert model.encode("one").tolist() == [1,0]
        pid = model._process.pid
        assert model.encode(["one", "two"]).shape == (2,2)
        assert model._process.pid == pid
    finally:
        model.close()
    assert model._process is None


def test_native_worker_exit_is_reported_without_exiting_parent(tmp_path):
    model = worker_model(tmp_path, "import os\nos._exit(7)\n")
    try:
        with pytest.raises(RuntimeError, match="退出码 7"):
            model.encode("query")
        assert model._process is None
        assert model.log_path.is_file()
    finally:
        model.close()


def test_worker_timeout_can_be_retried(tmp_path):
    model = worker_model(tmp_path, "import time\ntime.sleep(30)\n", timeout=0.2)
    try:
        with pytest.raises(RuntimeError, match="计算超时"):
            model.encode("query")
        assert model._process is None
    finally:
        model.close()


def test_worker_python_error_is_actionable_and_process_remains_available(tmp_path):
    model = worker_model(tmp_path, '''import sys,json
for line in sys.stdin:
    print(json.dumps({"error":"model not found"}),flush=True)
''')
    try:
        with pytest.raises(RuntimeError, match="model not found"):
            model.encode("query")
        assert model._process.poll() is None
    finally:
        model.close()
