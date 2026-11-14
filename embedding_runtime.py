"""Keep PyTorch/transformers native code outside the Streamlit server process."""
import atexit
from functools import lru_cache
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from threading import Lock, Thread
from uuid import uuid4
import numpy as np


def _read_responses(stream, responses):
    try:
        for line in stream:
            responses.put(line)
    finally:
        responses.put(None)


class IsolatedEmbeddingModel:
    def __init__(self, model_name, *, threads=4, timeout=180, log_dir=None, worker_path=None):
        if threads < 1 or timeout <= 0:
            raise ValueError("Embedding 线程数和超时必须大于 0")
        self.model_name, self.threads, self.timeout = model_name, threads, timeout
        self.log_dir = Path(log_dir or Path(__file__).parent / ".repoquest_runs")
        self.worker_path = Path(worker_path or Path(__file__).with_name("embedding_worker.py"))
        self._lock = Lock()
        self._process = None
        self._log = None
        self.log_path = None
        atexit.register(self.close)

    def _start(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"embedding-{uuid4().hex}.log"
        self._log = self.log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONFAULTHANDLER": "1",
                    "OMP_NUM_THREADS": str(self.threads), "MKL_NUM_THREADS": str(self.threads),
                    "OPENBLAS_NUM_THREADS": str(self.threads), "TOKENIZERS_PARALLELISM": "false"})
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-u", str(self.worker_path), self.model_name, str(self.threads)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                text=True, encoding="utf-8", bufsize=1, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except BaseException:
            self._log.close()
            self._log = None
            raise
        self._responses = Queue()
        Thread(target=_read_responses, args=(self._process.stdout, self._responses), daemon=True).start()

    def _failure(self, reason):
        self.close()
        return RuntimeError(f"Embedding 子进程{reason}，可重试；网站服务不受该子进程退出影响。日志：{self.log_path}")

    def encode(self, texts, **options):
        with self._lock:
            if self._process is None:
                self._start()
            try:
                self._process.stdin.write(json.dumps({"texts": texts, "options": options}, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise self._failure("连接已关闭") from exc
            try:
                line = self._responses.get(timeout=self.timeout)
            except Empty as exc:
                raise self._failure(f"计算超时（{self.timeout}s）") from exc
            if line is None:
                self._process.wait(timeout=5)
                code = self._process.returncode
                raise self._failure(f"异常退出（退出码 {code}）")
            try:
                response = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise self._failure("返回数据无效") from exc
            if "error" in response:
                raise RuntimeError(f"Embedding 计算失败：{response['error']}。日志：{self.log_path}")
            return np.asarray(response["vectors"], dtype=np.float32)

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        if self._log is not None:
            self._log.close()
            self._log = None


@lru_cache(maxsize=2)
def get_embedding_model(model_name):
    return IsolatedEmbeddingModel(model_name, threads=int(os.getenv("EMBEDDING_THREADS", "4")),
                                  timeout=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180")))
