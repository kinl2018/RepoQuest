"""JSONL worker. Native model failures terminate this process, not the web server."""
from contextlib import redirect_stdout
import faulthandler
import json
import sys
import traceback


def main():
    faulthandler.enable()
    model = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                if model is None:
                    import torch
                    torch.set_num_threads(int(sys.argv[2]))
                    from sentence_transformers import SentenceTransformer
                    try:
                        model = SentenceTransformer(sys.argv[1], device="cpu", local_files_only=True)
                    except OSError:
                        model = SentenceTransformer(sys.argv[1], device="cpu")
                vectors = model.encode(request["texts"], **request["options"])
            response = {"vectors": vectors.tolist()}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
