"""Smoke-test the local environment.

Run after installing dependencies:

    .venv\\Scripts\\python.exe scripts\\verify_env.py

Checks that every core dependency imports, that torch is the CPU build, that
caches point to the project folder, and that ChromaDB can persist and query
vectors on this machine. It does NOT download ML models or call Groq.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
import time
from pathlib import Path

from researchmate.config import configure_runtime_environment, ensure_directories, get_settings

# Must run before importing chromadb / huggingface libraries.
settings = get_settings()
configure_runtime_environment(settings)
ensure_directories(settings)

CORE_MODULES = [
    "fastapi", "uvicorn", "multipart", "pydantic", "pydantic_settings",
    "sqlalchemy", "chromadb", "pymupdf", "docx", "sentence_transformers",
    "rank_bm25", "langgraph", "openai", "streamlit", "httpx", "torch",
]


def check_imports() -> bool:
    ok = True
    for name in CORE_MODULES:
        start = time.perf_counter()
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # report every failure, not just the first
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            ok = False
            continue
        version = getattr(module, "__version__", "")
        print(f"  ok    {name:<22} {str(version):<14} ({time.perf_counter() - start:.1f}s)")
    return ok


def check_torch() -> bool:
    import torch

    cpu_build = torch.version.cuda is None
    print(f"  torch {torch.__version__}  cuda_build={torch.version.cuda}  "
          f"threads={torch.get_num_threads()}")
    return cpu_build


def check_paths() -> bool:
    import os

    hf_home = Path(os.environ["HF_HOME"])
    print(f"  python   = {sys.executable}")
    print(f"  data_dir = {settings.data_dir}")
    print(f"  HF_HOME  = {hf_home}")
    in_venv = Path(sys.prefix).resolve() != Path(sys.base_prefix).resolve()
    on_project_drive = hf_home.drive.upper() == Path(__file__).resolve().drive.upper()
    print(f"  running inside venv={in_venv}  hf_cache_on_project_drive={on_project_drive}")
    return in_venv and on_project_drive


def check_chroma() -> bool:
    """Persist 3 vectors, reopen the database from disk, query it, then clean up."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    tmp = Path(tempfile.mkdtemp(prefix="chroma-smoke-", dir=settings.data_dir))
    client = None
    try:
        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=str(tmp), settings=chroma_settings)
        col = client.create_collection(
            "smoke", embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
            documents=["alpha", "beta", "alpha-ish"],
            metadatas=[{"page": 1}, {"page": 2}, {"page": 3}],
        )
        client.close()
        # Reopen from disk to prove the data was persisted, not just held in memory.
        client = chromadb.PersistentClient(path=str(tmp), settings=chroma_settings)
        res = client.get_collection("smoke").query(
            query_embeddings=[[1.0, 0.0, 0.0]], n_results=2, include=["metadatas"]
        )
        ids, pages = res["ids"][0], [m["page"] for m in res["metadatas"][0]]
        print(f"  chromadb {chromadb.__version__}: nearest ids={ids} pages={pages}")
        return ids == ["a", "c"] and pages == [1, 3]
    finally:
        # Windows cannot delete files that are still open, so release Chroma's
        # SQLite/index handles first, then report (not hide) any cleanup failure.
        if client is not None:
            client.close()
        try:
            shutil.rmtree(tmp)
        except OSError as exc:
            print(f"  WARN  could not remove {tmp}: {exc}")


def main() -> int:
    results = {}
    for title, fn in [("imports", check_imports), ("torch", check_torch),
                      ("paths", check_paths), ("chromadb", check_chroma)]:
        print(f"[{title}]")
        try:
            results[title] = fn()
        except Exception as exc:
            print(f"  FAIL  {type(exc).__name__}: {exc}")
            results[title] = False
    print("\nSUMMARY: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
