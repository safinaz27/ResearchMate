# CLAUDE.md: ResearchMate

ResearchMate is an agentic research assistant powered by advanced RAG. Users upload PDF and DOCX papers and ask questions, and every answer must be grounded in retrieved evidence with **traceable citations (document name + page)**. This is a portfolio project: correctness, clarity and honest verification matter more than the number of features.

Current state and history: **read `docs/PROGRESS.md` first**, then the latest `docs/implementation/step_XX_*.md`.

## Working rules (from the user, mandatory)

1. **One step at a time.** Before changing anything, briefly say what you will do and why. Implement **only** that step.
2. After each step, explain the files you changed and the important logic in beginner-friendly but technically accurate language. The user wants to be able to defend the design in a technical interview.
3. **Verify for real.** Run the relevant checks and tests and show the actual output. Never claim something works without running it, and never invent test results or metrics.
4. **Document every completed step** in `docs/implementation/step_NN_<name>.md` with these sections: Objective, Why it matters, Implementation, Files, Code walkthrough, Technical decisions, Dependencies, Verification (real commands and outputs), Issues and solutions, Current status, Next step. Only document work that was actually done and verified.
5. **Update `docs/PROGRESS.md`** after each step: completed steps, verification status, decisions, known issues, next step. Keep it consistent with the step docs.
6. **Stop and wait for explicit approval** before starting the next step, even if it looks trivial.
7. Keep the user involved in technical decisions. If something is unclear, explain the trade-off and recommend the simplest practical option.
8. Commit to git only when the user asks.

## Approved plan (v2): scope rules

- **P0 first.** The end-to-end MVP is: upload → parse (page-aware) → chunk → embed → Chroma + BM25 → RRF → rerank → LangGraph → Groq → cited answer → Streamlit. P1 features (conversation history, comparison, richer fallback, evaluation) start only after the P0 Definition of Done is met. P2 is optional.
- **No Docker, WSL, PostgreSQL or Qdrant in this phase**, even though Docker Desktop happens to run on this machine. They belong to a later migration phase.
- Don't add dependencies or features just because they are interesting. Avoid unnecessary abstractions.
- Don't silently change the approved architecture or rewrite working code without a reason.

## Stack and decisions

| Area | Choice |
|---|---|
| Language | Python 3.11 (Windows) |
| API | FastAPI |
| Orchestration | LangGraph: simple graph `retrieve → rerank → check_evidence → generate → validate_citations`, plus an `insufficient_evidence` branch (basic version in P0) |
| LLM | Groq through its OpenAI-compatible API (`openai` SDK), base URL `https://api.groq.com/openai/v1`. Default `GROQ_MODEL=openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b`. `llama-3.3-70b-versatile` is **deprecated**, don't use it. Check model availability against the live `/models` endpoint, not from memory. |
| Embeddings | `BAAI/bge-small-en-v1.5`, local, CPU |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, local, CPU |
| Vector store | ChromaDB `PersistentClient`, always with `embedding_function=None` (we supply our own vectors), telemetry off |
| Lexical | `rank-bm25`, in-memory index rebuilt from SQLite |
| Fusion | Our own `rrf_fuse()` with k=60 |
| Database | SQLite through SQLAlchemy. Portable types only (UUID as `String(36)`, generic `JSON`, no arrays), so PostgreSQL later only needs a new `DATABASE_URL`. |
| PDF / DOCX | PyMuPDF (AGPL, accepted) / python-docx. DOCX has no pages, so label locations "section N", never as fake pages. |
| UI | Streamlit, talking to the API **only over HTTP** |
| Tests | pytest. Use a **FakeLLM**; the test suite must never call Groq or need `GROQ_API_KEY`. |
| Ingestion | Processed during the upload request in P0, with a `status` column (pending, processing, ready, failed) |

**Architecture principles:**
- **SQLite is the source of truth.** Chroma and BM25 are indexes that can be rebuilt from it.
- **Citations are built from our stored chunk metadata, never from LLM output.** The LLM only produces `[S#]` markers, and the validator removes markers that don't match a retrieved chunk.
- Interfaces (`typing.Protocol`) exist for the LLM client, embedder, vector store and reranker, each with one real implementation plus test fakes. Nothing more elaborate.

## Environment: everything stays on D:

The C: drive is nearly full. **Never write caches, temp files, models or the venv to C:.**

- Project root: `D:\Projects\ResearchMate`. Venv: `.venv` (Python 3.11.3). **Don't use the global Python.**
- pip's cache goes to `.pip-cache` through `.venv\pip.ini`. When installing, also point temp files at D: in the same command:
  ```powershell
  $env:TMP="D:\Projects\ResearchMate\.tmp"; $env:TEMP=$env:TMP; .\.venv\Scripts\python.exe -m pip install <pkg>
  ```
- **torch must be the CPU build.** It comes from `https://download.pytorch.org/whl/cpu`, is deliberately not listed in `pyproject.toml`, and is pinned in `requirements.lock`.
- After adding a dependency: add it to `pyproject.toml`, then regenerate `requirements.lock` (`pip freeze --exclude-editable`, keeping the header with the CPU index line).
- Hugging Face models go to `data\hf_cache`. Call `configure_runtime_environment(settings)` **before** importing `sentence_transformers`, `huggingface_hub` or `chromadb`.
- Runtime data (SQLite, Chroma, uploads, models) lives under `data/`, which git ignores.
- **Windows:** close Chroma clients (`client.close()`) before deleting their folders, because open files can't be deleted.

## Secrets

- `GROQ_API_KEY` lives only in `.env`, which git ignores. The user fills it in. **Never read, print, echo or log `.env` or the key**, and never ask the user to paste it into the chat.
- Settings hold the key as a `SecretStr`. Call `.get_secret_value()` in one place only: where the Groq client is created.
- Logging goes through `researchmate.logging_config`, which redacts secrets. Don't bypass it with `print()` of settings or headers.

## Commands (PowerShell, from the project root)

```powershell
.\.venv\Scripts\python.exe -m pytest -q                 # tests (must pass offline, without a key)
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe scripts\verify_env.py        # environment smoke test
```

## Code conventions

- `src/` layout, package `researchmate`. Type hints and helpful docstrings everywhere. Line length 100 (ruff).
- All tunable values belong in `researchmate/config.py` (`Settings`) and `.env.example`. Business logic never reads `os.environ`.
- In tests, use the `make_settings` and `clean_env` fixtures from `tests/conftest.py`, so tests don't depend on the real `.env` or your shell.
- Clear errors: map exceptions to JSON error responses in the API. No bare `except`, and don't hide failures silently.
