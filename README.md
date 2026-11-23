# Sourcebook — document chat with OpenRouter

A local Streamlit workspace for uploading documents and asking cited questions.
Default model: **`google/gemma-4-31b-it:free`**. Switch models at any time with the
live OpenRouter model picker, or enter a custom model ID.

## Open the running app

Go to **http://127.0.0.1:8501**. The app is bound to this computer only.

1. Enter your **OpenRouter API key** in the sidebar password field. Create one at
   [OpenRouter Keys](https://openrouter.ai/keys). One key works across OpenRouter models.
2. Click **Test key** to check authentication without generating a completion.
3. Click **Load sample documents**, or upload PDF, DOCX, TXT, MD, XLS, XLSX or CSV
   files and click **Index uploaded files**.
4. Select your sources under **Search within**, then ask a question.
5. Expand **Sources** below an answer to inspect passages and similarity scores.

The default free model was present in the live catalog during development. Availability,
capacity and rate limits can change. Use **↻ Models** to refresh the list.
The app does not silently switch to a paid model.

## Start / restart

In this workspace on Windows, run from the project directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

The launcher uses `.venv` if available, then the desktop bundled Python with the
project-local dependencies installed during development. Keep the terminal open
while using the app. Stop a foreground launch with Ctrl+C. The current background
server's PID is in `output/ui.pid`; its logs are `output/ui.*.log`.
To stop that background instance, verify its identity first:

```powershell
$sourcebookPid = [int](Get-Content output/ui.pid)
Get-CimInstance Win32_Process -Filter "ProcessId=$sourcebookPid" |
    Select-Object ProcessId, CommandLine
# If the command line is this project's Python run_ui.py:
Stop-Process -Id $sourcebookPid
```

For a fresh installation with Python 3.11+:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_ui.py
```

On Linux/macOS, use `python3 -m venv .venv`, install the requirements using
`.venv/bin/python -m pip install -r requirements.txt`, then run `.venv/bin/python run_ui.py`.

An API key may optionally come from `.env` or your environment with
`RAG_OPENROUTER_API_KEY`. UI-entered keys are held in memory and are not written to
files or chat exports. Every authenticated request includes Bearer authorization,
including follow-ups.

## Features

- Dark responsive UI, source library, upload progress, example questions and source selection.
- All seven original parsers, size/format validation, content-hash deduplication per session,
  and metadata-preserving chunks. Default input limit: 25 MB per file.
- Local TF-IDF cosine retrieval without model downloads. An adjustable threshold
  rejects questions with no matching passages before calling the model.
- Numbered source context, grounding instructions, conservative context budgets, short history,
  and citations. Source-selection changes clear prior chat to prevent stale context.
- Live catalog, free-model filter, custom model IDs, model switching and API key testing.
- Configurable temperature, output tokens, chunk size/overlap, top-k and minimum similarity.
- Reasoning **on**: complete responses store `reasoning_details` without modifying it
  and send it back for same-model follow-ups. Reasoning is not displayed as evidence
  or exported. Old turns are dropped intact when a context budget is exceeded.
- Reasoning **off**: true SSE text streaming. Switching models keeps conversation text
  and omits incompatible reasoning from the previous model.
- Safe errors for invalid keys, unavailable models, timeouts, credit requirements and
  rate limiting, with bounded backoff on retryable responses.
- Delete documents, reset the workspace, clear chat, copy answers and export Markdown/JSON.

## Privacy and storage

Documents and history live **in the Streamlit browser session on the local server**.
Uploaded files are parsed from a temporary directory that is removed afterward.
The UI does not save a persistent index. Restarting/closing the session can lose
the workspace; export chat before doing so. Keys are not logged. Selected passages,
questions and short history are sent to OpenRouter and its inference provider when
asking a question. The public model catalog is the only shared data cache.

## Verify

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check app tests ui scripts run_ui.py
ruff format --check app tests ui scripts run_ui.py
python -m app.cli samples/handbook.md samples/products.xlsx --preview
```

**62 tests** cover actual parsers, chunk coverage, cosine retrieval/no-hit refusal,
request authentication, reasoning preservation, model switching, rate limits,
streaming and the Streamlit sample → question → cited answer → switch → clear workflow.
Provider responses are mocked in tests; real generation needs your OpenRouter key.

## Structure

```text
app/config.py                Environment settings
app/chat.py                  Retrieval, prompts and OpenRouter client
app/ingestion/               File loaders, common models and chunker
app/utils/                   Validation, safe filenames and logging
ui/streamlit_app.py          Source library and chat UI
run_ui.py / run.ps1          Local launchers
tests/                      Ingestion, chat and UI tests
samples/                    Tiny files in all supported formats
docs/INGESTION_STEP1.md       Original ingestion milestone reference
```

## Current limits

This version uses **keyword TF-IDF retrieval**, not semantic embeddings or Chroma.
Paraphrases and cross-language questions can miss passages. The generation model
is independent of retrieval; switching chat models never mixes embedding spaces.
Session data is not persistent. FastAPI, website ingestion, semantic embedding
providers, persistent Chroma collections, other provider adapters and Docker remain
future work from the original six-stage plan.

Grounding instructions and thresholds reduce unsupported answers but do not guarantee
every claim is correct; inspect citations. Spreadsheet formulas use cached values.
PDF OCR needs `requirements-ocr.txt` and separately installed Tesseract/Poppler;
see the ingestion reference. Intended for trusted local use; no multi-user
authentication or isolated parser workers are included.

References: [OpenRouter reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
[key endpoint](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key),
[live catalog](https://openrouter.ai/api/v1/models).
