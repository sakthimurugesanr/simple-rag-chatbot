"""Sourcebook: an OpenRouter document chat workspace with per-browser-session state."""

import hashlib
import html
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from app.chat import (
    DEFAULT_MODEL,
    NO_ANSWER,
    ChatError,
    OpenRouterClient,
    build_messages,
    list_models,
    retrieval_question,
    retrieve,
)
from app.config import PROJECT_ROOT, Settings
from app.exceptions import IngestionError
from app.ingestion.chunker import chunk_documents
from app.ingestion.loaders import load_file
from app.utils.security import sanitized_filename

st.set_page_config(page_title="Sourcebook · Document chat", page_icon="✳", layout="wide")

st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(ellipse at 85% 0%,#17333155 0%,transparent 43%),#0b111b;}
[data-testid="stSidebar"] {background:#101927;border-right:1px solid #ffffff0b;}
.block-container {max-width:1160px;padding-top:2rem;padding-bottom:5rem;}
h1 {font-size:2.75rem!important;font-weight:600!important;letter-spacing:-1.7px!important;}
h2,h3 {letter-spacing:-.5px;}
.eyebrow {font-size:11px;font-weight:700;letter-spacing:2px;color:#8ba19f;margin-bottom:18px;}
.brand {display:flex;align-items:center;gap:11px;font-size:23px;font-weight:700;margin-bottom:5px;}
.brand-icon {background:#56d6b3;color:#0b211b;border-radius:10px;padding:3px 11px;font-size:29px;}
.subtle {color:#94a3b8;font-size:14px;line-height:1.7;}
.hero {padding:26px 0 20px;}
.hero h1 {margin:0 0 12px;}
.hero p {max-width:650px;color:#9cacc1;line-height:1.8;font-size:16px;}
.pill {display:inline-block;border:1px solid #56d6b333;background:#56d6b30c;color:#70dec0;
padding:5px 12px;border-radius:30px;font-size:11px;letter-spacing:.7px;}
.step {color:#56d6b3;font-size:11px;letter-spacing:1.2px;font-weight:700;margin-bottom:6px;}
.card-title {font-size:17px;font-weight:600;color:#e7edf5;margin-bottom:7px;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#ffffff12!important;border-radius:16px!important;}
[data-testid="stChatMessage"] {border:1px solid #ffffff0c;border-radius:16px;background:#121d2a;}
[data-testid="stMetricValue"] {font-size:26px;color:#d9f7ed;}
[data-testid="stMetricLabel"] {color:#92a4bb;}
div.stButton>button {border-radius:9px;min-height:40px;}
div.stButton>button[kind="primary"] {background:#56d6b3;color:#0a241c;font-weight:700;border:0;}
hr {border-color:#ffffff0d!important;}
footer {visibility:hidden;}
@media(max-width:700px){h1{font-size:2rem!important;}.block-container{padding-top:1rem;}}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600, show_spinner=False)
def catalog() -> list[dict[str, Any]]:
    """Cache only the public model catalog, never credentials or document content."""
    return list_models()


def clear_chat() -> None:
    st.session_state.messages = []


def add_file(path: Path, name: str, settings: Settings, mime: str | None = None) -> bool:
    """Extract and index into this browser session; hashes deduplicate repeated uploads."""
    units = load_file(path, settings, source_name=name, mime_type=mime)
    doc_id = str(units[0].metadata["doc_id"])
    if doc_id in st.session_state.documents:
        return False
    chunks = chunk_documents(units, settings.chunk_size, settings.chunk_overlap)
    # Per-session caps bound rerun cost and memory independently of per-file limits.
    total = sum(len(record["chunks"]) for record in st.session_state.documents.values())
    if total + len(chunks) > 10000:
        raise ChatError(
            "This workspace is limited to 10,000 chunks. Remove documents or use larger chunks."
        )
    st.session_state.documents[doc_id] = {
        "name": name,
        "chunks": chunks,
        "type": path.suffix[1:].upper(),
    }
    clear_chat()
    return True


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)} passages", expanded=False):
        for index, source in enumerate(sources, 1):
            metadata = source["metadata"]
            coordinates = [f"Page {metadata['page']}" for _ in [0] if "page" in metadata]
            if "sheet" in metadata:
                coordinates.append(
                    f"{metadata['sheet']} · rows {metadata['row_start']}–{metadata['row_end']}"
                )
            st.markdown(f"**[{index}] {html.escape(str(metadata.get('source', 'Document')))}**")
            st.caption(" · ".join([*coordinates, f"Similarity {source['score']:.0%}"]))
            st.text(source["text"][:900])


def main() -> None:
    settings = Settings()
    for key, default in {
        "documents": {},
        "messages": [],
        "selection": None,
        "catalog_loaded": False,
        "models": [],
        "catalog_error": None,
        "upload_errors": [],
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if not st.session_state.catalog_loaded:
        try:
            st.session_state.models = catalog()
        except ChatError as exc:
            st.session_state.catalog_error = str(exc)
        st.session_state.catalog_loaded = True

    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-icon">✳</span> Sourcebook</div>'
            '<div class="subtle">Your documents. Clear answers.</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("##### Model workspace")
        st.caption("PROVIDER")
        st.selectbox("Provider", ["OpenRouter"], label_visibility="collapsed")
        env_key = (
            settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else ""
        )
        api_key = st.text_input(
            "OpenRouter API key",
            value=env_key,
            type="password",
            placeholder="sk-or-v1-…",
            key="api_key",
        )
        st.caption("Your key stays in memory for this session.")
        only_free = st.toggle("Show free models only", value=True)
        models = st.session_state.models
        model_map = {item["id"]: item for item in models}
        options = [
            item["id"]
            for item in models
            if not only_free
            or (
                float(item.get("pricing", {}).get("prompt", "1")) == 0
                and float(item.get("pricing", {}).get("completion", "1")) == 0
            )
        ]
        default_model = settings.llm_model or DEFAULT_MODEL
        options = list(dict.fromkeys([default_model, *sorted(options)]))

        def model_label(model_id: str) -> str:
            item = model_map.get(model_id, {})
            label = item.get("name", model_id)
            return f"{label} · FREE" if model_id.endswith(":free") else label

        selected_model = st.selectbox(
            "Chat model", options, format_func=model_label, key="model_selection"
        )
        with st.expander("Use a custom model ID"):
            custom_model = st.text_input("OpenRouter model ID", placeholder="provider/model:free")
        model = custom_model.strip() or selected_model
        st.caption(model)
        refresh, test = st.columns(2)
        if refresh.button("↻ Models", use_container_width=True):
            catalog.clear()
            st.session_state.catalog_loaded = False
            st.rerun()
        if test.button("Test key", use_container_width=True):
            try:
                OpenRouterClient(api_key).test_connection()
                st.success("API key connected")
            except ChatError as exc:
                st.error(str(exc))
        if st.session_state.catalog_error and not models:
            st.caption(st.session_state.catalog_error)
        if models and model not in model_map:
            st.warning("This ID is not in the live catalog. It may be unavailable.")
        reasoning = st.toggle(
            "Enable reasoning",
            value=True,
            help="Keeps returned reasoning blocks intact for same-model follow-ups.",
        )
        if reasoning:
            st.caption(
                "Complete responses preserve reasoning details. Turn off for live text streaming."
            )
        with st.expander("Generation & retrieval"):
            temperature = st.slider("Temperature", 0.0, 1.0, settings.temperature, 0.05)
            max_tokens = st.select_slider(
                "Output token budget", [512, 1024, 2048, 4096, 8192], value=4096
            )
            top_k = st.slider("Source passages", 1, 10, 5)
            threshold = st.slider("Minimum similarity", 0.0, 0.8, 0.12, 0.01)
            chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100)
            overlap = st.slider("Chunk overlap", 0, 250, 150, 25)
            st.caption(
                "Local keyword retrieval (TF-IDF cosine). Chunk settings apply to new uploads."
            )
        ingest_settings = settings.model_copy(
            update={"chunk_size": chunk_size, "chunk_overlap": overlap}
        )
        st.divider()
        st.markdown("##### Session")
        if st.button("＋ New conversation", use_container_width=True):
            clear_chat()
            st.rerun()
        export = [
            {key: turn[key] for key in ("role", "content", "model", "sources") if key in turn}
            for turn in st.session_state.messages
        ]
        st.download_button(
            "↓ Export chat · JSON",
            json.dumps(export, ensure_ascii=False, indent=2),
            "sourcebook-chat.json",
            "application/json",
            use_container_width=True,
        )
        markdown = "\n\n".join(f"## {turn['role'].title()}\n\n{turn['content']}" for turn in export)
        st.download_button(
            "↓ Export chat · Markdown",
            markdown,
            "sourcebook-chat.md",
            "text/markdown",
            use_container_width=True,
        )
        st.caption(
            "Documents and conversation stay in this session. Selected source passages are sent to OpenRouter when you ask a question."
        )

    st.markdown(
        '<div class="eyebrow">SOURCEBOOK / RESEARCH WORKSPACE</div>', unsafe_allow_html=True
    )
    st.markdown('<span class="pill">✦ DOCUMENT-GROUNDED CHAT</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>Make sense of what you know.</h1>'
        "<p>Bring your documents. Ask a better question. Get clear answers with "
        "the original sources just one click away.</p></div>",
        unsafe_allow_html=True,
    )
    docs = st.session_state.documents
    columns = st.columns(3)
    columns[0].metric("Documents", len(docs))
    columns[1].metric("Indexed passages", sum(len(record["chunks"]) for record in docs.values()))
    columns[2].metric(
        "Conversation turns", sum(turn["role"] == "user" for turn in st.session_state.messages)
    )
    st.write("")
    library_tab, chat_tab = st.tabs(["✳  Workspace", "☰  Document library"])
    with chat_tab:
        st.subheader("Your source library")
        if not docs:
            st.info("Add files in the Workspace tab or load the sample documents to begin.")
        for doc_id, record in list(docs.items()):
            with st.container(border=True):
                name_col, info_col, delete_col = st.columns([5, 2, 1])
                name_col.write(record["name"])
                info_col.caption(f"{record['type']} · {len(record['chunks'])} passages")
                if delete_col.button("Delete", key=f"delete_{doc_id}"):
                    del docs[doc_id]
                    clear_chat()
                    st.rerun()
        if docs and st.button("Reset workspace", type="secondary"):
            docs.clear()
            clear_chat()
            st.rerun()

    with library_tab:
        with st.container(border=True):
            st.markdown(
                '<div class="step">01 / BUILD YOUR CONTEXT</div>'
                '<div class="card-title">Add your source material</div>',
                unsafe_allow_html=True,
            )
            upload_col, sample_col = st.columns([3, 2])
            with upload_col:
                for error in st.session_state.upload_errors:
                    st.error(error)
                uploads = st.file_uploader(
                    "Drop documents here",
                    accept_multiple_files=True,
                    type=["pdf", "docx", "txt", "md", "xlsx", "xls", "csv"],
                    key="uploads",
                )
                if st.button("Index uploaded files", type="primary", disabled=not uploads):
                    st.session_state.upload_errors = []
                    progress = st.progress(0, text="Reading your documents…")
                    for index, upload in enumerate(uploads):
                        try:
                            if upload.size > settings.max_upload_mb * 1024**2:
                                raise IngestionError("File exceeds the 25 MB upload limit")
                            with tempfile.TemporaryDirectory(prefix="sourcebook_") as directory:
                                path = Path(directory) / sanitized_filename(upload.name)
                                path.write_bytes(upload.getbuffer())
                                added = add_file(path, upload.name, ingest_settings, upload.type)
                            st.toast(f"{'Indexed' if added else 'Already indexed'}: {upload.name}")
                        except (IngestionError, ChatError, OSError) as exc:
                            st.session_state.upload_errors.append(f"{upload.name}: {exc}")
                        progress.progress((index + 1) / len(uploads))
                    st.rerun()
            with sample_col:
                st.markdown("**Take it for a spin**")
                st.caption(
                    "Load a sample employee handbook and product catalog to explore citations and tables."
                )
                if st.button("Load sample documents", use_container_width=True):
                    for name in ("handbook.md", "products.csv"):
                        add_file(PROJECT_ROOT / "samples" / name, name, ingest_settings)
                    st.rerun()
                st.caption("PDF · Word · Excel · CSV · Text · Markdown\n\nUp to 25 MB per file")
        labels = {key: record["name"] for key, record in docs.items()}
        selected = st.multiselect(
            "Search within",
            list(labels),
            default=list(labels),
            format_func=lambda value: labels[value],
            placeholder="Add documents to choose sources",
        )
        fingerprint = hashlib.sha256("|".join(sorted(selected)).encode()).hexdigest()
        if st.session_state.selection is not None and st.session_state.selection != fingerprint:
            clear_chat()
        st.session_state.selection = fingerprint
        active = [chunk for doc_id in selected for chunk in docs[doc_id]["chunks"]]
        st.divider()
        st.markdown('<div class="step">02 / FOLLOW YOUR CURIOSITY</div>', unsafe_allow_html=True)
        st.subheader("Ask your documents")
        if not st.session_state.messages:
            st.caption(
                "Answers stay grounded in your selected sources. Switch models anytime in the sidebar."
            )
            a, b, c = st.columns(3)
            suggestions = [
                "How many days of annual leave do employees receive?",
                "What is the remote work policy?",
                "Compare the product prices and stock.",
            ]
            prompt = None
            for column, suggestion in zip((a, b, c), suggestions):
                if column.button(suggestion, use_container_width=True, disabled=not active):
                    prompt = suggestion
        else:
            prompt = None
        for turn in st.session_state.messages:
            with st.chat_message(turn["role"], avatar="✳" if turn["role"] == "assistant" else None):
                st.markdown(turn["content"])
                if turn["role"] == "assistant":
                    st.caption(turn.get("model", "Document search"))
                    render_sources(turn.get("sources", []))
                    with st.expander("Copy answer"):
                        st.code(turn["content"], language=None)
        typed = st.chat_input("Ask a question about your selected documents…", disabled=not active)
        prompt = typed or prompt
        if not active:
            st.info("Start by adding documents, then select the sources you want to chat with.")
        elif not api_key:
            st.info(
                "Ready to chat. Enter your OpenRouter API key in the sidebar. Get one at https://openrouter.ai/keys"
            )
        if prompt:
            history = st.session_state.messages
            question = retrieval_question(prompt, history)
            hits = retrieve(active, question, top_k, threshold)
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                sources: list[dict[str, Any]] = []
                message: dict[str, Any] = {"content": NO_ANSWER}
                if hits:
                    if not api_key:
                        raise ChatError(
                            "Enter your OpenRouter API key in the sidebar, then ask again."
                        )
                    model_context = int(model_map.get(model, {}).get("context_length", 32768))
                    budget = min(24000, model_context - max_tokens - 1024)
                    if budget < 2000:
                        raise ChatError(
                            "Reduce the output token budget for this model's context size."
                        )
                    messages, included = build_messages(prompt, hits, history, model, budget)
                    sources = [
                        {"text": hit.chunk.text, "metadata": hit.chunk.metadata, "score": hit.score}
                        for hit in included
                    ]
                    client = OpenRouterClient(api_key)
                    with st.chat_message("assistant", avatar="✳"):
                        if reasoning:
                            with st.spinner("Reading sources and preparing your answer…"):
                                message = client.generate(
                                    messages, model, temperature, max_tokens, reasoning=True
                                )
                            st.markdown(message["content"])
                        else:
                            message = {
                                "content": st.write_stream(
                                    client.stream(messages, model, temperature, max_tokens)
                                )
                            }
                            if not message["content"]:
                                raise ChatError(
                                    "The model returned an empty answer. Try another model."
                                )
                        render_sources(sources)
                history.extend(
                    [
                        {"role": "user", "content": prompt},
                        {
                            "role": "assistant",
                            "content": message["content"],
                            "model": model,
                            "reasoning_details": message.get("reasoning_details"),
                            "sources": sources,
                            "created_at": datetime.now(UTC).isoformat(),
                        },
                    ]
                )
                st.rerun()
            except ChatError as exc:
                st.error(str(exc))
                st.caption(
                    "Your conversation was kept. You can change models and retry your question."
                )


if __name__ == "__main__":
    main()
