"""Verify request auth, intact reasoning round trips, retrieval and actual UI workflows."""

import copy
import json
from typing import Any

import pytest

from app.chat import (
    DEFAULT_MODEL,
    ChatError,
    OpenRouterClient,
    build_messages,
    cosine_similarity,
    retrieval_question,
    retrieve,
)
from app.config import PROJECT_ROOT
from app.ingestion.models import Chunk


class Response:
    def __init__(
        self, body: Any = None, status: int = 200, lines: list[bytes] | None = None
    ) -> None:
        self.status_code = status
        self.body = body
        self.lines = lines or []
        self.closed = False

    def json(self) -> Any:
        return self.body

    def iter_lines(self) -> Any:
        return iter(self.lines)

    def close(self) -> None:
        self.closed = True


class Transport:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Response:
        self.calls.append({"method": method, "url": url, **copy.deepcopy(kwargs)})
        return self.responses.pop(0)


def passages() -> list[Chunk]:
    return [
        Chunk(
            "Annual leave allowance: 24 days. Manager approval is required.",
            {"source": "handbook.md", "doc_id": "a"},
        ),
        Chunk(
            "Product: keyboard; price INR: 2500; stock: 12.",
            {"source": "products.csv", "doc_id": "b"},
        ),
    ]


def test_cosine_math() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)
    assert cosine_similarity([0, 0], [1, 2]) == 0
    with pytest.raises(ValueError):
        cosine_similarity([1], [1, 2])


def test_retrieval_threshold_and_no_hits() -> None:
    hits = retrieve(passages(), "How many days of annual leave?", threshold=0.12)
    assert hits[0].chunk.metadata["doc_id"] == "a"
    assert retrieve(passages(), "ocean tides lunar cycle") == []
    assert retrieve(passages(), "leave", threshold=1) == []
    assert retrieve([], "leave") == []


def test_preserve_reasoning_exactly_and_both_requests_authenticated() -> None:
    details = [{"type": "reasoning.encrypted", "id": "r1", "data": "opaque-payload", "index": 0}]
    assistant = {"role": "assistant", "content": "24 days [1]", "reasoning_details": details}
    transport = Transport(
        [
            Response({"choices": [{"message": assistant}]}),
            Response({"choices": [{"message": assistant}]}),
        ]
    )
    client = OpenRouterClient("test-key", transport)
    hits = retrieve(passages(), "annual leave")
    first, _ = build_messages("Annual leave?", hits, [], DEFAULT_MODEL)
    result = client.generate(first)
    history = [{"role": "user", "content": "Annual leave?"}, {**result, "model": DEFAULT_MODEL}]
    second, _ = build_messages("Are you sure?", hits, history, DEFAULT_MODEL)
    client.generate(second)
    assert second[2]["reasoning_details"] == details
    assert second[2]["reasoning_details"] is not details
    assert result == assistant
    assert all(call["headers"]["Authorization"] == "Bearer test-key" for call in transport.calls)
    assert all(call["json"]["reasoning"] == {"enabled": True} for call in transport.calls)
    assert transport.calls[1]["json"]["messages"][2]["reasoning_details"] == details


def test_switch_models_preserves_text_excludes_incompatible_reasoning() -> None:
    history = [
        {
            "role": "assistant",
            "content": "24 days",
            "model": "previous/model",
            "reasoning_details": [{"data": "opaque"}],
        }
    ]
    messages, _ = build_messages("Leave?", retrieve(passages(), "leave"), history, DEFAULT_MODEL)
    assert messages[1] == {"role": "assistant", "content": "24 days"}


@pytest.mark.parametrize(
    "status,match",
    [(401, "API key"), (402, "credits"), (404, "unavailable"), (429, "rate limited")],
)
def test_safe_provider_errors(status: int, match: str) -> None:
    transport = Transport([Response(status=status) for _ in range(3)])
    with pytest.raises(ChatError, match=match):
        OpenRouterClient("secret-value", transport, sleep=lambda _: None).generate([])


def test_rate_limit_backoff_recovers() -> None:
    delays: list[float] = []
    transport = Transport(
        [Response(status=429), Response({"choices": [{"message": {"content": "ok"}}]})]
    )
    assert OpenRouterClient("key", transport, delays.append).generate([])["content"] == "ok"
    assert delays == [1]


def test_missing_key_never_calls_network() -> None:
    transport = Transport([])
    with pytest.raises(ChatError, match="API key"):
        OpenRouterClient("", transport).generate([])
    assert transport.calls == []


def test_stream_text_and_auth() -> None:
    event = {"choices": [{"delta": {"content": "24 days"}}]}
    response = Response(lines=[b"data: " + json.dumps(event).encode(), b"data: [DONE]"])
    transport = Transport([response])
    assert (
        "".join(OpenRouterClient("key", transport).stream([], DEFAULT_MODEL, 0.2, 512)) == "24 days"
    )
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer key"
    assert response.closed


def test_context_budget_drops_old_history_without_editing_reasoning() -> None:
    history = [
        {"role": "user", "content": "previous"},
        {
            "role": "assistant",
            "content": "text",
            "model": DEFAULT_MODEL,
            "reasoning_details": [{"data": "x" * 12000}],
        },
    ]
    original = copy.deepcopy(history)
    messages, hits = build_messages(
        "annual leave",
        retrieve(passages(), "annual leave"),
        history,
        DEFAULT_MODEL,
        input_budget=4000,
    )
    assert history == original
    assert len(messages) == 2
    assert len(json.dumps(messages).encode()) < 4000
    assert hits


def test_followup_retrieves_prior_topic() -> None:
    history = [{"role": "user", "content": "Annual leave?"}]
    assert "Annual leave?" in retrieval_question("Are you sure?", history)


def test_streamlit_load_samples_chat_and_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        "app.chat.list_models",
        lambda: [
            {
                "id": DEFAULT_MODEL,
                "name": "Gemma 4",
                "context_length": 262144,
                "pricing": {"prompt": "0", "completion": "0"},
            },
            {
                "id": "openrouter/free",
                "name": "Free router",
                "context_length": 32768,
                "pricing": {"prompt": "0", "completion": "0"},
            },
        ],
    )
    monkeypatch.setattr(
        OpenRouterClient,
        "generate",
        lambda *args, **kwargs: {
            "role": "assistant",
            "content": "Employees receive 24 days [1].",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
        },
    )
    app = AppTest.from_file(PROJECT_ROOT / "ui" / "streamlit_app.py", default_timeout=30).run()
    assert not app.exception
    next(button for button in app.button if button.label == "Load sample documents").click().run()
    assert not app.exception
    assert len(app.session_state["documents"]) == 2
    app.text_input(key="api_key").set_value("test-session-key").run()
    app.chat_input[0].set_value("How many days of annual leave?").run()
    assert not app.exception
    assert app.session_state["messages"][-1]["content"] == "Employees receive 24 days [1]."
    assert app.session_state["messages"][-1]["sources"]
    app.selectbox(key="model_selection").select("openrouter/free").run()
    assert app.session_state["messages"][-1]["content"]
    assert not app.exception
    # An unanswerable question is refused locally without generating invented facts.
    app.chat_input[0].set_value("Explain oceanographic bathymetry").run()
    assert "do not contain" in app.session_state["messages"][-1]["content"]
    next(button for button in app.button if button.label == "＋ New conversation").click().run()
    assert app.session_state["messages"] == []
