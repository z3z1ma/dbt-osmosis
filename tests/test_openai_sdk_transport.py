"""Exercise optional SDK serialization with an in-process HTTP transport only."""

import importlib
import json
import socket
from functools import partial

import pytest

from dbt_osmosis.core import llm
from dbt_osmosis.core.exceptions import LLMResponseError

openai = pytest.importorskip("openai")
http = importlib.import_module("httpx2" if int(openai.__version__.split(".")[0]) >= 3 else "httpx")


@pytest.fixture(params=["openai", "azure-openai"])
def sdk_transport(request, monkeypatch):
    """Use real SDK clients; never permit a network connection or ambient credentials."""

    def reject_network(*args, **kwargs):
        raise AssertionError("SDK transport tests must never connect to the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setenv("LLM_PROVIDER", request.param)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "offline-model")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://offline.invalid")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "offline-model")
    replies = []
    calls = []

    def handle(req):
        body = json.loads(req.content)
        assert body["model"] == "offline-model"
        assert body["messages"]
        assert req.url.path.endswith("/chat/completions")
        calls.append(body)
        status, content = replies.pop(0)
        if status == 429:
            return http.Response(status, json={"error": {"message": "retry", "type": "rate_limit"}})
        return http.Response(
            status,
            json={
                "id": "offline-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "offline-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    with http.Client(transport=http.MockTransport(handle)) as client:
        monkeypatch.setattr(
            llm, "OpenAI", partial(openai.OpenAI, http_client=client, max_retries=0)
        )
        monkeypatch.setattr(
            llm, "AzureOpenAI", partial(openai.AzureOpenAI, http_client=client, max_retries=0)
        )
        yield replies, calls
        assert not replies


def test_real_sdk_column_documentation(sdk_transport):
    replies, calls = sdk_transport
    replies.append((200, "  Customer identifier.  "))
    assert llm.generate_column_doc("customer_id", table_name="customers") == "Customer identifier."
    assert len(calls) == 1


def test_real_sdk_json_documentation(sdk_transport):
    replies, _ = sdk_transport
    expected = {"description": "Customers", "columns": [{"name": "id", "description": "ID"}]}
    replies.append((200, json.dumps(expected)))
    assert llm.generate_model_spec_as_json("select id from customers") == expected


def test_real_sdk_empty_response(sdk_transport):
    replies, _ = sdk_transport
    replies.append((200, None))
    with pytest.raises(LLMResponseError, match="empty response"):
        llm.generate_column_doc("id")


def test_real_sdk_rate_limit_retry(sdk_transport, monkeypatch):
    replies, calls = sdk_transport
    replies.extend([(429, None), (200, "Customer identifier.")])
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    assert llm.generate_column_doc("id") == "Customer identifier."
    assert len(calls) == 2
