from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace


def _load_ai_assistant_module(monkeypatch):
    class FakeMui:
        def __init__(self) -> None:
            self.typography_calls: list[str] = []
            self.chip_labels: list[str] = []
            self.icon = SimpleNamespace(PsychologyOutlined=lambda *args, **kwargs: None)

        def Paper(self, *args, **kwargs):
            return nullcontext()

        def Box(self, *args, **kwargs):
            return nullcontext()

        def Typography(self, text, *args, **kwargs):
            self.typography_calls.append(text)
            return None

        def Chip(self, *args, **kwargs):
            self.chip_labels.append(kwargs["label"])
            return None

    fake_streamlit = ModuleType("streamlit")
    fake_streamlit.session_state = SimpleNamespace(app=SimpleNamespace(model="SCRATCH"))

    fake_mui = FakeMui()
    fake_streamlit_elements = ModuleType("streamlit_elements_fluence")
    fake_streamlit_elements.mui = fake_mui
    fake_streamlit_elements.dashboard = SimpleNamespace(
        Grid=lambda *args, **kwargs: nullcontext(),
        Item=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.setitem(sys.modules, "streamlit_elements_fluence", fake_streamlit_elements)
    monkeypatch.delitem(sys.modules, "dbt_osmosis.workbench.components.dashboard", raising=False)
    monkeypatch.delitem(sys.modules, "dbt_osmosis.workbench.components.ai_assistant", raising=False)

    module = importlib.import_module("dbt_osmosis.workbench.components.ai_assistant")
    return module, fake_mui, fake_streamlit.session_state


def _build_assistant(module):
    assistant = module.AIAssistant.__new__(module.AIAssistant)
    assistant._key = "assistant"
    assistant.title_bar = lambda *args, **kwargs: nullcontext()
    return assistant


def test_selected_model_name_ignores_scratch(monkeypatch):
    module, _fake_mui, state = _load_ai_assistant_module(monkeypatch)

    assert module.AIAssistant._selected_model_name() is None

    state.app.model = None
    assert module.AIAssistant._selected_model_name() is None


def test_selected_model_name_reads_app_state(monkeypatch):
    module, _fake_mui, state = _load_ai_assistant_module(monkeypatch)
    state.app.model = SimpleNamespace(name="orders")

    assert module.AIAssistant._selected_model_name() == "orders"


def test_render_shows_truthful_unavailable_panel_without_top_level_state(monkeypatch):
    module, fake_mui, state = _load_ai_assistant_module(monkeypatch)
    state.app.model = SimpleNamespace(name="orders")
    assistant = _build_assistant(module)

    assistant()

    assert not hasattr(state, "ai_assistant")
    assert "Unavailable" in fake_mui.chip_labels
    assert "Selected model: orders" in fake_mui.typography_calls
    assert any(
        "not wired into the real documentation pipeline" in text
        for text in fake_mui.typography_calls
    )
    assert any(
        "does not generate or accept suggestions" in text
        or "misleading to generate or accept suggestions" in text
        for text in fake_mui.typography_calls
    )
