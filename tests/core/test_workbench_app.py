import importlib
import sys
import types

from dbt_osmosis.core.settings import YamlRefactorContext


class _DataFrame:
    def __init__(self, data=None):
        self.data = data or []
        self.empty = not self.data


class _Dashboard:
    class Item:
        pass


def _component_module(class_name: str, class_value: object | None = None) -> types.ModuleType:
    module = types.ModuleType(class_name.lower())
    setattr(module, class_name, class_value or type(class_name, (), {}))
    return module


def _import_workbench_app_with_stubs(monkeypatch):
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = types.SimpleNamespace()
    streamlit.set_page_config = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "streamlit", streamlit)

    fluence = types.ModuleType("streamlit_elements_fluence")
    fluence.elements = lambda *args, **kwargs: None
    fluence.event = types.SimpleNamespace(Hotkey=lambda *args, **kwargs: None)
    fluence.JSCallback = lambda code: code
    fluence.mui = types.SimpleNamespace()
    fluence.sync = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "streamlit_elements_fluence", fluence)
    monkeypatch.setitem(sys.modules, "feedparser", types.ModuleType("feedparser"))
    pandas = types.ModuleType("pandas")
    pandas.DataFrame = _DataFrame
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    ydata_profiling = types.ModuleType("ydata_profiling")
    ydata_profiling.ProfileReport = type("ProfileReport", (), {})
    monkeypatch.setitem(sys.modules, "ydata_profiling", ydata_profiling)

    monkeypatch.setitem(
        sys.modules,
        "dbt_osmosis.workbench.components.ai_assistant",
        _component_module("AIAssistant"),
    )
    monkeypatch.setitem(
        sys.modules,
        "dbt_osmosis.workbench.components.dashboard",
        _component_module("Dashboard", _Dashboard),
    )
    editor = _component_module("Editor")
    editor.TabName = types.SimpleNamespace(SQL="SQL")
    monkeypatch.setitem(sys.modules, "dbt_osmosis.workbench.components.editor", editor)
    monkeypatch.setitem(
        sys.modules,
        "dbt_osmosis.workbench.components.feed",
        _component_module("RssFeed"),
    )
    monkeypatch.setitem(
        sys.modules,
        "dbt_osmosis.workbench.components.profiler",
        _component_module("Profiler"),
    )
    monkeypatch.setitem(
        sys.modules,
        "dbt_osmosis.workbench.components.renderer",
        _component_module("Renderer"),
    )
    monkeypatch.delitem(sys.modules, "dbt_osmosis.workbench.app", raising=False)
    monkeypatch.delitem(sys.modules, "dbt_osmosis.workbench.components.preview", raising=False)
    app = importlib.import_module("dbt_osmosis.workbench.app")
    monkeypatch.delitem(sys.modules, "dbt_osmosis.workbench.app", raising=False)
    monkeypatch.delitem(sys.modules, "dbt_osmosis.workbench.components.preview", raising=False)
    return app


def test_workbench_compile_preserves_raw_scratch_sql(
    yaml_context: YamlRefactorContext, monkeypatch
) -> None:
    app = _import_workbench_app_with_stubs(monkeypatch)
    app.state.app = types.SimpleNamespace(ctx=yaml_context.project)

    assert app.compile(app.default_prompt) == app.default_prompt


def test_workbench_run_query_uses_default_template_for_raw_scratch_sql(
    yaml_context: YamlRefactorContext, monkeypatch
) -> None:
    app = _import_workbench_app_with_stubs(monkeypatch)
    preview_state = app.Preview.initial_state()
    app.state.app = types.SimpleNamespace(ctx=yaml_context.project)
    compiled_query = app.compile(app.default_prompt)
    app.state.app = types.SimpleNamespace(
        ctx=yaml_context.project,
        compiled_query=compiled_query,
        query_adapter_resp=preview_state["query_adapter_resp"],
        query_result_columns=preview_state["query_result_columns"],
        query_result_df=preview_state["query_result_df"],
        query_result_rows=preview_state["query_result_rows"],
        query_state=preview_state["query_state"],
        query_template=preview_state["query_template"],
    )

    app.run_query()

    assert app.state.app.query_state == "success"
    assert app.state.app.query_result_columns == [
        {"field": "id", "headerName": "ID"},
        {"field": "name", "headerName": "NAME"},
    ]
    assert app.state.app.query_result_rows == [{"id": 1, "name": "hello"}]
