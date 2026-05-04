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


class _CompiledNode:
    def __init__(self, compiled_code: str):
        self.compiled_code = compiled_code


class _FakeDbtContext:
    def __init__(
        self,
        target_name: str,
        *,
        project_dir: str = "/project",
        profiles_dir: str = "/profiles",
        profile_name: str = "default",
        project_name: str = "pkg",
    ) -> None:
        self.config = types.SimpleNamespace(
            project_dir=project_dir,
            profiles_dir=profiles_dir,
            target=target_name,
            profile=profile_name,
            threads=4,
            vars={"feature": "enabled"},
            quiet=False,
            disable_introspection=True,
        )
        self.runtime_cfg = types.SimpleNamespace(
            target_name=target_name,
            profile_name=profile_name,
            project_name=project_name,
            project_root=project_dir,
        )
        self.manifest = types.SimpleNamespace(
            nodes={
                f"model.{project_name}.{target_name}": types.SimpleNamespace(
                    resource_type="model",
                    package_name=project_name,
                    name=f"{target_name}_model",
                ),
                f"model.other.{target_name}": types.SimpleNamespace(
                    resource_type="model",
                    package_name="other",
                    name="other_model",
                ),
            }
        )
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


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


def test_workbench_change_target_rebuilds_context_and_closes_old_context(monkeypatch) -> None:
    app = _import_workbench_app_with_stubs(monkeypatch)
    old_ctx = _FakeDbtContext("dev")
    new_ctx = _FakeDbtContext("prod")
    created_configs = []

    def create_context(config):
        created_configs.append(config)
        return new_ctx

    app.state.app = types.SimpleNamespace(
        ctx=old_ctx,
        target_name="dev",
        query="select 1",
        compiled_query="old compiled",
        model_nodes=[],
    )
    app.state.target_name = "prod"
    monkeypatch.setattr(app, "create_dbt_project_context", create_context)
    monkeypatch.setattr(
        app,
        "compile_sql_code",
        lambda ctx, sql: _CompiledNode(f"{ctx.runtime_cfg.target_name}: {sql}"),
    )
    monkeypatch.setattr(app, "set_invocation_context", lambda env: None)
    monkeypatch.setattr(app, "get_env", lambda: {})

    app.change_target()

    assert old_ctx.runtime_cfg.target_name == "dev"
    assert app.state.app.ctx is new_ctx
    assert app.state.app.target_name == "prod"
    assert app.state.app.compiled_query == "prod: select 1"
    assert old_ctx.close_calls == 1
    assert [node.name for node in app.state.app.model_nodes] == ["prod_model"]
    assert len(created_configs) == 1
    config = created_configs[0]
    assert config.project_dir == "/project"
    assert config.profiles_dir == "/profiles"
    assert config.profile == "default"
    assert config.target == "prod"
    assert config.threads == 4
    assert config.vars == {"feature": "enabled"}
    assert config.quiet is False
    assert config.disable_introspection is True


def test_workbench_change_target_failure_keeps_old_context_and_reports_error(monkeypatch) -> None:
    app = _import_workbench_app_with_stubs(monkeypatch)
    old_ctx = _FakeDbtContext("dev")
    errors = []
    app.st.error = errors.append
    app.state.app = types.SimpleNamespace(
        ctx=old_ctx,
        target_name="dev",
        query="select 1",
        compiled_query="old compiled",
        model_nodes=[],
    )
    app.state.target_name = "prod"
    monkeypatch.setattr(
        app,
        "create_dbt_project_context",
        lambda config: (_ for _ in ()).throw(RuntimeError("bad target")),
    )
    monkeypatch.setattr(app, "set_invocation_context", lambda env: None)
    monkeypatch.setattr(app, "get_env", lambda: {})

    app.change_target()

    assert app.state.app.ctx is old_ctx
    assert app.state.app.target_name == "dev"
    assert app.state.target_name == "dev"
    assert app.state.app.compiled_query == "old compiled"
    assert old_ctx.close_calls == 0
    assert errors == ["Failed to change dbt target to 'prod': bad target"]
