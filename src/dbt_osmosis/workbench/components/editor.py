# pyright: reportAny=false, reportImplicitOverride=false
import typing as t
from enum import Enum
from functools import partial

from streamlit import session_state as state
from streamlit_elements_fluence import (  # pyright: ignore[reportMissingTypeStubs]
    editor,
    lazy,  # pyright: ignore[reportUnknownVariableType]
    mui,
    sync,
)

from .dashboard import Dashboard


class TabName(str, Enum):
    SQL = "SQL"
    YAML = "YAML"


@t.final
class Editor(Dashboard.Item):
    @staticmethod
    def initial_state() -> dict[str, t.Any]:
        return {"query": "", "theme": "dark", "lang": "sql"}

    def __init__(
        self,
        *args: t.Any,
        compile_action: t.Callable[[str], str],
        **kwargs: t.Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tabs = {
            TabName.SQL: {"content": "", "language": "sql"},
            TabName.YAML: {"content": "version: 2\n", "language": "yaml"},
        }
        self._compile_action = compile_action
        self._index = 0
        self._editor_box_style = {
            "flex": 1,
            "minHeight": 0,
            "borderBottom": 1,
            "borderTop": 1,
            "borderColor": "divider",
        }
        self._realtime_compilation = False

    def _change_tab(self, event: t.Any, index: int) -> None:
        print(event)
        self._index = index

    def update_content(self, tab_name: TabName, content: str) -> None:
        self.tabs[tab_name]["content"] = content
        if tab_name == TabName.SQL:
            state.app.compiled_query = self._compile_action(content)

    def get_content(self, tab_name: TabName) -> str:
        return self.tabs[tab_name]["content"]

    def __call__(self, **props: t.Any) -> None:
        with mui.Paper(
            key=self._key,
            sx={
                "display": "flex",
                "flexDirection": "column",
                "borderRadius": 3,
                "overflow": "hidden",
            },
            elevation=1,
        ):
            with self.title_bar("0px 15px 0px 15px"):
                _ = mui.icon.Terminal()
                _ = mui.Typography("dbt Workbench")

                with mui.Tabs(
                    value=self._index,
                    onChange=self._change_tab,
                    scrollButtons=True,
                    variant="scrollable",
                    sx={"flex": 1},
                ):
                    for tab_name in self.tabs.keys():
                        _ = mui.Tab(label=tab_name)

            for index, (tab_name, tab) in enumerate(self.tabs.items()):
                update_tab_content = partial(self.update_content, tab_name)
                with mui.Box(sx=self._editor_box_style, hidden=(index != self._index)):
                    if self._realtime_compilation:
                        on_change = update_tab_content
                    else:
                        on_change = lazy(update_tab_content)
                    if state.app.model and state.app.model != "SCRATCH":
                        path = state.app.model.original_file_path
                    else:
                        path = tab_name
                    _ = editor.Monaco(  # pyright: ignore[reportUnknownMemberType]
                        css={"padding": "0 2px 0 2px"},
                        defaultValue=tab["content"],
                        language=tab["language"],
                        onChange=on_change,
                        theme="vs-dark" if self._dark_mode else "light",
                        path=path,
                        options=props,
                        key="editor-{}".format(index),
                    )

            with mui.Stack(direction="row", spacing=2, alignItems="center", sx={"padding": "10px"}):
                compile_keybind = "Ctrl+Enter"
                _ = mui.Button("Compile", variant="contained", onClick=sync())
                _ = mui.Typography(f"Or press {compile_keybind}", sx={"flex": 1})
                _ = mui.Switch(
                    checked=self._realtime_compilation,
                    onChange=lambda _, checked: setattr(self, "_realtime_compilation", checked),  # pyright: ignore[reportUnknownLambdaType]
                    sx={"marginRight": 1},
                )
                _ = mui.Typography("Realtime Compilation")
