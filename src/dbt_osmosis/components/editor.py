from enum import Enum
from functools import partial

from streamlit import session_state as state
from streamlit_elements_fluence import editor, lazy, mui, sync

from .dashboard import Dashboard


class Tabs(str, Enum):
    SQL = "SQL"
    YAML = "YAML"


class DarkMode:
    def __bool__(self):
        return state.w.theme == "dark"


class Editor(Dashboard.Item):
    def __init__(self, *args, sql_compiler=lambda content: content, **kwargs):
        super().__init__(*args, **kwargs)
        self._dark_mode = DarkMode()

        self.tabs = {
            Tabs.SQL: {"content": "", "language": "sql"},
            Tabs.YAML: {"content": "version: 2\n", "language": "yaml"},
        }
        self.sql_compiler = sql_compiler

        self._index = 0
        self._editor_box_style = {
            "flex": 1,
            "minHeight": 0,
            "borderBottom": 1,
            "borderTop": 1,
            "borderColor": "divider",
        }
        self._realtime_compilation = False

    def _switch_theme(self):
        if state.w.theme == "dark":
            state.w.theme = "light"
        else:
            state.w.theme = "dark"

    def _change_tab(self, _, index):
        self._index = index

    def update_content(self, label, content):
        self.tabs[label]["content"] = content
        if label == Tabs.SQL:
            state.w.compiled_sql = self.sql_compiler(state.w.ctx, content)

    def get_content(self, label):
        return self.tabs[label]["content"]

    def __call__(self, **options):
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
                mui.icon.Terminal()
                mui.Typography("dbt Workbench")

                with mui.Tabs(
                    value=self._index,
                    onChange=self._change_tab,
                    scrollButtons=True,
                    variant="scrollable",
                    sx={"flex": 1},
                ):
                    for label in self.tabs.keys():
                        mui.Tab(label=label)

            for index, (label, tab) in enumerate(self.tabs.items()):
                with mui.Box(sx=self._editor_box_style, hidden=(index != self._index)):
                    if self._realtime_compilation:
                        onChange = partial(self.update_content, label)
                    else:
                        onChange = lazy(partial(self.update_content, label))
                    if state.w.model and state.w.model != "SCRATCH":
                        path = state.w.model.original_file_path
                    else:
                        path = label
                    editor.Monaco(
                        css={"padding": "0 2px 0 2px"},
                        defaultValue=tab["content"],
                        language=tab["language"],
                        onChange=onChange,
                        theme="vs-dark" if self._dark_mode else "light",
                        path=path,
                        options=options,
                        key=f"editor-{state.w.cache_version}-{index}",
                    )

            with mui.Stack(direction="row", spacing=2, alignItems="center", sx={"padding": "10px"}):
                mui.Button("Compile", variant="contained", onClick=sync())
                key = "Ctrl+Enter"
                mui.Typography(f"Or press {key}", sx={"flex": 1})
                mui.Switch(
                    checked=self._realtime_compilation,
                    onChange=lambda _, checked: setattr(self, "_realtime_compilation", checked),
                    sx={"marginRight": 1},
                )
                mui.Typography("Realtime Compilation")
