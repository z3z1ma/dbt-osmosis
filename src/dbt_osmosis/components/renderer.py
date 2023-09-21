from hashlib import md5
from uuid import uuid4

from streamlit import session_state as state
from streamlit_elements_fluence import dashboard, editor, mui

from .dashboard import Dashboard


class DarkMode:
    def __bool__(self):
        return state.w.theme == "dark"


class Renderer(Dashboard.Item):
    def __init__(self, board, x, y, w, h, **item_props):
        self._key = str(uuid4())
        self._draggable_class = Dashboard.DRAGGABLE_CLASS
        board._register(dashboard.Item(self._key, x, y, w, h, **item_props))
        self._editor_box_style = {
            "flex": 1,
            "minHeight": 0,
            "borderBottom": 1,
            "borderTop": 1,
            "borderColor": "divider",
        }
        self._dark_mode = DarkMode()

    def __call__(
        self,
        **options,
    ):
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
            with self.title_bar("12px 15px 12px 15px", dark_switcher=False):
                mui.icon.Preview()
                mui.Typography("Compiled SQL")

            with mui.Box(sx=self._editor_box_style):
                editor.Monaco(
                    css={"padding": "0 2px 0 2px"},
                    defaultValue=state.w.compiled_sql or "",
                    language="sql",
                    theme="vs-dark" if self._dark_mode else "light",
                    options={**options, "readOnly": True},
                    key=(
                        md5(
                            state.w.compiled_sql.encode("utf-8")
                            + state.target_profile.encode("utf-8")
                        ).hexdigest()
                        if state.w.compiled_sql
                        else "__empty__"
                    ),
                )
