# pyright: reportAny=false, reportImplicitOverride=false
import typing as t
from hashlib import md5

from streamlit import session_state as state
from streamlit_elements_fluence import editor, mui  # pyright: ignore[reportMissingTypeStubs]

from .dashboard import Dashboard


@t.final
class Renderer(Dashboard.Item):
    @staticmethod
    def initial_state() -> dict[str, t.Any]:
        return {"compiled_query": ""}

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self._editor_box_style = {
            "flex": 1,
            "minHeight": 0,
            "borderBottom": 1,
            "borderTop": 1,
            "borderColor": "divider",
        }

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
            with self.title_bar("12px 15px 12px 15px", dark_switcher=False):
                _ = mui.icon.Preview()
                _ = mui.Typography("Compiled SQL")

            with mui.Box(sx=self._editor_box_style):
                _ = editor.Monaco(  # pyright: ignore[reportUnknownMemberType]
                    css={"padding": "0 2px 0 2px"},
                    defaultValue=state.app.compiled_query or "",
                    language="sql",
                    theme="vs-dark" if self._dark_mode else "light",
                    options={**props, "readOnly": True},
                    key=(
                        md5(
                            state.app.compiled_query.encode("utf-8")
                            + state.app.target_name.encode("utf-8")
                        ).hexdigest()
                        if state.app.compiled_query
                        else "__empty__"
                    ),
                )
