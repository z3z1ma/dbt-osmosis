# pyright: reportAny=false, reportImplicitOverride=false
import typing as t

from streamlit import session_state as state
from streamlit_elements_fluence import extras, mui  # pyright: ignore[reportMissingTypeStubs]

from .dashboard import Dashboard


@t.final
class RssFeed(Dashboard.Item):
    @staticmethod
    def initial_state() -> dict[str, t.Any]:
        return {"feed_html": ""}

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
                "overflow": "scroll",
            },
            elevation=1,
        ):
            with self.title_bar("12px 15px 12px 15px", dark_switcher=False):
                _ = mui.icon.Feed()
                _ = mui.Typography("Hacker News Feed")

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                extras.InnerHTML(state.app.feed_html, **props)  # pyright: ignore[reportUnknownMemberType]
