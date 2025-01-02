from streamlit import session_state as state
from streamlit_elements_fluence import extras, mui

from .dashboard import Dashboard


class RssFeed(Dashboard.Item):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dark_theme = False
        self._editor_box_style = {
            "flex": 1,
            "minHeight": 0,
            "borderBottom": 1,
            "borderTop": 1,
            "borderColor": "divider",
        }

    def __call__(self, **options):
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
                mui.icon.Feed()
                mui.Typography("Hacker News Feed")

            # Render hacker news feed
            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                extras.InnerHTML(state.w.feed_contents)
