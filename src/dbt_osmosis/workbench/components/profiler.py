# pyright: reportAny=false, reportImplicitOverride=false
import typing as t

from streamlit import session_state as state
from streamlit_elements_fluence import html, mui  # pyright: ignore[reportMissingTypeStubs]

from .dashboard import Dashboard


@t.final
class Profiler(Dashboard.Item):
    @staticmethod
    def initial_state() -> dict[str, t.Any]:
        return {"profile_html": ""}

    def __init__(self, *args: t.Any, prof_action: t.Callable[[], None], **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self._prof_action = prof_action

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
            with self.title_bar(padding="10px 15px 10px 15px", dark_switcher=False):
                _ = mui.icon.Psychology()
                _ = mui.Typography("Pandas Profiler")
                if state.app.query_result_df.empty:
                    _ = mui.Typography(
                        "No data to profile, execute a query first", sx={"color": "text.secondary"}
                    )

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                if state.app.profile_html:
                    _ = html.Iframe(
                        srcDoc=state.app.profile_html,
                        style={"width": "100%", "height": "100%", "border": "none"},
                    )
                else:
                    _ = html.Iframe(
                        src="https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/",
                        style={"width": "100%", "height": "100%", "border": "none"},
                    )

            with mui.Stack(direction="row", spacing=2, alignItems="center", sx={"padding": "10px"}):
                _ = mui.Button(
                    "Profile Results",
                    variant="contained",
                    onClick=lambda: self._prof_action(),
                )
