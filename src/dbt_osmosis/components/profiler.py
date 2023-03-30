from streamlit import session_state as state
from streamlit_elements_fluence import extras, html, mui

from .dashboard import Dashboard


class Profiler(Dashboard.Item):
    def __call__(self, run_profile_fn):
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
                mui.icon.Psychology()
                mui.Typography("Pandas Profiler")
                if state.w.sql_result_df.empty:
                    mui.Typography(
                        "No data to profile, execute a query first", sx={"color": "text.secondary"}
                    )

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                if state.w.profile_html:
                    html.Iframe(
                        srcDoc=state.w.profile_html,
                        style={"width": "100%", "height": "100%", "border": "none"},
                    )
                else:
                    # extras.InnerHTML("<a href='https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/' target='_blank'>Docs</a>")
                    html.Iframe(
                        src="https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/",
                        style={"width": "100%", "height": "100%", "border": "none"},
                    )

            with mui.Stack(direction="row", spacing=2, alignItems="center", sx={"padding": "10px"}):
                mui.Button(
                    "Profile Results",
                    variant="contained",
                    onClick=lambda: run_profile_fn(),
                )
