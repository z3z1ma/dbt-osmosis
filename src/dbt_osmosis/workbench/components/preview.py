# pyright: reportAny=false, reportMissingTypeStubs=false, reportImplicitOverride=false
import typing as t

from streamlit import session_state as state
from streamlit_elements_fluence import JSCallback, mui

from .dashboard import Dashboard


@t.final
class Preview(Dashboard.Item):
    @staticmethod
    def initial_state() -> dict[str, t.Any]:
        import pandas as pd

        return {
            "query_adapter_resp": None,
            "query_result_df": pd.DataFrame(),
            "query_result_columns": [],
            "query_result_rows": [],
            "query_state": "test",
            "query_template": "select * from ({sql}) as _query limit 200",
        }

    def __init__(self, *args: t.Any, query_action: t.Callable[[], None], **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self._query_action = query_action

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
                _ = mui.icon.ViewCompact()
                _ = mui.Typography("Query Preview")
                if state.app.query_state == "success":
                    _ = mui.Typography(
                        "Adapter Response: {}".format(state.app.query_adapter_resp),
                        sx={"marginRight": "auto", "color": "text.secondary"},
                    )

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                if state.app.query_state == "running":
                    _ = mui.CircularProgress(sx={"padding": "25px"})
                elif state.app.query_state == "error":
                    _ = mui.Typography(
                        "Error running query\n\n{}".format(state.app.query_adapter_resp),
                        sx={"padding": "25px"},
                    )
                elif not state.app.query_result_columns:
                    _ = mui.Typography("No results to show...", sx={"padding": "25px"})
                else:
                    _ = mui.DataGrid(
                        columns=state.app.query_result_columns,
                        rows=state.app.query_result_rows,
                        pageSize=20,
                        rowsPerPageOptions=[20, 50, 100],
                        checkboxSelection=False,
                        disableSelectionOnClick=True,
                        getRowId=JSCallback("(row) => Math.random()"),
                    )

            with mui.Stack(direction="row", spacing=2, alignItems="center", sx={"padding": "10px"}):
                run_keybind = "Ctrl+Shift+Enter"
                _ = mui.Button(
                    "Run Query",
                    variant="contained",
                    onClick=lambda: self._query_action(),
                )
                _ = mui.Typography(f"Or press {run_keybind}", sx={"flex": 1})
