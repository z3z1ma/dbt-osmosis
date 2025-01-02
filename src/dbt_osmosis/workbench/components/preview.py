from streamlit import session_state as state
from streamlit_elements_fluence import JSCallback, mui

from .dashboard import Dashboard


class Preview(Dashboard.Item):
    def __call__(self, query_run_fn):
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
                mui.icon.ViewCompact()
                mui.Typography("Query Preview")
                if state.w.sql_query_state == "success":
                    mui.Typography(
                        "Adapter Response: {}".format(state.w.sql_adapter_resp),
                        sx={"marginRight": "auto", "color": "text.secondary"},
                    )

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                if state.w.sql_query_state == "running":
                    mui.CircularProgress(sx={"padding": "25px"})
                elif state.w.sql_query_state == "error":
                    mui.Typography(
                        "Error running query\n\n{}".format(state.w.sql_adapter_resp),
                        sx={"padding": "25px"},
                    )
                elif not state.w.sql_result_columns:
                    mui.Typography("No results to show...", sx={"padding": "25px"})
                else:
                    mui.DataGrid(
                        columns=state.w.sql_result_columns,
                        rows=state.w.sql_result_rows,
                        pageSize=20,
                        rowsPerPageOptions=[20, 50, 100],
                        checkboxSelection=False,
                        disableSelectionOnClick=True,
                        getRowId=JSCallback("(row) => Math.random()"),
                    )

            with mui.Stack(direction="row", spacing=2, alignItems="center", sx={"padding": "10px"}):
                mui.Button(
                    "Run Query",
                    variant="contained",
                    onClick=lambda: query_run_fn(),
                )
                key = "Ctrl+Shift+Enter"
                mui.Typography(f"Or press {key}", sx={"flex": 1})
