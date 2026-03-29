"""Truthful AI assistant panel for the dbt-osmosis workbench.

The workbench currently does not have a real suggestion-generation or writeback
pipeline. This component therefore renders an honest status panel instead of
pretending to generate or apply documentation changes.
"""

from __future__ import annotations

import typing as t

from streamlit import session_state as state
from streamlit_elements_fluence import mui  # pyright: ignore[reportMissingTypeStubs]

from .dashboard import Dashboard

__all__ = ["AIAssistant"]

_UNAVAILABLE_TITLE = "No live workbench AI suggestions yet"
_UNAVAILABLE_MESSAGE = (
    "The workbench AI panel is not wired into the real documentation pipeline, "
    "so it would be misleading to generate or accept suggestions here."
)
_UNAVAILABLE_NEXT_STEP = (
    "Use the CLI documentation commands for real changes until the workbench "
    "path can surface and apply truthful results."
)


@t.final
class AIAssistant(Dashboard.Item):
    """Render the current AI-assistant status without faking unsupported behavior."""

    @staticmethod
    def initial_state() -> dict[str, t.Any]:
        return {"ai_assistant_status": "unavailable"}

    def __call__(self, **props: t.Any) -> None:
        del props

        model_name = self._selected_model_name()

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
            with self.title_bar("10px 15px 10px 15px", dark_switcher=False):
                _ = mui.icon.PsychologyOutlined()
                _ = mui.Typography("AI Documentation Assistant")
                _ = mui.Chip(
                    label="Unavailable",
                    size="small",
                    variant="outlined",
                    color="warning",
                    sx={"marginLeft": "auto"},
                )

            with mui.Box(
                sx={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "12px",
                    "padding": "16px",
                    "height": "100%",
                    "justifyContent": "center",
                }
            ):
                _ = mui.Typography(_UNAVAILABLE_TITLE, variant="subtitle1")
                _ = mui.Typography(
                    self._selected_model_message(model_name),
                    variant="body2",
                    sx={"color": "text.secondary"},
                )
                _ = mui.Typography(_UNAVAILABLE_MESSAGE, variant="body2")
                _ = mui.Typography(
                    _UNAVAILABLE_NEXT_STEP,
                    variant="body2",
                    sx={"color": "text.secondary"},
                )

    @staticmethod
    def _selected_model_name() -> str | None:
        app_state = getattr(state, "app", None)
        model = getattr(app_state, "model", None)
        if model is None or model == "SCRATCH":
            return None

        model_name = getattr(model, "name", None)
        if isinstance(model_name, str) and model_name:
            return model_name
        return None

    @staticmethod
    def _selected_model_message(model_name: str | None) -> str:
        if model_name is None:
            return "Select a dbt model to inspect future workbench AI support."
        return f"Selected model: {model_name}"
