"""AI Documentation Assistant component for dbt-osmosis workbench.

This component provides an interactive interface for the AI documentation co-pilot,
allowing users to generate, review, and apply AI-powered documentation suggestions.
"""

from __future__ import annotations

import time
import typing as t

import streamlit as st
from streamlit_elements import elements, mui

if t.TYPE_CHECKING:
    from dbt_osmosis.workbench.components.dashboard import Dashboard

__all__ = ["AIAssistant"]


@t.final
class AIAssistant(Dashboard.Item):
    """AI Documentation Assistant component for the workbench.

    Provides interactive AI-powered documentation generation with:
    - Model-level documentation suggestions
    - Column-level documentation suggestions
    - Confidence scoring for each suggestion
    - Accept/reject workflow
    - Voice learning from project patterns
    """

    @staticmethod
    def initial_state() -> dict[str, t.Any]:
        """Initialize the AI assistant state."""
        return {
            "suggestions": [],
            "selected_suggestion": None,
            "is_generating": False,
            "target_model": None,
            "target_column": None,
            "confidence_threshold": 0.7,
            "learning_mode": True,
            "last_error": None,
            "stats": {
                "generated": 0,
                "accepted": 0,
                "rejected": 0,
            },
        }

    def __call__(self, **props: t.Any) -> None:
        """Render the AI assistant component.

        Args:
            **props: Component properties including action callbacks
        """
        # Get state
        if not hasattr(st.session_state, "ai_assistant"):
            st.session_state.ai_assistant = self.initial_state()

        ai_state = st.session_state.ai_assistant

        # Process generation if flag is set
        if ai_state["is_generating"]:
            self._process_generation(ai_state)

        with elements("ai_assistant"):
            with mui.Paper(
                elevation=1,
                sx={
                    "padding": "10px 15px 10px 15px",
                    "borderRadius": 3,
                    "height": "100%",
                    "display": "flex",
                    "flexDirection": "column",
                },
            ):
                # Header
                with mui.Box(
                    sx={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "marginBottom": "10px",
                    }
                ):
                    with mui.Box(sx={"display": "flex", "alignItems": "center", "gap": "8px"}):
                        mui.icon.PsychologyOutlined()
                        mui.Typography("AI Documentation Assistant", variant="h6")

                    # Learning mode toggle
                    with mui.Tooltip(title="Enable voice learning from project patterns"):
                        mui.Switch(
                            checked=ai_state["learning_mode"],
                            onChange=self._toggle_learning_mode,
                            label="Learning",
                        )

                # Content area
                with mui.Box(sx={"flexGrow": 1, "overflowY": "auto"}):
                    if not ai_state["is_generating"] and not ai_state["suggestions"]:
                        self._render_empty_state(ai_state)
                    elif ai_state["is_generating"]:
                        self._render_loading_state()
                    else:
                        self._render_suggestions(ai_state)

                # Footer with controls
                self._render_footer(ai_state)

    def _render_empty_state(self, ai_state: dict[str, t.Any]) -> None:
        """Render the empty state when no suggestions are available.

        Args:
            ai_state: The AI assistant state
        """
        has_model = self._is_model_available()

        with mui.Box(
            sx={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "height": "100%",
                "gap": "16px",
                "padding": "20px",
            }
        ):
            mui.icon.AutoAwesomeOutlined(sx={"fontSize": 64, "color": "text.secondary"})
            mui.Typography("Generate AI-powered documentation", variant="body1")

            if has_model:
                mui.Typography(
                    f"Model: {st.session_state.app.model.name}",
                    variant="body2",
                    sx={"color": "text.secondary"},
                )
            else:
                mui.Typography(
                    "Select a model to get started",
                    variant="body2",
                    sx={"color": "text.secondary"},
                )

            # Quick action buttons
            with mui.Box(sx={"display": "flex", "gap": "8px", "flexDirection": "column"}):
                mui.Button(
                    mui.icon.DocumentText(),
                    "Document Model",
                    variant="contained",
                    onClick=self._start_model_generation,
                    color="primary",
                    disabled=not has_model,
                    sx={"width": "100%"},
                )
                mui.Button(
                    mui.icon.ViewColumn(),
                    "Document Columns",
                    variant="outlined",
                    onClick=self._start_column_generation,
                    color="primary",
                    disabled=not has_model,
                    sx={"width": "100%"},
                )

    def _render_loading_state(self) -> None:
        """Render the loading state while generating suggestions."""
        with mui.Box(
            sx={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "height": "100%",
                "gap": "16px",
                "padding": "20px",
            }
        ):
            mui.CircularProgress()
            mui.Typography("Generating suggestions...", variant="body2")

    def _render_suggestions(self, ai_state: dict[str, t.Any]) -> None:
        """Render the list of suggestions.

        Args:
            ai_state: The AI assistant state
        """
        for i, suggestion in enumerate(ai_state["suggestions"]):
            with mui.Accordion(
                key=f"suggestion-{i}",
                defaultExpanded=(i == len(ai_state["suggestions"]) - 1),
            ):
                with mui.AccordionSummary(
                    expandIcon=mui.icon.ExpandMore(),
                    sx={
                        "backgroundColor": "rgba(0,0,0,0.05)"
                        if self._dark_mode
                        else "rgba(0,0,0,0.03)"
                    },
                ):
                    with mui.Box(
                        sx={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "width": "100%",
                        }
                    ):
                        # Confidence indicator
                        confidence = suggestion.get("confidence", 0.0)
                        confidence_color = self._get_confidence_color(confidence)
                        mui.CircularProgress(
                            variant="determinate",
                            value=confidence * 100,
                            size=24,
                            sx={"color": confidence_color},
                        )

                        # Suggestion title
                        target = suggestion.get("target", "Unknown")
                        mui.Typography(
                            f"{target}",
                            variant="subtitle2",
                        )

                with mui.AccordionDetails:
                    # Suggestion content
                    mui.Typography(suggestion.get("text", ""), variant="body2")

                    # Metadata
                    with mui.Box(
                        sx={"marginTop": "8px", "display": "flex", "gap": "4px", "flexWrap": "wrap"}
                    ):
                        mui.Chip(
                            label=f"Source: {suggestion.get('source', 'llm')}",
                            size="small",
                            variant="outlined",
                        )
                        mui.Chip(
                            label=f"Reason: {suggestion.get('reason', '')[:30]}...",
                            size="small",
                            variant="outlined",
                        )

                    # Action buttons
                    with mui.Box(sx={"marginTop": "12px", "display": "flex", "gap": "8px"}):
                        mui.Button(
                            mui.icon.Check(),
                            "Accept",
                            size="small",
                            variant="contained",
                            color="success",
                            onClick=lambda s=suggestion, idx=i: self._accept_suggestion(s, idx),
                        )
                        mui.Button(
                            mui.icon.Close(),
                            "Reject",
                            size="small",
                            variant="outlined",
                            color="error",
                            onClick=lambda idx=i: self._reject_suggestion(idx),
                        )

    def _render_footer(self, ai_state: dict[str, t.Any]) -> None:
        """Render the footer with controls and stats.

        Args:
            ai_state: The AI assistant state
        """
        stats = ai_state["stats"]

        with mui.Box(
            sx={
                "marginTop": "10px",
                "paddingTop": "10px",
                "borderTop": "1px solid rgba(0,0,0,0.12)"
                if self._dark_mode
                else "1px solid rgba(0,0,0,0.06)",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
            }
        ):
            # Statistics
            stats_text = (
                f"Generated: {stats['generated']} | "
                f"Accepted: {stats['accepted']} | "
                f"Rejected: {stats['rejected']}"
            )
            mui.Typography(
                stats_text,
                variant="caption",
                sx={"color": "text.secondary"},
            )

            # Clear button
            if ai_state["suggestions"]:
                mui.Button(
                    "Clear All",
                    size="small",
                    variant="text",
                    onClick=self._clear_suggestions,
                )

    def _start_model_generation(self) -> None:
        """Start generating a documentation suggestion for the current model."""
        st.session_state.ai_assistant["is_generating"] = True
        st.session_state.ai_assistant["generation_type"] = "model"
        st.session_state.ai_assistant["last_error"] = None
        st.rerun()

    def _start_column_generation(self) -> None:
        """Start generating documentation suggestions for all columns."""
        st.session_state.ai_assistant["is_generating"] = True
        st.session_state.ai_assistant["generation_type"] = "columns"
        st.session_state.ai_assistant["last_error"] = None
        st.rerun()

    def _process_generation(self, ai_state: dict[str, t.Any]) -> None:
        """Process the AI suggestion generation.

        Args:
            ai_state: The AI assistant state
        """
        try:
            # Simulate API delay
            time.sleep(0.5)

            model = st.session_state.app.model
            if model == "SCRATCH":
                ai_state["is_generating"] = False
                ai_state["last_error"] = "Cannot document SCRATCH model"
                st.rerun()
                return

            generation_type = ai_state.get("generation_type", "model")

            # Generate mock suggestions for now
            # In production, this would call the actual LLM APIs
            if generation_type == "model":
                suggestion = {
                    "target": f"Model: {model.name}",
                    "text": (
                    f"AI-generated description for {model.name}. "
                    f"This model contains transformed data from upstream sources."
                ),
                    "confidence": 0.85,
                    "reason": f"Improving documentation for {model.name}",
                    "source": "ai-co-pilot",
                }
                ai_state["suggestions"] = [suggestion]
            else:  # columns
                suggestions = []
                for col_name in list(model.columns.keys())[:5]:  # Limit to 5 columns
                    suggestions.append({
                        "target": f"Column: {col_name}",
                        "text": f"AI-generated description for {col_name} column",
                        "confidence": 0.75,
                        "reason": f"Generating documentation for {col_name}",
                        "source": "ai-co-pilot",
                    })
                ai_state["suggestions"] = suggestions

            ai_state["stats"]["generated"] += len(ai_state["suggestions"])
            ai_state["is_generating"] = False

        except Exception as e:
            ai_state["is_generating"] = False
            ai_state["last_error"] = str(e)

        st.rerun()

    def _accept_suggestion(self, suggestion: dict[str, t.Any], index: int) -> None:
        """Accept a suggestion and apply it.

        Args:
            suggestion: The suggestion to accept
            index: The index of the suggestion in the list
        """
        # Update editor or manifest with the suggestion
        # This will be implemented with actual dbt context integration
        ai_state = st.session_state.ai_assistant
        ai_state["stats"]["accepted"] += 1
        ai_state["suggestions"].pop(index)
        st.rerun()

    def _reject_suggestion(self, index: int) -> None:
        """Reject a suggestion.

        Args:
            index: The index of the suggestion to reject
        """
        ai_state = st.session_state.ai_assistant
        ai_state["stats"]["rejected"] += 1
        ai_state["suggestions"].pop(index)
        st.rerun()

    def _clear_suggestions(self) -> None:
        """Clear all suggestions."""
        st.session_state.ai_assistant["suggestions"] = []
        st.rerun()

    def _toggle_learning_mode(self, event: t.Any, checked: bool) -> None:
        """Toggle learning mode on/off.

        Args:
            event: The toggle event
            checked: The new checked state
        """
        st.session_state.ai_assistant["learning_mode"] = checked
        st.rerun()

    def _is_model_available(self) -> bool:
        """Check if a model is available for documentation.

        Returns:
            True if a model is selected, False otherwise
        """
        return (
            hasattr(st.session_state, "app")
            and hasattr(st.session_state.app, "model")
            and st.session_state.app.model is not None
            and st.session_state.app.model != "SCRATCH"
        )

    def _get_confidence_color(self, confidence: float) -> str:
        """Get the color for a confidence level.

        Args:
            confidence: Confidence score from 0.0 to 1.0

        Returns:
            CSS color string
        """
        if confidence >= 0.8:
            return "#4caf50"  # Green
        elif confidence >= 0.6:
            return "#ff9800"  # Orange
        else:
            return "#f44336"  # Red

    @property
    def _dark_mode(self) -> bool:
        """Check if dark mode is enabled.

        Returns:
            True if dark mode is enabled, False otherwise
        """
        # This would be integrated with the main app theme
        return False
