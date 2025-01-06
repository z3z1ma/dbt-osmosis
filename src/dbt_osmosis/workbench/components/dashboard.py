# pyright: reportAny=false
from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from uuid import uuid4

from streamlit import session_state as state
from streamlit_elements_fluence import dashboard, mui  # pyright: ignore[reportMissingTypeStubs]


class ImplementsBool(t.Protocol):
    def __bool__(self) -> bool: ...


class DarkMode:
    def __bool__(self) -> bool:
        return state.app.theme == "dark"


@t.final
class Dashboard:
    DRAGGABLE_CLASS = "draggable"

    def __init__(self) -> None:
        self._layout: list[Dashboard.Item] = []

    def register(self, item: Dashboard.Item) -> None:
        self._layout.append(item)

    @contextmanager
    def __call__(self, **props: t.Any) -> Generator[None, None, None]:
        props["draggableHandle"] = f".{Dashboard.DRAGGABLE_CLASS}"

        with dashboard.Grid(self._layout, **props):  # pyright: ignore[reportUnknownMemberType]
            yield

    class Item(ABC):
        @staticmethod
        def initial_state() -> dict[str, t.Any]:
            return {}

        def __init__(
            self, board: Dashboard, x: int, y: int, w: int, h: int, **item_props: t.Any
        ) -> None:
            self._key: str = str(uuid4())
            self._draggable_class: str = Dashboard.DRAGGABLE_CLASS
            self._dark_mode: ImplementsBool | bool = DarkMode()
            board.register(dashboard.Item(self._key, x, y, w, h, **item_props))  # pyright: ignore[reportUnknownMemberType,reportArgumentType]

        def _switch_theme(self):
            self._dark_mode = not self._dark_mode

        @contextmanager
        def title_bar(self, padding: str = "5px 15px 5px 15px", dark_switcher: bool = True):
            with mui.Stack(
                className=self._draggable_class,
                alignItems="center",
                direction="row",
                spacing=1,
                sx={
                    "padding": padding,
                    "borderBottom": 1,
                    "borderColor": "divider",
                },
            ):
                yield

                if dark_switcher:
                    if self._dark_mode:
                        _ = mui.IconButton(mui.icon.DarkMode, onClick=self._switch_theme)
                    else:
                        _ = mui.IconButton(
                            mui.icon.LightMode, sx={"color": "#ffc107"}, onClick=self._switch_theme
                        )

        @abstractmethod
        def __call__(self) -> None:
            """Show elements."""
            raise NotImplementedError
