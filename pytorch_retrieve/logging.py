"""
pytorch_retrieve.logging
"""
from __future__ import annotations

from typing import Any
from rich.progress import Progress


class MaybeProgress:
    """
    Drop-in wrapper for rich.progress.Progress with optional display.

    When show=False, all Progress methods become no-ops but the API remains valid.
    """

    def __init__(
            self,
            *args,
            show: bool = True,
            **kwargs
    ):
        """
        Args:
            *args: Forwarded to Progress class.
            show: Whether or not to show the progress rab.
            **kwargs: Forwarded to Progress class.
        """
        self._show = show
        self._progress = Progress(*args, **kwargs) if show else None

    def __enter__(self):
        if self._show:
            self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._show:
            return self._progress.__exit__(exc_type, exc, tb)
        return False


    def add_task(self, *args, **kwargs):
        if self._show:
            return self._progress.add_task(*args, **kwargs)
        return 0  # dummy task id

    def advance(self, *args, **kwargs):
        if self._show:
            return self._progress.advance(*args, **kwargs)

    def update(self, *args, **kwargs):
        if self._show:
            return self._progress.update(*args, **kwargs)

    def remove_task(self, *args, **kwargs):
        if self._show:
            return self._progress.remove_task(*args, **kwargs)

    def stop(self):
        if self._show:
            return self._progress.stop()

    def __getattr__(self, name: str) -> Any:
        if not self._show:
            raise AttributeError(name)
        return getattr(self._progress, name)
