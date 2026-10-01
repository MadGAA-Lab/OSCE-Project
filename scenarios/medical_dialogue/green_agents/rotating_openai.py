"""Small OpenAI client facade that fails over between API keys."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from openai import OpenAI


logger = logging.getLogger("medical_judge.api_keys")
ROTATABLE_STATUS_CODES = {401, 403, 429}


def parse_api_keys(*values: str | None) -> list[str]:
    keys: list[str] = []
    for value in values:
        if not value:
            continue
        for candidate in value.replace("\n", ",").split(","):
            key = candidate.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


class _CompletionsFacade:
    def __init__(self, pool: "RotatingOpenAI", beta: bool) -> None:
        self._pool = pool
        self._beta = beta

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._pool._call(
            lambda client: client.chat.completions.create(*args, **kwargs)
        )

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        if not self._beta:
            raise AttributeError("parse is only available through beta.chat.completions")
        return self._pool._call(
            lambda client: client.beta.chat.completions.parse(*args, **kwargs)
        )


class RotatingOpenAI:
    """Expose the OpenAI methods used by this scenario with key failover.

    Authentication/permission failures permanently disable a key for this
    process. Rate limits move to the next key but keep the limited key eligible
    for a later request.
    """

    def __init__(
        self,
        api_keys: Sequence[str],
        *,
        client_factory: Callable[..., Any] = OpenAI,
        **client_kwargs: Any,
    ) -> None:
        keys = parse_api_keys(",".join(api_keys))
        if not keys:
            raise ValueError("At least one API key is required")

        self._clients = [
            client_factory(api_key=key, **client_kwargs)
            for key in keys
        ]
        self._current = 0
        self._disabled: set[int] = set()
        self._lock = threading.Lock()

        self.chat = SimpleNamespace(
            completions=_CompletionsFacade(self, beta=False)
        )
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(
                completions=_CompletionsFacade(self, beta=True)
            )
        )

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status if isinstance(response_status, int) else None

    def _candidate_indices(self) -> list[int]:
        with self._lock:
            return [
                (self._current + step) % len(self._clients)
                for step in range(len(self._clients))
                if (self._current + step) % len(self._clients) not in self._disabled
            ]

    def _call(self, operation: Callable[[Any], Any]) -> Any:
        candidates = self._candidate_indices()
        if not candidates:
            raise RuntimeError("All configured OpenAI API keys are disabled")

        last_error: Exception | None = None
        for position, index in enumerate(candidates):
            try:
                result = operation(self._clients[index])
                with self._lock:
                    self._current = index
                return result
            except Exception as exc:
                status_code = self._status_code(exc)
                if status_code not in ROTATABLE_STATUS_CODES:
                    raise

                last_error = exc
                with self._lock:
                    if status_code in {401, 403}:
                        self._disabled.add(index)
                    self._current = (index + 1) % len(self._clients)

                remaining = len(candidates) - position - 1
                logger.warning(
                    "OpenAI request failed with status %s; rotating API key (%s remaining)",
                    status_code,
                    remaining,
                )

        assert last_error is not None
        raise last_error
