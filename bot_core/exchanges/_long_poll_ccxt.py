"""Wspólny miksin dostarczający obsługę long-polla dla adapterów CCXT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, MutableMapping

from bot_core.exchanges.streaming import LocalLongPollStream


class CCXTLongPollMixin:
    """Mieszanka konfigurująca :class:`LocalLongPollStream` na bazie ustawień."""

    _STREAM_FALLBACK: Mapping[str, Any] = {
        "poll_interval": 0.5,
        "timeout": 10.0,
        "max_retries": 3,
        "backoff_base": 0.25,
        "backoff_cap": 2.0,
        "jitter": (0.05, 0.30),
        "channel_param": "channels",
        "cursor_param": "cursor",
    }

    def _stream_defaults(self) -> Mapping[str, Any]:  # pragma: no cover - nadpisywane w podklasach
        return {}

    def _resolve_stream_settings(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = dict(self._STREAM_FALLBACK)
        defaults = self._stream_defaults()
        for key, value in defaults.items():
            if isinstance(value, Mapping):
                payload[key] = dict(value)
            else:
                payload[key] = value
        configured = getattr(self, "_settings", {}).get("stream")  # type: ignore[attr-defined]
        if isinstance(configured, Mapping):
            for key, value in configured.items():
                if isinstance(value, Mapping) and isinstance(payload.get(key), Mapping):
                    merged = dict(payload[key])  # type: ignore[index]
                    merged.update(value)
                    payload[key] = merged
                else:
                    payload[key] = value
        return payload

    def _build_long_poll_stream(
        self,
        scope: str,
        channels: Sequence[str],
    ) -> LocalLongPollStream:
        settings = self._resolve_stream_settings()
        base_url = str(settings.get("base_url", settings.get("url", "http://127.0.0.1:8765")))
        base_url = base_url.rstrip("/") or "http://127.0.0.1:8765"

        default_path = settings.get("path") or f"/stream/{self.name}/{scope}"
        path = settings.get(f"{scope}_path", default_path) or default_path
        path = str(path)
        if not path.startswith("/"):
            path = f"/{path}"

        def _pick(name: str, fallback: Any = None) -> Any:
            scoped_key = f"{scope}_{name}"
            if scoped_key in settings:
                return settings[scoped_key]
            return settings.get(name, fallback)

        poll_interval = float(_pick("poll_interval"))
        timeout = float(_pick("timeout"))
        max_retries = int(_pick("max_retries"))
        backoff_base = float(_pick("backoff_base"))
        backoff_cap = float(_pick("backoff_cap"))
        jitter = _pick("jitter") or (0.05, 0.30)

        channel_param = _pick("channel_param")
        cursor_param = _pick("cursor_param")
        initial_cursor = _pick("initial_cursor")

        params: MutableMapping[str, Any] = {}
        raw_params = _pick("params")
        if isinstance(raw_params, Mapping):
            params.update(raw_params)
        raw_scope_params = settings.get(f"{scope}_params")
        if isinstance(raw_scope_params, Mapping):
            params.update(raw_scope_params)

        headers = settings.get("headers")
        header_map = dict(headers) if isinstance(headers, Mapping) else None

        params_in_body = bool(_pick("params_in_body", False))
        channels_in_body = bool(_pick("channels_in_body", False))
        cursor_in_body = bool(_pick("cursor_in_body", False))
        body_params = _pick("body_params")
        body_encoder = _pick("body_encoder")

        return LocalLongPollStream(
            base_url=base_url,
            path=path,
            channels=channels,
            adapter=self.name,  # type: ignore[attr-defined]
            scope=scope,
            environment=self._environment.value,  # type: ignore[attr-defined]
            params=params or None,
            headers=header_map,
            poll_interval=poll_interval,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            jitter=jitter,
            channel_param=channel_param,
            cursor_param=cursor_param,
            initial_cursor=str(initial_cursor) if initial_cursor not in (None, "") else None,
            params_in_body=params_in_body,
            channels_in_body=channels_in_body,
            cursor_in_body=cursor_in_body,
            body_params=body_params if isinstance(body_params, Mapping) else None,
            body_encoder=body_encoder,
            metrics_registry=self._metrics,  # type: ignore[attr-defined]
        )


__all__ = ["CCXTLongPollMixin"]

