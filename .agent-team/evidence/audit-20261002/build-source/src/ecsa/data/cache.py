"""Atomic disk cache and the sole generic HTTP transport boundary."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

from ecsa.data.ratelimit import TokenBucket


class DiskCache:
    """URL/parameter cache; TTL and injected clock use seconds since the UTC epoch."""

    def __init__(
        self, directory: Path, ttl: float = 86400, clock: Callable[[], float] = time.time
    ) -> None:
        self.directory, self.ttl, self.clock = Path(directory), ttl, clock
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(url: str, params: Mapping[str, Any] | None = None) -> str:
        """Return an order-independent SHA-256 of URL and JSON parameters."""
        value = json.dumps(
            [url, params or {}], sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(value.encode()).hexdigest()

    def get_bytes(self, url: str, params: Mapping[str, Any] | None = None) -> bytes | None:
        """Return cached bytes, or None for missing, expired or corrupt entries."""
        try:
            entry = json.loads((self.directory / (self.key(url, params) + ".json")).read_text())
            if self.clock() - entry["created"] >= self.ttl:
                return None
            return base64.b64decode(entry["body"], validate=True)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put_bytes(self, url: str, data: bytes, params: Mapping[str, Any] | None = None) -> None:
        """Atomically cache a byte response without storing credential headers."""
        payload = json.dumps({"created": self.clock(), "body": base64.b64encode(data).decode()})
        fd, name = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.directory / (self.key(url, params) + ".json"))
        finally:
            Path(name).unlink(missing_ok=True)

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        """Return decoded cached JSON or None on a cache miss."""
        value = self.get_bytes(url, params)
        return json.loads(value) if value is not None else None

    def put_json(self, url: str, value: Any, params: Mapping[str, Any] | None = None) -> None:
        """Cache a JSON-serializable value."""
        self.put_bytes(url, json.dumps(value, allow_nan=False).encode(), params)


class HttpClient:
    """Cached HTTP requests with bounded retries and per-request rate limiting."""

    def __init__(
        self,
        cache: DiskCache,
        *,
        rate: float = 8,
        retries: int = 4,
        timeout: float = 30,
        backoff: float = 1,
        headers: Mapping[str, str] | None = None,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache, self.retries, self.timeout, self.backoff = cache, retries, timeout, backoff
        self.headers, self.session, self.sleep = (
            dict(headers or {}),
            session or requests.Session(),
            sleep,
        )
        self.limiter = TokenBucket(rate, sleep=sleep)

    def request(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        body: dict[str, Any] | None = None,
    ) -> bytes:
        """GET or JSON POST; retry 429/5xx and transport failures; never cache failures."""
        key = {"params": dict(params or {}), "body": body}
        cached = self.cache.get_bytes(url, key)
        if cached is not None:
            return cached
        for attempt in range(self.retries + 1):
            self.limiter.acquire()
            delay = self.backoff * 2**attempt
            try:
                response = self.session.request(
                    "POST" if body is not None else "GET",
                    url,
                    params=params,
                    json=body,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                retryable = response.status_code == 429 or 500 <= response.status_code < 600
                if not retryable or attempt == self.retries:
                    response.raise_for_status()
                    data = bytes(response.content)
                    self.cache.put_bytes(url, data, key)
                    return data
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        delay = max(
                            delay, parsedate_to_datetime(retry_after).timestamp() - time.time()
                        )
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.retries:
                    raise
            self.sleep(max(0, delay))
        raise RuntimeError("HTTP retry loop exhausted")

    def json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        body: dict[str, Any] | None = None,
    ) -> Any:
        """Decode a cached/retried HTTP response as JSON."""
        return json.loads(self.request(url, params, body=body))
