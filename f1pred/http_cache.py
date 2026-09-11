"""Throttled, on-disk HTTP cache with provenance.

Every external fetch goes through :class:`CachedSession`. Responses are stored
under ``data/cache`` keyed by a hash of the URL, alongside a sidecar JSON record
holding the URL and the UTC fetch time. Experiments are therefore reproducible
offline, and every derived feature can be traced back to a timestamped request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from .config import CACHE_DIR, HTTP_RETRIES, HTTP_TIMEOUT_S, USER_AGENT

log = logging.getLogger(__name__)

# Status codes worth retrying: rate limiting and transient server faults.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass
class FetchRecord:
    """Provenance for one cached response."""

    url: str
    fetched_at: str
    status: int
    bytes: int


class CachedSession:
    """A polite HTTP client: one shared connection, rate limit, disk cache.

    Parameters
    ----------
    name:
        Sub-directory of the cache, one per upstream API.
    min_interval_s:
        Minimum wall-clock spacing between two requests to this API.
    burst:
        Optional sustained cap, as ``(requests, seconds)``. A minimum spacing
        alone cannot honour a published per-minute quota: three requests a second
        satisfies any 0.33 s spacing rule and still hits ninety a minute. Both
        limits are enforced together when a window is given.
    """

    def __init__(
        self,
        name: str,
        min_interval_s: float,
        cache_dir: Path | None = None,
        burst: tuple[int, float] | None = None,
    ) -> None:
        self.name = name
        self.min_interval_s = float(min_interval_s)
        self.burst = burst
        self.cache_dir = (cache_dir or CACHE_DIR) / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._lock = threading.Lock()
        self._last_request_at = 0.0
        self._recent: deque[float] = deque()

    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            if self.burst:
                limit, window = self.burst
                now = time.monotonic()
                while self._recent and now - self._recent[0] >= window:
                    self._recent.popleft()
                if len(self._recent) >= limit:
                    # The oldest request in the window has to age out before this
                    # one is allowed, or the sustained quota is breached.
                    sleep_for = window - (now - self._recent[0])
                    if sleep_for > 0:
                        log.debug(
                            "%s: sustained limit reached, waiting %.1fs", self.name, sleep_for
                        )
                        time.sleep(sleep_for)
                    while self._recent and time.monotonic() - self._recent[0] >= window:
                        self._recent.popleft()
                self._recent.append(time.monotonic())
            self._last_request_at = time.monotonic()

    def _paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{key}.json", self.cache_dir / f"{key}.meta.json"

    def get_json(self, url: str, *, refresh: bool = False) -> Any:
        """Return parsed JSON for ``url``, fetching only on a cache miss.

        Raises
        ------
        RuntimeError
            If every retry failed and no cached copy exists.
        """
        body_path, meta_path = self._paths(url)
        if body_path.exists() and not refresh:
            return json.loads(body_path.read_text())

        last_error: Exception | None = None
        for attempt in range(HTTP_RETRIES):
            self._throttle()
            try:
                response = self._session.get(url, timeout=HTTP_TIMEOUT_S)
            except requests.RequestException as exc:  # network-level failure
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code in _RETRY_STATUS:
                last_error = RuntimeError(f"HTTP {response.status_code}")
                # Honour Retry-After when the server supplies it.
                delay = float(response.headers.get("Retry-After") or 0) or 2.0 * (attempt + 1)
                log.debug("%s rate limited on %s; sleeping %.1fs", self.name, url, delay)
                time.sleep(delay)
                continue
            if response.status_code == 404:
                raise FileNotFoundError(f"404 for {url}")
            response.raise_for_status()
            payload = response.json()
            body_path.write_text(json.dumps(payload))
            meta_path.write_text(
                json.dumps(
                    FetchRecord(
                        url=url,
                        fetched_at=datetime.now(UTC).isoformat(),
                        status=response.status_code,
                        bytes=len(response.content),
                    ).__dict__
                )
            )
            return payload

        if body_path.exists():
            log.warning("%s: serving stale cache for %s (%s)", self.name, url, last_error)
            return json.loads(body_path.read_text())
        raise RuntimeError(f"{self.name}: failed to fetch {url}") from last_error

    def provenance(self) -> list[FetchRecord]:
        """Return the fetch record of every cached response for this API."""
        records: list[FetchRecord] = []
        for meta in sorted(self.cache_dir.glob("*.meta.json")):
            records.append(FetchRecord(**json.loads(meta.read_text())))
        return records

    def record_for(self, url: str) -> FetchRecord:
        """Return provenance for a cached URL, never inventing a fetch timestamp."""
        _, metadata = self._paths(url)
        return FetchRecord(**json.loads(metadata.read_text()))
