"""Space-Track.org REST API client.

Docs: https://www.space-track.org/documentation#/api
"""
import time
import logging
from typing import Iterator, List, Optional
import requests
from config.settings import SPACETRACK_USERNAME, SPACETRACK_PASSWORD

log = logging.getLogger(__name__)

BASE_URL = "https://www.space-track.org"
LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
LOGOUT_URL = f"{BASE_URL}/ajaxauth/logout"


class SpaceTrackClient:
    """Authenticated session wrapper around Space-Track REST API."""

    def __init__(self, username: str = SPACETRACK_USERNAME,
                 password: str = SPACETRACK_PASSWORD):
        self._username = username
        self._password = password
        self._session = requests.Session()
        self._logged_in = False

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def login(self) -> None:
        resp = self._session.post(
            LOGIN_URL,
            data={"identity": self._username, "password": self._password},
            timeout=30,
        )
        resp.raise_for_status()
        if "Login" in resp.text:
            raise ValueError("Space-Track login failed – check credentials")
        self._logged_in = True
        log.info("Logged in to Space-Track")

    def logout(self) -> None:
        if self._logged_in:
            self._session.get(LOGOUT_URL, timeout=10)
            self._logged_in = False

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, *_):
        self.logout()

    # ------------------------------------------------------------------
    # Low-level request helper
    # ------------------------------------------------------------------
    def _get(self, url: str, **params) -> dict | list:
        if not self._logged_in:
            self.login()
        time.sleep(0.3)          # Space-Track rate limit: ~30 req/min
        resp = self._session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # GP (General Perturbations) / TLE data
    # ------------------------------------------------------------------
    def get_latest_gp(
        self,
        object_types: Optional[List[str]] = None,
        epoch_window_days: int = 30,
        limit: int = 0,
    ) -> List[dict]:
        """Fetch the most recent GP mean elements for catalog objects.

        Parameters
        ----------
        object_types : list[str], optional
            Filter by OBJECT_TYPE. E.g. ['DEBRIS', 'PAYLOAD', 'ROCKET BODY']
            Default: all types.
        epoch_window_days : int
            Only include objects with epoch within this many days (freshness).
        limit : int
            Maximum number of records. 0 (default) means no limit – returns the
            full active catalog (~25 000–27 000 objects as of 2026).
        """
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=epoch_window_days)
                  ).strftime("%Y-%m-%d")

        url_parts = [
            BASE_URL,
            "basicspacedata",
            "query",
            "class/gp",
            f"EPOCH/%3E{cutoff}",   # >cutoff
            "orderby/NORAD_CAT_ID asc",
            "format/json",
        ]
        if limit > 0:
            # Insert before "format/json" so the URL order is consistent
            url_parts.insert(-1, f"limit/{limit}")
        if object_types:
            ot_str = ",".join(object_types)
            url_parts.insert(5, f"OBJECT_TYPE/{ot_str}")

        url = "/".join(url_parts)
        log.info("Fetching GP data: %s", url)
        # Full-catalog responses can be 30–60 MB; use a generous timeout
        timeout = 30 if limit and limit <= 1000 else 300
        resp = self._session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()  # type: ignore[return-value]

    def get_tle_by_norad(self, norad_id: int) -> Optional[dict]:
        """Fetch a single object's latest GP record by NORAD catalog ID."""
        url = (
            f"{BASE_URL}/basicspacedata/query/class/gp"
            f"/NORAD_CAT_ID/{norad_id}/orderby/EPOCH desc/limit/1/format/json"
        )
        data = self._get(url)
        return data[0] if data else None

    def get_decay_predictions(self, limit: int = 500) -> List[dict]:
        """Fetch decay prediction records."""
        url = (
            f"{BASE_URL}/basicspacedata/query/class/decay"
            f"/orderby/DECAY_EPOCH desc/limit/{limit}/format/json"
        )
        return self._get(url)  # type: ignore[return-value]

    def get_conjunction_data(self, limit: int = 1000) -> List[dict]:
        """Fetch latest Conjunction Data Messages (CDM) screening events."""
        url = (
            f"{BASE_URL}/basicspacedata/query/class/cdm_public"
            f"/orderby/TCA desc/limit/{limit}/format/json"
        )
        return self._get(url)  # type: ignore[return-value]

    def get_launch_sites(self) -> List[dict]:
        """Fetch launch site catalog."""
        url = f"{BASE_URL}/basicspacedata/query/class/launch_site/format/json"
        return self._get(url)  # type: ignore[return-value]

    def iter_gp_chunks(
        self,
        chunk_size: int = 2000,
        total_limit: int = 50000,
        **kwargs,
    ) -> Iterator[List[dict]]:
        """Yield chunks of GP records to avoid memory spikes."""
        for offset in range(0, total_limit, chunk_size):
            url_parts = [
                BASE_URL,
                "basicspacedata",
                "query",
                "class/gp",
                "orderby/NORAD_CAT_ID asc",
                f"limit/{chunk_size},{offset}",
                "format/json",
            ]
            url = "/".join(url_parts)
            chunk = self._get(url)
            if not chunk:
                break
            yield chunk
            if len(chunk) < chunk_size:
                break
