"""TinyFish search service.

Uses urllib.request (Python standard library) to call TinyFish's Fetch API
and save raw responses to the .fluximp/web/page/<domain>/ directory with
metadata sidecar files.
"""

import json
import time
import urllib.request
from fluximp_agent.search_service_base import BaseSearchService


class _RateLimiter:
    """Sliding-window rate limiter."""

    def __init__(self, max_calls: int, period_seconds: float):
        self._max_calls = max_calls
        self._period_seconds = period_seconds
        self._timestamps = []

    def acquire(self, count: int = 1) -> float:
        """Reserve ``count`` slots, sleeping if necessary. Returns seconds slept."""
        now = time.time()
        self._prune(now)

        sleep_time = 0.0
        while self._timestamps and len(self._timestamps) + count > self._max_calls:
            oldest = self._timestamps[0]
            wait = self._period_seconds - (now - oldest) + 0.1
            if wait > 0:
                time.sleep(wait)
                sleep_time += wait
            now = time.time()
            self._prune(now)

        record_time = time.time()
        self._timestamps.extend([record_time] * count)
        return sleep_time

    def _prune(self, now: float):
        self._timestamps = [t for t in self._timestamps if now - t < self._period_seconds]


class TinyFishSearchService(BaseSearchService):
    """Search service for TinyFish Fetch API.

    Uses REST API via urllib.request (no SDK dependency).
    API key retrieved from keyring using service name 'tinyfish.ai'
    and username 'api_key'.
    """

    _provider_name = "tinyfish"
    _FETCH_API_URL = "https://api.fetch.tinyfish.ai"
    _RATE_LIMIT_URLS = 50  # Max URLs per minute (TinyFish allows 150; 50 allows 3 parallel processes)
    _RATE_LIMIT_WINDOW = 60  # Window size in seconds
    _SEARCH_API_URL = "https://api.search.tinyfish.ai"
    _RATE_LIMIT_QUERIES = 10  # Max queries per minute (TinyFish allows 30; 10 allows 3 parallel processes)

    def __init__(self, filesystem_service, state_service, experiments_service, io_path):
        super().__init__(filesystem_service, state_service, experiments_service, io_path)

        # Get API key from keyring
        self._api_key = self._get_api_key("tinyfish.ai")
        self._fetch_url_limiter = _RateLimiter(
            self._RATE_LIMIT_URLS, self._RATE_LIMIT_WINDOW
        )
        self._search_limiter = _RateLimiter(
            self._RATE_LIMIT_QUERIES, self._RATE_LIMIT_WINDOW
        )

    def fetch(self, urls, format: str = "markdown", links: bool = False,
              image_links: bool = False, ttl: int = 3600,
              per_url_timeout_ms: int = 45000) -> dict:
        """Execute TinyFish fetch and save raw response.

        Fetches and extracts clean content from URLs using TinyFish's
        browser-rendered fetch API. Results are saved to
        .fluximp/web/page/<domain>/<workflow_id>/tinyfish_fetch/.

        Args:
            urls: List of URLs to fetch (max 10 per request)
            format: Output format - 'markdown', 'html', or 'json'
                    (default: 'markdown')
            links: Include extracted links in response (default: False)
            image_links: Include image links in response (default: False)
            ttl: Cache TTL in seconds (default: 3600)
            per_url_timeout_ms: Per-URL fetch timeout in milliseconds
                               (default: 45000)

        Returns:
            Dict with keys:
                - result_file_path: str (relative path to saved raw result)
                - metadata_file_path: str (relative path to metadata sidecar)
                - result_count: int (number of successfully fetched URLs)
                - failed_count: int (number of failed URLs)
                - summary: str (concise summary for agent)
                - file_view_update: dict (state update for file view windows)
        """
        # Ensure urls is a list for the API request
        urls = self._ensure_list(urls)

        domain = self._first_url_domain(urls)
        first_url = urls[0] if urls else ""

        # Enforce self-rate limiting (50 URLs/min per process)
        rate_limit_sleep = self._fetch_url_limiter.acquire(len(urls))

        # Build request body
        body = {
            "urls": urls,
            "format": format,
            "links": links,
            "image_links": image_links,
            "ttl": ttl,
            "per_url_timeout_ms": per_url_timeout_ms,
        }
        body_json = json.dumps(body).encode("utf-8")

        # Build params dict for metadata
        params = {
            "urls": urls,
            "format": format,
            "links": links,
            "image_links": image_links,
            "ttl": ttl,
            "per_url_timeout_ms": per_url_timeout_ms,
        }

        # Execute POST request
        req = urllib.request.Request(
            self._FETCH_API_URL,
            data=body_json,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self._api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))

        # Serialize raw response as JSON
        raw_content = self._serialize_response(response_data)

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="tinyfish_fetch",
            raw_content=raw_content,
            content_type="application/json",
            query=first_url,
            query_params=params,
            category="page",
            subdir=domain,
        )

        # Build summary
        results = response_data.get("results", [])
        errors = response_data.get("errors", [])
        url_count = len(urls)
        summary_parts = [
            "TinyFish fetch completed.",
            f"URLs requested: {url_count}",
            f"Successfully fetched: {len(results)}",
            f"Failed: {len(errors)}",
            f"Format: {format}",
        ]
        if rate_limit_sleep > 0:
            summary_parts.append(f"Rate limit: slept {rate_limit_sleep:.1f}s to stay within 50 URLs/min")

        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=len(results),
            failed_count=len(errors),
        )

    def search(self, query: str, domain_type: str = "web",
               page: int = 0, recency_minutes: int = None,
               after_date: str = None, before_date: str = None,
               location: str = None, language: str = None) -> dict:
        """Execute TinyFish search and save raw response.

        Performs structured web search returning titles, snippets, and URLs.
        Results are saved to .fluximp/web/search/<workflow_id>/tinyfish_search/.

        Args:
            query: Search query. Supports operators like site: and -site:
            domain_type: 'web' (default), 'news', or 'research_paper'
            page: Page number (0-10, default: 0)
            recency_minutes: Freshness window in minutes (1-5256000).
                            Cannot combine with date filters.
            after_date: YYYY-MM-DD; only results after this date.
                        Must be <= before_date.
            before_date: YYYY-MM-DD; only results before this date.
            location: Geo-targeting code (e.g., 'US', 'GB')
            language: Language code (e.g., 'en', 'fr')

        Returns:
            Dict with keys:
                - result_file_path: str
                - metadata_file_path: str
                - result_count: int
                - summary: str
                - file_view_update: dict
        """
        from urllib.parse import urlencode

        # Enforce self-rate limiting (10 queries/min per process)
        rate_limit_sleep = self._search_limiter.acquire(1)

        # Build query parameters for API URL
        url_params = {"query": query, "domain_type": domain_type, "page": page}
        if recency_minutes is not None:
            url_params["recency_minutes"] = recency_minutes
        if after_date is not None:
            url_params["after_date"] = after_date
        if before_date is not None:
            url_params["before_date"] = before_date
        if location is not None:
            url_params["location"] = location
        if language is not None:
            url_params["language"] = language

        # Build URL with encoded query parameters
        url = f"{self._SEARCH_API_URL}?{urlencode(url_params)}"

        # Execute GET request
        req = urllib.request.Request(
            url,
            headers={"X-API-Key": self._api_key},
            method="GET",
        )
        with urllib.request.urlopen(req) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))

        # Serialize raw response
        raw_content = self._serialize_response(response_data)

        # Build params dict for metadata (excludes query, which is passed separately)
        params = {k: v for k, v in url_params.items() if k != "query"}

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="tinyfish_search",
            raw_content=raw_content,
            content_type="application/json",
            query=query,
            query_params=params,
        )

        # Build summary
        results = response_data.get("results", [])
        result_count = len(results)
        summary_parts = [
            "TinyFish search completed.",
            f'Query: "{query}"',
            f"Results: {result_count}",
            f"Domain type: {domain_type}",
            f"Page: {page}",
        ]
        if recency_minutes is not None:
            summary_parts.append(f"Recency: {recency_minutes} minutes")
        if after_date:
            summary_parts.append(f"After date: {after_date}")
        if before_date:
            summary_parts.append(f"Before date: {before_date}")
        if location:
            summary_parts.append(f"Location: {location}")
        if language:
            summary_parts.append(f"Language: {language}")
        if rate_limit_sleep > 0:
            summary_parts.append(f"Rate limit: slept {rate_limit_sleep:.1f}s to stay within 10 queries/min")
        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=result_count,
        )
