"""Parallel.ai search and extract service.

Provides search (declarative semantic search with objective + keyword queries)
and extract (objective-driven content extraction from URLs) using the Parallel
API via the parallel-web SDK.
"""

from parallel import Parallel
from fluximp_agent.search_service_base import BaseSearchService


class ParallelSearchService(BaseSearchService):
    """Service for Parallel.ai search and extract APIs.

    Uses the parallel-web SDK (from parallel import Parallel).
    API key is stored in keyring under service='parallel.ai', username='api_key'.
    """

    _provider_name = "parallel"

    def __init__(self, filesystem_service, state_service, experiments_service, io_path):
        super().__init__(filesystem_service, state_service, experiments_service, io_path)
        api_key = self._get_api_key("parallel.ai")
        self._client = Parallel(api_key=api_key)

    def search(self, search_queries, objective=None, mode=None,
               max_chars_total=None, session_id=None) -> dict:
        """Execute Parallel search and save raw response.

        Performs declarative semantic search using keyword search_queries plus
        an optional natural-language objective. Returns ranked URLs with
        token-dense compressed excerpts from Parallel's proprietary web index.
        Results are saved to .fluximp/web/search/<workflow_id>/parallel_search/.

        Args:
            search_queries: List of 2-3 concise keyword queries, 3-6 words each.
            objective: Natural-language description of the underlying question
                       or goal driving the search.
            mode: Search mode: 'turbo' (fastest), 'basic' (low latency), or
                  'advanced' (highest quality, default when omitted by SDK).
            max_chars_total: Upper bound on total characters across excerpts.
            session_id: Session identifier to track calls across search/extract.

        Returns:
            Dict with keys:
                - result_file_path: str
                - metadata_file_path: str
                - result_count: int
                - summary: str
                - file_view_update: dict
        """
        # Ensure search_queries is a list
        search_queries = self._ensure_list(search_queries)

        # Build kwargs for SDK call
        kwargs = self._build_kwargs(
            {"search_queries": search_queries},
            objective=objective,
            mode=mode,
            max_chars_total=max_chars_total,
            session_id=session_id,
        )

        # Execute search
        response = self._client.search(**kwargs)

        # Serialize raw response
        raw_content = self._serialize_response(response)

        # Build query string for metadata (use first search query)
        query_str = search_queries[0] if search_queries else ""

        # Build params dict for metadata
        params = {
            "search_queries": search_queries,
            "objective": objective,
            "mode": mode,
            "max_chars_total": max_chars_total,
            "session_id": session_id,
        }

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="parallel_search",
            raw_content=raw_content,
            content_type="application/json",
            query=query_str,
            query_params=params,
        )

        # Build summary
        results = response.results if hasattr(response, "results") else []
        result_count = len(results)
        summary_parts = [
            "Parallel search completed.",
            f"Queries: {search_queries}",
        ]
        if objective:
            summary_parts.append(f"Objective: {objective}")
        if mode:
            summary_parts.append(f"Mode: {mode}")
        else:
            summary_parts.append("Mode: advanced (default)")
        summary_parts.append(f"Results: {result_count}")
        if hasattr(response, "session_id") and response.session_id:
            summary_parts.append(f"Session ID: {response.session_id}")
        if hasattr(response, "warnings") and response.warnings:
            summary_parts.append(f"Warnings: {len(response.warnings)}")
        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=result_count,
        )

    def extract(self, urls, objective=None, search_queries=None,
                full_content=False, max_chars_total=None,
                session_id=None) -> dict:
        """Execute Parallel extract and save raw response.

        Extracts content from specific URLs with optional objective-driven
        excerpts. By default returns ranked excerpts; set full_content=True
        for full page markdown. Handles JS-rendered pages and PDFs.
        Results are saved to .fluximp/web/page/<domain>/<workflow_id>/parallel_extract/.

        Args:
            urls: List of URLs to extract content from (up to 20).
            objective: Natural-language extraction goal. When provided,
                       excerpts are focused on relevant content.
            search_queries: Optional keyword queries to focus excerpts.
            full_content: If True, also return full page markdown content
                          (default: False, excerpts only).
            max_chars_total: Upper bound on total excerpt characters.
            session_id: Session identifier to track calls across search/extract.

        Returns:
            Dict with keys:
                - result_file_path: str
                - metadata_file_path: str
                - result_count: int
                - summary: str
                - file_view_update: dict
        """

        # Ensure urls is a list
        urls = self._ensure_list(urls)
        first_url = urls[0] if urls else ""

        # Build kwargs for SDK call
        kwargs = self._build_kwargs(
            {"urls": urls},
            objective=objective,
            search_queries=search_queries,
            advanced_settings={"full_content": True} if full_content else None,
            max_chars_total=max_chars_total,
            session_id=session_id,
        )

        # Execute extract
        response = self._client.extract(**kwargs)

        # Serialize raw response
        raw_content = self._serialize_response(response)

        # Determine domain from first URL for file path
        domain = self._first_url_domain(urls)

        # Build params dict for metadata
        params = {
            "urls": urls,
            "objective": objective,
            "search_queries": search_queries,
            "full_content": full_content,
            "max_chars_total": max_chars_total,
            "session_id": session_id,
        }

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="parallel_extract",
            raw_content=raw_content,
            content_type="application/json",
            query=first_url,
            query_params=params,
            category="page",
            subdir=domain,
        )

        # Build summary
        results = response.results if hasattr(response, "results") else []
        errors = response.errors if hasattr(response, "errors") else []
        result_count = len(results)
        error_count = len(errors)
        summary_parts = [
            "Parallel extract completed.",
            f"URLs requested: {len(urls)}",
            f"Results: {result_count}",
        ]
        if error_count > 0:
            summary_parts.append(f"Errors: {error_count}")
            for err in errors[:3]:
                summary_parts.append(f"  - {err.url}: {err.error_type}")
        if objective:
            summary_parts.append(f"Objective: {objective}")
        if full_content:
            summary_parts.append("Content: excerpts + full content")
        else:
            summary_parts.append("Content: excerpts only")
        if hasattr(response, "session_id") and response.session_id:
            summary_parts.append(f"Session ID: {response.session_id}")
        if hasattr(response, "warnings") and response.warnings:
            summary_parts.append(f"Warnings: {len(response.warnings)}")
        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=result_count,
        )
