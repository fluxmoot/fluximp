"""Exa search service.

Wraps the exa_py SDK to execute neural/semantic searches and save raw
responses to the .fluximp/web/search/ directory with metadata sidecar files.
"""

from exa_py import Exa
from fluximp_agent.search_service_base import BaseSearchService


class ExaSearchService(BaseSearchService):
    """Search service for Exa API.

    Wraps the exa_py SDK. API key retrieved from keyring
    using service name 'exa.ai' and username 'api_key'.
    """

    _provider_name = "exa"

    def __init__(self, filesystem_service, state_service, experiments_service, io_path):
        super().__init__(filesystem_service, state_service, experiments_service, io_path)

        # Get API key from keyring
        api_key = self._get_api_key("exa.ai")
        self._client = Exa(api_key=api_key)

    def search(self, query: str, num_results: int = 10,
               type: str = None, category: str = None,
               include_domains=None, exclude_domains=None,
               start_published_date=None, end_published_date=None,
               contents=None) -> dict:
        """Execute Exa search and save raw response.

        Maps to Exa.search() - Exa's neural/semantic search endpoint.

        Args:
            query: Natural-language search query
            num_results: Number of results (1-100, default 10)
            type: Search type - 'auto', 'instant', 'fast', 'deep-lite',
                  'deep', 'deep-reasoning'
            category: Data category - 'company', 'research paper', 'news',
                      'personal site', 'financial report', 'people'
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            start_published_date: ISO 8601 date filter (published after)
            end_published_date: ISO 8601 date filter (published before)
            contents: Dict of content options, e.g. {"highlights": True},
                      {"text": True}, {"summary": True}, or False to disable

        Returns:
            Dict with keys:
                - result_file_path: str (relative path to saved raw result)
                - metadata_file_path: str (relative path to metadata sidecar)
                - result_count: int (number of results)
                - summary: str (concise summary for agent)
                - file_view_update: dict (state update for file view windows)
        """
        # Build kwargs, only including non-None values (Exa SDK is keyword-only)
        kwargs = self._build_kwargs(
            type=type, category=category,
            include_domains=include_domains, exclude_domains=exclude_domains,
            start_published_date=start_published_date, end_published_date=end_published_date,
            contents=contents,
        )

        # Execute search
        response = self._client.search(
            query=query,
            num_results=num_results,
            **kwargs,
        )

        # Serialize raw response as JSON
        raw_content = self._serialize_response(response)

        # Build params dict for metadata
        params = {
            "num_results": num_results,
            "type": type,
            "category": category,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "start_published_date": start_published_date,
            "end_published_date": end_published_date,
            "contents": contents,
        }

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="exa_search",
            raw_content=raw_content,
            content_type="application/json",
            query=query,
            query_params=params,
        )

        # Build summary
        result_count = len(response.results) if hasattr(response, 'results') else 0
        resolved_type = getattr(response, 'resolved_search_type', None)
        summary_parts = [
            "Exa search completed.",
            f'Query: "{query}"',
            f"Results: {result_count}",
            f"Search type: {type or 'auto'}",
        ]
        if resolved_type:
            summary_parts.append(f"Resolved type: {resolved_type}")
        if category:
            summary_parts.append(f"Category: {category}")

        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=result_count,
        )

    def contents(self, urls, text=True, highlights=False, summary=False,
                 subpages=None, subpage_target=None) -> dict:
        """Extract content from known URLs using Exa's Contents API.

        Retrieves clean content from URLs with options for highlights
        (token-efficient relevant extracts), full text, AI summaries, and
        subpage exploration. Results are saved to
        .fluximp/web/page/<domain>/<workflow_id>/exa_contents/.

        Args:
            urls: URL or list of URLs to extract content from
            text: Include full page text (default: True)
            highlights: Include token-efficient relevant extracts (default: False)
            summary: Include AI-generated summary (default: False)
            subpages: Number of subpages to explore per URL (default: None)
            subpage_target: Comma-separated subpage paths to target
                           (e.g., "about,contact,blog")

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

        # Build kwargs for SDK - explicitly pass text to prevent SDK auto-default
        kwargs = self._build_kwargs(
            {"text": True if text else False},
            highlights=True if highlights else None,
            summary=True if summary else None,
            subpages=subpages,
            subpage_target=subpage_target,
        )

        # Execute contents extraction
        response = self._client.get_contents(urls, **kwargs)

        # Serialize raw response
        raw_content = self._serialize_response(response)

        # Determine domain from first URL for file path
        domain = self._first_url_domain(urls)

        # Build params dict for metadata
        params = {
            "urls": urls,
            "text": text,
            "highlights": highlights,
            "summary": summary,
            "subpages": subpages,
            "subpage_target": subpage_target,
        }

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="exa_contents",
            raw_content=raw_content,
            content_type="application/json",
            query=first_url,
            query_params=params,
            category="page",
            subdir=domain,
        )

        # Build summary
        result_count = len(response.results) if hasattr(response, 'results') else 0
        content_types = []
        if text:
            content_types.append("text")
        if highlights:
            content_types.append("highlights")
        if summary:
            content_types.append("summary")
        summary_parts = [
            "Exa contents extraction completed.",
            f"URLs requested: {len(urls)}",
            f"Results: {result_count}",
            f"Content: {', '.join(content_types) or 'none'}",
        ]
        if subpages is not None:
            summary_parts.append(f"Subpages explored: {subpages}")
        if subpage_target:
            summary_parts.append(f"Subpage target: {subpage_target}")
        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=result_count,
        )
