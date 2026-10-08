"""Tavily search service.

Wraps the tavily-python SDK to execute searches and save raw responses
to the .fluximp/web/search/ directory with metadata sidecar files.
"""

from tavily import TavilyClient
from fluximp_agent.search_service_base import BaseSearchService


class TavilySearchService(BaseSearchService):
    """Search service for Tavily API.

    Wraps the tavily-python SDK. API key retrieved from keyring
    using service name 'tavily.com' and username 'api_key'.
    """

    _provider_name = "tavily"

    def __init__(self, filesystem_service, state_service, experiments_service, io_path):
        super().__init__(filesystem_service, state_service, experiments_service, io_path)

        # Get API key from keyring
        api_key = self._get_api_key("tavily.com")
        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5,
               search_depth: str = "basic", topic: str = "general",
               include_answer: bool = False, include_raw_content: bool = False,
               include_domains=None, exclude_domains=None) -> dict:
        """Execute Tavily search and save raw response.

        Maps to TavilyClient.search() - the full Tavily search endpoint.

        Returns:
            Dict with keys:
                - result_file_path: str (relative path to saved raw result)
                - metadata_file_path: str (relative path to metadata sidecar)
                - result_count: int (number of results)
                - summary: str (concise summary for agent)
                - file_view_update: dict (state update for file view windows)
        """
        # Build params dict for metadata
        params = {
            "max_results": max_results,
            "search_depth": search_depth,
            "topic": topic,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
        }

        # Execute search
        response = self._client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            topic=topic,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

        # Serialize raw response as JSON
        raw_content = self._serialize_response(response)

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="tavily_search",
            raw_content=raw_content,
            content_type="application/json",
            query=query,
            query_params=params,
        )

        # Build summary
        result_count = len(response.get("results", []))
        has_answer = bool(response.get("answer"))
        summary_parts = [
            f"Tavily search completed.",
            f'Query: "{query}"',
            f"Results: {result_count}",
            f"Search depth: {search_depth}",
        ]
        if has_answer:
            summary_parts.append("AI answer included.")
        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=result_count,
        )

    def extract(self, urls, extract_depth=None, format=None,
                include_images=None, include_favicon=None,
                include_usage=None, query=None,
                chunks_per_source=None) -> dict:
        """Execute Tavily extract and save raw response.

        Extracts clean content from one or more URLs. Results are saved
        to .fluximp/web/page/<domain>/<workflow_id>/tavily_extract/.
        """
        domain = self._first_url_domain(urls)
        first_url = urls if isinstance(urls, str) else (urls[0] if urls else "")

        params = {
            "urls": urls,
            "extract_depth": extract_depth,
            "format": format,
            "include_images": include_images,
            "include_favicon": include_favicon,
            "include_usage": include_usage,
            "query": query,
            "chunks_per_source": chunks_per_source,
        }

        response = self._client.extract(
            urls=urls,
            extract_depth=extract_depth,
            format=format,
            include_images=include_images,
            include_favicon=include_favicon,
            include_usage=include_usage,
            query=query,
            chunks_per_source=chunks_per_source,
        )

        raw_content = self._serialize_response(response)

        result_path, metadata_path = self._save_result(
            tool_name="tavily_extract",
            raw_content=raw_content,
            content_type="application/json",
            query=first_url,
            query_params=params,
            category="page",
            subdir=domain,
        )

        results = response.get("results", [])
        failed_results = response.get("failed_results", [])
        url_count = len(urls) if isinstance(urls, list) else 1
        summary_parts = [
            "Tavily extract completed.",
            f"URLs requested: {url_count}",
            f"Successfully extracted: {len(results)}",
            f"Failed: {len(failed_results)}",
            f"Extract depth: {extract_depth or 'basic'}",
        ]

        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=len(results),
            failed_count=len(failed_results),
        )
