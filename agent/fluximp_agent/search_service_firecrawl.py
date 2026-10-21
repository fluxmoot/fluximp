"""Firecrawl search and scrape service.

Wraps the firecrawl-py SDK to execute web searches and single-URL scrapes and
save raw responses to the .fluximp/web/ directory with metadata sidecar files.
"""

from firecrawl import Firecrawl
from fluximp_agent.search_service_base import BaseSearchService


class FirecrawlSearchService(BaseSearchService):
    """Search service for Firecrawl API.

    Wraps the firecrawl-py SDK (v2 client). API key retrieved from keyring
    using service name 'firecrawl.dev' and username 'api_key'.
    """

    _provider_name = "firecrawl"

    def __init__(self, filesystem_service, state_service, experiments_service, io_path):
        super().__init__(filesystem_service, state_service, experiments_service, io_path)

        # Get API key from keyring
        api_key = self._get_api_key("firecrawl.dev")
        self._client = Firecrawl(api_key=api_key)

    def search(self, query: str, sources=None, categories=None, limit: int = 5,
               tbs: str = None, location: str = None,
               ignore_invalid_urls: bool = None, timeout: int = None) -> dict:
        """Execute Firecrawl search and save raw response.

        Maps to Firecrawl client.search() - discovery search over Firecrawl's
        web index. Results are saved to
        .fluximp/web/search/<workflow_id>/firecrawl_search/.

        Args:
            query: Search query string. Use 'site:example.com' to limit results
                   to a domain.
            sources: List of source names to search - 'web', 'news', 'images'
            categories: List of categories to filter results - 'developer',
                        'gov', 'research', 'pdf'
            limit: Maximum number of results per source (default 5)
            tbs: Time-based filter (e.g. 'qdr:d', 'qdr:w')
            location: Localized results (e.g. 'San Francisco,California,United States')
            ignore_invalid_urls: Drop URLs that cannot be scraped by other endpoints
            timeout: Request timeout in milliseconds

        Returns:
            Dict with keys:
                - result_file_path: str (relative path to saved raw result)
                - metadata_file_path: str (relative path to metadata sidecar)
                - result_count: int (number of results)
                - summary: str (concise summary for agent)
                - file_view_update: dict (state update for file view windows)
        """
        # Build kwargs, only including non-None values (SDK is keyword-only)
        kwargs = self._build_kwargs(
            {"limit": limit},
            sources=sources,
            categories=categories,
            tbs=tbs,
            location=location,
            ignore_invalid_urls=ignore_invalid_urls,
            timeout=timeout,
        )

        # Execute search
        response = self._client.search(query, **kwargs)

        # Serialize raw response as JSON
        raw_content = self._serialize_response(response)

        # Build params dict for metadata (excludes query)
        params = {
            "sources": sources,
            "categories": categories,
            "limit": limit,
            "tbs": tbs,
            "location": location,
            "ignore_invalid_urls": ignore_invalid_urls,
            "timeout": timeout,
        }

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="firecrawl_search",
            raw_content=raw_content,
            content_type="application/json",
            query=query,
            query_params=params,
        )

        # Build summary - SearchData has optional web/news/images buckets
        web = getattr(response, "web", None) or []
        news = getattr(response, "news", None) or []
        images = getattr(response, "images", None) or []
        result_count = len(web) + len(news) + len(images)
        summary_parts = [
            "Firecrawl search completed.",
            f'Query: "{query}"',
            f"Results: {result_count}",
            f"Web: {len(web)}, News: {len(news)}, Images: {len(images)}",
        ]
        if sources:
            summary_parts.append(f"Sources: {sources}")
        if categories:
            summary_parts.append(f"Categories: {categories}")
        if tbs:
            summary_parts.append(f"Time filter: {tbs}")
        if location:
            summary_parts.append(f"Location: {location}")

        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=result_count,
        )

    def scrape(self, url: str, formats=None, only_main_content: bool = None,
               wait_for: int = None, max_age: int = None,
               proxy: str = None, timeout: int = None) -> dict:
        """Execute Firecrawl scrape and save raw response.

        Maps to Firecrawl client.scrape() - scrapes a single URL and returns
        the requested formats (markdown by default). Results are saved to
        .fluximp/web/page/<domain>/<workflow_id>/firecrawl_scrape/.

        Args:
            url: URL to scrape (single URL, not a list)
            formats: List of output formats - 'markdown' (default), 'html',
                     'rawHtml', 'links', 'images', 'screenshot', 'summary'
            only_main_content: Strip nav, footer, and other boilerplate
            wait_for: Milliseconds to wait for the page to render
            max_age: Return cached content up to this age (milliseconds)
            proxy: Proxy control - 'basic', 'stealth', 'enhanced', 'auto'
            timeout: Request timeout in milliseconds

        Returns:
            Dict with keys:
                - result_file_path: str
                - metadata_file_path: str
                - result_count: int
                - summary: str
                - file_view_update: dict
        """
        # Default to markdown when no formats specified
        if not formats:
            formats = ["markdown"]

        # Build kwargs, only including non-None values
        kwargs = self._build_kwargs(
            {"formats": formats},
            only_main_content=only_main_content,
            wait_for=wait_for,
            max_age=max_age,
            proxy=proxy,
            timeout=timeout,
        )

        # Execute scrape
        response = self._client.scrape(url, **kwargs)

        # Serialize raw response
        raw_content = self._serialize_response(response)

        # Determine domain from URL for file path
        domain = self._first_url_domain(url)

        # Build params dict for metadata
        params = {
            "url": url,
            "formats": formats,
            "only_main_content": only_main_content,
            "wait_for": wait_for,
            "max_age": max_age,
            "proxy": proxy,
            "timeout": timeout,
        }

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="firecrawl_scrape",
            raw_content=raw_content,
            content_type="application/json",
            query=url,
            query_params=params,
            category="page",
            subdir=domain,
        )

        # Build summary
        markdown = getattr(response, "markdown", None) or ""
        summary_parts = [
            "Firecrawl scrape completed.",
            f"URL: {url}",
            f"Formats: {', '.join(formats)}",
            f"Markdown length: {len(markdown)} chars",
        ]
        if only_main_content is not None:
            summary_parts.append(f"Only main content: {only_main_content}")
        if proxy:
            summary_parts.append(f"Proxy: {proxy}")

        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=1,
        )
