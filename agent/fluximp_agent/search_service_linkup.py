
"""Linkup search service - provides web search and URL fetch via the Linkup API."""

import json

import linkup

from fluximp_agent.search_service_base import BaseSearchService


class LinkupSearchService(BaseSearchService):
    """Search service using the Linkup API.

    Provides web search with 3 depth modes (fast/standard/deep) and 3 output
    types (searchResults/sourcedAnswer/structured), plus URL fetch with JS
    rendering support.
    """

    _provider_name = "linkup"

    def __init__(self, filesystem_service, state_service, experiments_service, io_path):
        super().__init__(filesystem_service, state_service, experiments_service, io_path)
        # Get API key from keyring
        api_key = self._get_api_key("linkup.so")
        self._client = linkup.Client(api_key=api_key)

    def search(self, query: str, depth: str = "standard",
               output_type: str = "searchResults",
               structured_output_schema=None, include_images=None,
               from_date=None, to_date=None, exclude_domains=None,
               include_domains=None, max_results=None,
               include_inline_citations=None, include_sources=None) -> dict:
        """Execute Linkup search and save raw response.

        Performs agentic web search with configurable depth and output type.
        Results are saved to .fluximp/web/search/<workflow_id>/linkup_search/.

        Args:
            query: Search query string
            depth: 'fast' (<1s, keyword only), 'standard' (1-3s, agentic),
                   'deep' (5-30s, multi-iteration agentic search-and-scrape)
            output_type: 'searchResults' (ranked URLs), 'sourcedAnswer'
                         (answer + citations), 'structured' (JSON matching schema)
            structured_output_schema: JSON schema dict (required if output_type='structured')
            include_images: Whether to include image results
            from_date: Start date filter (YYYY-MM-DD)
            to_date: End date filter (YYYY-MM-DD)
            exclude_domains: Domains to exclude
            include_domains: Domains to restrict search to
            max_results: Maximum number of results
            include_inline_citations: For sourcedAnswer, include inline citations
            include_sources: For structured, include sources (changes response type)

        Returns:
            Dict with keys: result_file_path, metadata_file_path, summary, file_view_update
        """
        # Build kwargs for SDK - only pass non-None optional parameters
        # depth and output_type are required by the SDK
        kwargs = self._build_kwargs(
            {"depth": depth, "output_type": output_type},
            structured_output_schema=structured_output_schema,
            include_images=include_images,
            from_date=from_date,
            to_date=to_date,
            exclude_domains=exclude_domains,
            include_domains=include_domains,
            max_results=max_results,
            include_inline_citations=include_inline_citations,
            include_sources=include_sources,
        )

        # Execute search
        response = self._client.search(query, **kwargs)

        # Serialize raw response
        raw_content = self._serialize_response(response)

        # Build params dict for metadata (excludes query)
        params = dict(kwargs)

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="linkup_search",
            raw_content=raw_content,
            content_type="application/json",
            query=query,
            query_params=params,
        )

        # Build summary based on output_type
        summary_parts = [
            "Linkup search completed.",
            f'Query: "{query}"',
            f"Depth: {depth}",
            f"Output type: {output_type}",
        ]

        if output_type == "searchResults":
            results = response.results if hasattr(response, "results") else []
            result_count = len(results)
            summary_parts.append(f"Results: {result_count}")
            if results:
                for i, r in enumerate(results[:3]):
                    name = getattr(r, "name", "") if hasattr(r, "name") else ""
                    url = getattr(r, "url", "") if hasattr(r, "url") else ""
                    summary_parts.append(f"  {i+1}. {name} - {url}")
                if result_count > 3:
                    summary_parts.append(f"  ... and {result_count - 3} more")
        elif output_type == "sourcedAnswer":
            answer = getattr(response, "answer", "") if hasattr(response, "answer") else ""
            sources = getattr(response, "sources", []) if hasattr(response, "sources") else []
            answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
            summary_parts.append(f"Answer: {answer_preview}")
            summary_parts.append(f"Sources: {len(sources)}")
        elif output_type == "structured":
            if hasattr(response, "data"):
                data_preview = json.dumps(response.data, default=str)[:200]
                summary_parts.append(f"Data: {data_preview}...")
                sources = getattr(response, "sources", []) if hasattr(response, "sources") else []
                summary_parts.append(f"Sources: {len(sources)}")
            else:
                data_preview = json.dumps(response, default=str)[:200]
                summary_parts.append(f"Data: {data_preview}...")

        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
        )

    def fetch(self, url: str, render_js: bool = True,
              include_raw_html: bool = False,
              extract_images: bool = False) -> dict:
        """Fetch URL content via Linkup API and save raw response.

        Fetches a single URL and returns clean markdown content with optional
        raw HTML and image extraction. Supports JavaScript rendering for
        dynamic pages. Results are saved to
        .fluximp/web/page/<domain>/<workflow_id>/linkup_fetch/.

        Args:
            url: URL to fetch (single URL, not a list)
            render_js: Whether to render JavaScript before extraction (default: True)
            include_raw_html: Whether to include raw HTML in response (default: False)
            extract_images: Whether to extract images from the page (default: False)

        Returns:
            Dict with keys: result_file_path, metadata_file_path, summary, file_view_update
        """
        # Build kwargs for SDK - only pass non-None optional parameters
        kwargs = self._build_kwargs(
            render_js=render_js,
            include_raw_html=include_raw_html,
            extract_images=extract_images,
        )

        # Execute fetch
        response = self._client.fetch(url, **kwargs)

        # Serialize raw response
        raw_content = self._serialize_response(response)

        # Determine domain for file path
        domain = self._first_url_domain(url)

        # Build params dict for metadata
        params = {
            "url": url,
            "render_js": render_js,
            "include_raw_html": include_raw_html,
            "extract_images": extract_images,
        }

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="linkup_fetch",
            raw_content=raw_content,
            content_type="application/json",
            query=url,
            query_params=params,
            category="page",
            subdir=domain,
        )

        # Build summary
        content_len = len(response.markdown) if hasattr(response, "markdown") else 0
        summary_parts = [
            "Linkup fetch completed.",
            f"URL: {url}",
            f"Content length: {content_len} chars",
            f"JS rendering: {render_js}",
        ]
        if include_raw_html:
            raw_len = len(response.raw_html) if hasattr(response, "raw_html") and response.raw_html else 0
            summary_parts.append(f"Raw HTML: {raw_len} chars")
        if extract_images:
            img_count = len(response.images) if hasattr(response, "images") and response.images else 0
            summary_parts.append(f"Images: {img_count}")
        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
        )
