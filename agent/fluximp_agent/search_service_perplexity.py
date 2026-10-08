"""Perplexity search service implementation."""

from perplexity import Perplexity
from fluximp_agent.search_service_base import BaseSearchService


class PerplexitySearchService(BaseSearchService):
    """Search service using Perplexity API.

    Provides raw web search with multiple search modes (web, academic, SEC filings),
    people search, and context size control. Uses the perplexityai Python SDK.

    API key is stored in keyring under service='perplexity.ai', username='api_key'.
    """

    _provider_name = "perplexity"

    def __init__(self, filesystem_service, state_service, experiments_service, io_path):
        super().__init__(filesystem_service, state_service, experiments_service, io_path)
        api_key = self._get_api_key("perplexity.ai")
        self._client = Perplexity(api_key=api_key)

    def search(self, query, search_mode=None, search_type=None,
               search_context_size=None, max_results=None,
               search_domain_filter=None, search_recency_filter=None,
               search_language_filter=None, country=None) -> dict:
        """Execute Perplexity search and save raw response.

        Performs raw web search returning ranked URLs with snippets.
        Supports academic papers, SEC filings, and people search modes.
        Results are saved to .fluximp/web/search/<workflow_id>/perplexity_search/.

        Args:
            query: Search query (string or list of strings for multi-query)
            search_mode: 'web' (general), 'academic' (academic papers),
                        'sec' (SEC filings)
            search_type: 'web' (general), 'people' (people search, supports
                         >20 results)
            search_context_size: 'low', 'medium', or 'high' - controls content
                                per result
            max_results: Maximum number of results to return
            search_domain_filter: List of domains to include in results
            search_recency_filter: 'hour', 'day', 'week', 'month', 'year'
            search_language_filter: List of language filters
            country: Country code for geo-targeting

        Returns:
            Dict with keys:
                - result_file_path: str
                - metadata_file_path: str
                - result_count: int
                - summary: str
                - file_view_update: dict
        """
        # Build kwargs dict, only including non-None values
        kwargs = self._build_kwargs(
            search_mode=search_mode, search_type=search_type,
            search_context_size=search_context_size, max_results=max_results,
            search_domain_filter=search_domain_filter, search_recency_filter=search_recency_filter,
            search_language_filter=search_language_filter, country=country,
        )

        # Execute search
        response = self._client.search.create(query=query, **kwargs)

        # Serialize raw response
        raw_content = self._serialize_response(response)

        # Build params dict for metadata (excludes query)
        params = dict(kwargs)

        # Save result + metadata
        result_path, metadata_path = self._save_result(
            tool_name="perplexity_search",
            raw_content=raw_content,
            content_type="application/json",
            query=str(query),
            query_params=params,
        )

        # Build summary
        result_count = len(response.results) if hasattr(response, 'results') else 0
        summary_parts = [
            "Perplexity search completed.",
            f'Query: "{query}"',
            f"Results: {result_count}",
        ]
        if search_mode:
            summary_parts.append(f"Search mode: {search_mode}")
        if search_type:
            summary_parts.append(f"Search type: {search_type}")
        if search_context_size:
            summary_parts.append(f"Context size: {search_context_size}")
        if max_results is not None:
            summary_parts.append(f"Max results: {max_results}")
        if search_domain_filter:
            summary_parts.append(f"Domain filter: {search_domain_filter}")
        if search_recency_filter:
            summary_parts.append(f"Recency filter: {search_recency_filter}")
        if search_language_filter:
            summary_parts.append(f"Language filter: {search_language_filter}")
        if country:
            summary_parts.append(f"Country: {country}")
        return self._build_result(
            result_path, metadata_path,
            self._format_summary(summary_parts, result_path),
            result_count=result_count,
        )
