"""Search tool functions for web search providers.

Each tool function wraps a search provider service, saves raw results
to files, and adds the result file to the agent's file view windows.
"""

from fluximp_agent.tools.tool_definitions import ToolResult


def tavily_search_tool(query: str, max_results: int = 5,
                       search_depth: str = "basic", topic: str = "general",
                       include_answer: bool = False, include_raw_content: bool = False,
                       include_domains=None, exclude_domains=None, **kwargs) -> ToolResult:
    """Search the web using Tavily's AI-optimized search API.

    Saves the raw JSON response to .fluximp/web/search/ and adds the result
    file to the agent's file view windows. The raw response contains all
    data returned by the Tavily API, including results with titles, URLs,
    content snippets, relevance scores, and optionally an AI-generated answer.
    """
    tavily_search_service = kwargs["tavily_search_service"]

    try:
        result = tavily_search_service.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            topic=topic,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    return ToolResult(
        success=True,
        content=result["summary"],
        state_updates=result.get("file_view_update", {})
    )


def tavily_extract_tool(urls, extract_depth=None, format=None,
                        include_images=None, include_favicon=None,
                        include_usage=None, query=None,
                        chunks_per_source=None, **kwargs) -> ToolResult:
    """Extract clean content from one or more URLs using Tavily's extract API.

    Given URLs, retrieves and extracts the main content from those pages.
    Raw JSON response is saved to a file and added to your file view windows.
    Use extract_depth='advanced' for higher success rate and more data.
    Use format='text' for plain text instead of markdown.

    Saves results to .fluximp/web/page/<domain>/<conversation_id>/tavily_extract/.
    """
    tavily_search_service = kwargs["tavily_search_service"]

    try:
        result = tavily_search_service.extract(
            urls=urls,
            extract_depth=extract_depth,
            format=format,
            include_images=include_images,
            include_favicon=include_favicon,
            include_usage=include_usage,
            query=query,
            chunks_per_source=chunks_per_source,
        )
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    return ToolResult(
        success=True,
        content=result["summary"],
        state_updates=result.get("file_view_update", {})
    )



def exa_search_tool(query: str, num_results: int = 10,
                    type: str = None, category: str = None,
                    include_domains=None, exclude_domains=None,
                    start_published_date=None, end_published_date=None,
                    contents=None, **kwargs) -> ToolResult:
    """Search the web using Exa's neural/semantic search engine.

    Exa uses neural embeddings to match semantic intent rather than keywords.
    Supports 6 search types from instant (250ms) to deep-reasoning (40s),
    and category-specific indexes for companies, research papers, people, news.
    Raw JSON response is saved to a file and added to your file view windows.

    Use type='instant' or type='fast' for real-time applications.
    Use type='deep' or type='deep-reasoning' for multi-step research.
    Use category='research paper' or category='company' for domain-specific search.
    Use contents={"highlights": True} for token-efficient relevant extracts.
    """
    exa_search_service = kwargs["exa_search_service"]

    try:
        result = exa_search_service.search(
            query=query,
            num_results=num_results,
            type=type,
            category=category,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_published_date=start_published_date,
            end_published_date=end_published_date,
            contents=contents,
        )
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    return ToolResult(
        success=True,
        content=result["summary"],
        state_updates=result.get("file_view_update", {})
    )


def tinyfish_fetch_tool(urls, format: str = "markdown", links: bool = False,
                        image_links: bool = False, ttl: int = 3600,
                        per_url_timeout_ms: int = 45000, **kwargs) -> ToolResult:
    """Fetch and extract clean content from URLs using TinyFish's browser-rendered fetch API.

    Uses real full-browser rendering (headless Chromium) to handle JavaScript-heavy
    pages, dynamic content, and SPAs. Returns clean markdown, HTML, or JSON.
    Raw JSON response is saved to a file and added to your file view windows.

    This tool is free (150 URLs/min) and uses browser rendering, making it a reliable
    fallback for pages where other extraction tools fail. Failed URLs don't count
    against quota.

    Saves results to .fluximp/web/page/<domain>/<conversation_id>/tinyfish_fetch/.
    """
    tinyfish_fetch_service = kwargs["tinyfish_fetch_service"]

    try:
        result = tinyfish_fetch_service.fetch(
            urls=urls,
            format=format,
            links=links,
            image_links=image_links,
            ttl=ttl,
            per_url_timeout_ms=per_url_timeout_ms,
        )
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    return ToolResult(
        success=True,
        content=result["summary"],
        state_updates=result.get("file_view_update", {})
    )


def exa_contents_tool(urls, text=True, highlights=False, summary=False,
                      subpages=None, subpage_target=None, **kwargs) -> ToolResult:
    """Extract content from URLs using Exa's Contents API.

    Retrieves clean content from known URLs. Supports highlights (token-efficient
    relevant extracts, ~10x smaller than full text), full text, AI summaries,
    and subpage exploration (lightweight site crawling).

    Use highlights=True to scan many URLs efficiently before deciding which warrant full fetch.
    Use subpages=N to discover and fetch subpages of each URL.
    Raw JSON response is saved to a file and added to your file view windows.
    """
    exa_search_service = kwargs["exa_search_service"]

    try:
        result = exa_search_service.contents(
            urls=urls,
            text=text,
            highlights=highlights,
            summary=summary,
            subpages=subpages,
            subpage_target=subpage_target,
        )
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    return ToolResult(
        success=True,
        content=result["summary"],
        state_updates=result.get("file_view_update", {})
    )


def tinyfish_search_tool(query, domain_type="web", page=0,
                         recency_minutes=None, after_date=None,
                         before_date=None, location=None, language=None,
                         **kwargs) -> ToolResult:
    """Search the web using TinyFish's structured search API.

    Returns titles, snippets, and URLs designed for LLM consumption.
    Free (30 queries/min) with domain_type filters for web, news, and research papers.
    Supports search operators like site: and -site: in the query string.

    Use domain_type='research_paper' for academic results with authors and citations.
    Use domain_type='news' for news results with publisher and date.
    Raw JSON response is saved to a file and added to your file view windows.
    """
    tinyfish_fetch_service = kwargs["tinyfish_fetch_service"]

    try:
        result = tinyfish_fetch_service.search(
            query=query,
            domain_type=domain_type,
            page=page,
            recency_minutes=recency_minutes,
            after_date=after_date,
            before_date=before_date,
            location=location,
            language=language,
        )
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    return ToolResult(
        success=True,
        content=result["summary"],
        state_updates=result.get("file_view_update", {})
    )



def linkup_search_tool(query, depth="standard", output_type="searchResults",
                       structured_output_schema=None, include_images=None,
                       from_date=None, to_date=None, exclude_domains=None,
                       include_domains=None, max_results=None,
                       include_inline_citations=None, include_sources=None,
                       **kwargs) -> ToolResult:
    """Search the web using Linkup agentic search API.

    Offers 3 depth modes: fast (<1s, keyword only), standard (1-3s, agentic),
    and deep (5-30s, multi-iteration agentic search-and-scrape).
    Offers 3 output types: searchResults (ranked URLs), sourcedAnswer (answer+citations),
    and structured (JSON matching schema).

    Use depth=deep for complex research requiring multi-iteration search-and-scrape.
    Use output_type=sourcedAnswer for a direct answer with citations.
    Use output_type=structured with structured_output_schema for structured JSON output.
    Note: deep mode costs 10x more than standard ($0.05 vs $0.005 per call).
    Raw JSON response is saved to a file and added to your file view windows.
    """
    linkup_search_service = kwargs["linkup_search_service"]

    try:
        result = linkup_search_service.search(
            query=query,
            depth=depth,
            output_type=output_type,
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
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    return ToolResult(
        success=True,
        content=result["summary"],
        state_updates=result.get("file_view_update", {})
    )


def linkup_fetch_tool(url, render_js=True, include_raw_html=False,
                      extract_images=False, **kwargs) -> ToolResult:
    """Fetch and extract clean content from a URL using Linkup fetch API.

    Returns clean markdown with optional raw HTML and image extraction.
    Supports JavaScript rendering for dynamic pages (enabled by default).
    Accepts a single URL (not a list).

    Use render_js=true for JavaScript-heavy pages and SPAs.
    Use include_raw_html=true if you need both markdown and original HTML.
    Raw JSON response is saved to a file and added to your file view windows.
    """
    linkup_search_service = kwargs["linkup_search_service"]

    try:
        result = linkup_search_service.fetch(
            url=url,
            render_js=render_js,
            include_raw_html=include_raw_html,
            extract_images=extract_images,
        )
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    return ToolResult(
        success=True,
        content=result["summary"],
        state_updates=result.get("file_view_update", {})
    )

if False:
    def parallel_search_tool(search_queries, objective=None, mode=None,
                            max_chars_total=None, session_id=None, **kwargs) -> ToolResult:
        """Search using Parallel declarative semantic search. Provide keyword search_queries
        plus optional natural-language objective. Returns token-dense excerpts.
        Raw JSON response saved to file and added to file view windows."""
        parallel_search_service = kwargs["parallel_search_service"]
        try:
            result = parallel_search_service.search(search_queries=search_queries, objective=objective,
                mode=mode, max_chars_total=max_chars_total, session_id=session_id)
        except Exception as e:
            return ToolResult(success=False, content=f"Error: {str(e)}")
        return ToolResult(success=True, content=result["summary"], state_updates=result.get("file_view_update", {}))


    def parallel_extract_tool(urls, objective=None, search_queries=None,
                            full_content=False, max_chars_total=None,
                            session_id=None, **kwargs) -> ToolResult:
        """Extract content from URLs using Parallel objective-driven extraction.
        By default returns ranked excerpts; set full_content=True for full page markdown.
        Up to 20 URLs per request. Raw JSON response saved to file."""
        parallel_search_service = kwargs["parallel_search_service"]
        try:
            result = parallel_search_service.extract(urls=urls, objective=objective,
                search_queries=search_queries, full_content=full_content,
                max_chars_total=max_chars_total, session_id=session_id)
        except Exception as e:
            return ToolResult(success=False, content=f"Error: {str(e)}")
        return ToolResult(success=True, content=result["summary"], state_updates=result.get("file_view_update", {}))



def perplexity_search_tool(query, search_mode=None, search_type=None,
                           search_context_size=None, max_results=None,
                           search_domain_filter=None, search_recency_filter=None,
                           search_language_filter=None, country=None, **kwargs) -> ToolResult:
    """Search the web using Perplexity raw web search API.
    Returns ranked URLs with snippets. Supports academic papers, SEC filings,
    and people search modes. Use search_mode='academic' for academic papers,
    search_mode='sec' for SEC filings, or search_type='people' for people search.
    Use search_context_size to control content per result (low/medium/high).
    Raw JSON response saved to file and added to file view windows."""
    perplexity_search_service = kwargs["perplexity_search_service"]
    try:
        result = perplexity_search_service.search(query=query, search_mode=search_mode,
            search_type=search_type, search_context_size=search_context_size,
            max_results=max_results, search_domain_filter=search_domain_filter,
            search_recency_filter=search_recency_filter,
            search_language_filter=search_language_filter, country=country)
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")
    return ToolResult(success=True, content=result["summary"], state_updates=result.get("file_view_update", {}))


def firecrawl_search_tool(query: str, sources=None, categories=None,
                          limit: int = 5, tbs: str = None, location: str = None,
                          ignore_invalid_urls: bool = None,
                          timeout: int = None, **kwargs) -> ToolResult:
    """Search the web using Firecrawl's search API.

    Returns web, news, and image results from Firecrawl's index. Use
    'site:example.com' in the query to limit results to a domain. Results are
    saved to .fluximp/web/search/<workflow_id>/firecrawl_search/.

    Use sources=['web', 'news'] to search multiple sources in one call.
    Use categories=['research'] for academic results or ['pdf'] for PDFs.
    Raw JSON response is saved to a file and added to your file view windows.
    """
    firecrawl_search_service = kwargs["firecrawl_search_service"]

    try:
        result = firecrawl_search_service.search(
            query=query,
            sources=sources,
            categories=categories,
            limit=limit,
            tbs=tbs,
            location=location,
            ignore_invalid_urls=ignore_invalid_urls,
            timeout=timeout,
        )
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    return ToolResult(
        success=True,
        content=result["summary"],
        state_updates=result.get("file_view_update", {})
    )


def firecrawl_scrape_tool(url, formats=None, only_main_content=None,
                          wait_for: int = None, max_age: int = None,
                          proxy: str = None, timeout: int = None,
                          **kwargs) -> ToolResult:
    """Scrape a single URL using Firecrawl's scrape API.

    Returns clean markdown (default) plus any other requested formats.
    Handles JavaScript-rendered pages, PDFs, and anti-bot protection
    server-side. Accepts a single URL (not a list). Results are saved to
    .fluximp/web/page/<domain>/<workflow_id>/firecrawl_scrape/.

    Use formats=['markdown', 'links'] to also get page links.
    Use only_main_content=true to strip nav and footer boilerplate.
    Raw JSON response is saved to a file and added to your file view windows.
    """
    firecrawl_search_service = kwargs["firecrawl_search_service"]

    try:
        result = firecrawl_search_service.scrape(
            url=url,
            formats=formats,
            only_main_content=only_main_content,
            wait_for=wait_for,
            max_age=max_age,
            proxy=proxy,
            timeout=timeout,
        )
    except Exception as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    return ToolResult(
        success=True,
        content=result["summary"],
        state_updates=result.get("file_view_update", {})
    )
