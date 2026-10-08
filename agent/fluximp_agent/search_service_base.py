"""Base search service for search provider integrations.

Provides shared utilities for saving raw API responses and metadata
sidecar files, and for building file view window state updates.
Each provider subclass handles its own API client and method signatures.
"""

import json
import dataclasses
import keyring
from pathlib import Path
from datetime import datetime, timezone
import urllib.parse


class BaseSearchService:
    """Base class for search provider services.

    Provides shared utilities for saving raw API responses and metadata
    sidecar files, and for building file view window state updates.
    Each provider subclass handles its own API client and method signatures.
    """

    def __init__(self, filesystem_service, state_service, experiments_service, io_path):
        """
        Args:
            filesystem_service: FileSystemService for path validation and writing
            state_service: StateService for building file view window updates
            experiments_service: ExperimentsService for checking experiment flags
            io_path: Path to the I/O directory (conversation_dir / "io")
        """
        self._filesystem_service = filesystem_service
        self._state_service = state_service
        self._experiments_service = experiments_service
        self._io_path = io_path
        # Extract conversation directory name as workflow_id
        self._workflow_id = io_path.parent.name

    def _get_api_key(self, provider_domain: str) -> str:
        """Get API key from keyring for the given provider domain.

        Args:
            provider_domain: The provider's domain name (e.g., "tavily.com")

        Returns:
            The API key string.

        Raises:
            RuntimeError: If the API key is not found in keyring.
        """
        api_key = keyring.get_password(provider_domain, "api_key")
        if not api_key:
            raise RuntimeError(
                f"API key not found in keyring for '{provider_domain}'. "
                f"Set it with: python -m keyring set {provider_domain} api_key"
            )
        return api_key

    @staticmethod
    def _first_url_domain(urls_or_url) -> str:
        """Return the netloc of the first URL, or 'unknown' if none can be parsed.

        Accepts a single URL string or a list/sequence of URL strings.
        """
        if isinstance(urls_or_url, (list, tuple)):
            first = urls_or_url[0] if urls_or_url else ""
        else:
            first = urls_or_url or ""

        parsed = urllib.parse.urlparse(first)
        return parsed.netloc or "unknown"

    def _save_result(self, tool_name: str, raw_content: str, content_type: str,
                     query: str, query_params: dict,
                     category: str = "search", subdir: str = None) -> tuple:
        """Save raw API response and metadata sidecar to .fluximp/web/<category>/.

        File path format: .fluximp/web/<category>/[<subdir>/]<workflow_id>/<tool_name>/YYYYMMDD_HHMMSSffffff.<ext>
        Metadata sidecar:  YYYYMMDD_HHMMSSffffff.metadata.json at same directory

        Args:
            tool_name: Name of the tool (e.g., "tavily_search", "tavily_extract")
            raw_content: Raw response content as a string
            content_type: MIME type (e.g., "application/json")
            query: The search query or URL
            query_params: Dict of all query parameters
            category: Subdirectory under .fluximp/web/ (e.g., "search", "page")
            subdir: Optional subdirectory between category and workflow_id
                    (e.g., domain name for page retrievals)

        Returns:
            Tuple of (result_file_path, metadata_file_path) as relative path strings
        """
        # Generate timestamp: YYYYMMDD_HHMMSSffffff (microseconds)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S%f")

        # Map content type to extension
        ext = self._content_type_to_extension(content_type)

        # Build relative paths
        path_parts = [".fluximp", "web", category]
        if subdir:
            subdir = self._sanitize_path_component(subdir)
            path_parts.append(subdir)
        path_parts.extend([self._workflow_id, tool_name])
        base_dir = Path(*path_parts)
        result_filename = f"{timestamp}.{ext}"
        metadata_filename = f"{timestamp}.metadata.json"

        result_path_str = str(base_dir / result_filename)
        metadata_path_str = str(base_dir / metadata_filename)

        # Validate paths via filesystem_service
        validated_result_path, error = self._filesystem_service.validate_path(result_path_str)
        if error:
            raise RuntimeError(f"Path validation failed: {error}")

        validated_metadata_path, _ = self._filesystem_service.validate_path(metadata_path_str)

        # Create directories
        validated_result_path.parent.mkdir(parents=True, exist_ok=True)

        # Write raw result
        validated_result_path.write_text(raw_content, encoding="utf-8", newline="")

        # Write metadata sidecar
        metadata = {
            "query": query,
            "tool_name": tool_name,
            "provider": self._provider_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_params": query_params,
            "content_type": content_type,
            "result_file": result_filename,
        }
        validated_metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8", newline=""
        )

        return result_path_str, metadata_path_str

    def _content_type_to_extension(self, content_type: str) -> str:
        """Map MIME type to file extension."""
        mapping = {
            "application/json": "json",
            "text/markdown": "md",
            "text/html": "html",
            "text/plain": "txt",
            "application/xml": "xml",
        }
        # Strip parameters like "; charset=utf-8"
        ct = content_type.split(";")[0].strip().lower()
        return mapping.get(ct, "txt")

    def _build_file_view_update(self, result_file_path: str) -> dict:
        """Build file view window state update for the result file.

        Only active when file_view_windows experiment is enabled.
        Limits to first 2500 lines.

        Returns:
            State update dict (empty if experiment not active)
        """
        if (not self._experiments_service or
                not self._experiments_service.is_enabled("file_view_windows") or
                not self._state_service):
            return {}

        from fluximp_agent.state_service import FileViewRequest
        return self._state_service.build_file_view_windows_update(
            [FileViewRequest(file_path=result_file_path, offset=1, limit=2500)],
            source="edit"

        )

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        """Sanitize a string for safe use as a single filesystem directory name.

        Replaces any character that is not alphanumeric, dot, or hyphen
        with underscore. Prevents path traversal via backslashes, forward
        slashes, or other special characters.

        Args:
            value: Raw string (e.g., domain name from URL parsing)

        Returns:
            Sanitized string safe to use as a directory name.
            Returns "unknown" if the result would be empty, ".", or "..".
        """
        import re
        # Replace anything that is not alphanumeric, dot, or hyphen
        sanitized = re.sub(r'[^a-zA-Z0-9.\-]', '_', value)
        # Truncate to reasonable length
        sanitized = sanitized[:255]
        # Reject dangerous results
        if not sanitized or sanitized == '.' or sanitized == '..':
            return "unknown"
        return sanitized

    @staticmethod
    def _serialize_response(response) -> str:
        """Serialize API response to JSON string.

        Handles pydantic BaseModel (via model_dump), dataclasses (via asdict),
        plain dicts, and other objects (via json.dumps with str fallback).
        """
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif isinstance(response, dict):
            data = response
        elif dataclasses.is_dataclass(response):
            data = dataclasses.asdict(response)
        else:
            data = response
        return json.dumps(data, indent=2, default=str)

    def _build_result(self, result_path: str, metadata_path: str,
                      summary: str, **extra) -> dict:
        """Build the standard result dict returned by all search/extract methods.

        Args:
            result_path: Relative path to the saved raw result file
            metadata_path: Relative path to the metadata sidecar file
            summary: Concise summary string for the agent
            **extra: Additional keys to include (e.g., result_count, failed_count)

        Returns:
            Dict with standard keys plus any extra keys.
        """
        result = {
            "result_file_path": result_path,
            "metadata_file_path": metadata_path,
            "summary": summary,
            "file_view_update": self._build_file_view_update(result_path),
        }
        result.update(extra)
        return result

    @staticmethod
    def _build_kwargs(required: dict = None, **optional) -> dict:
        """Build kwargs dict for SDK calls, including non-None optional values.

        Args:
            required: Dict of parameters that should always be included
            **optional: Parameters included only if their value is not None

        Returns:
            Combined kwargs dict
        """
        kwargs = dict(required or {})
        kwargs.update({k: v for k, v in optional.items() if v is not None})
        return kwargs

    @staticmethod
    def _ensure_list(value):
        """Ensure value is a list. If it's a string, wrap it in a list."""
        if isinstance(value, str):
            return [value]
        return value

    @staticmethod
    def _format_summary(parts: list, result_path: str) -> str:
        """Join summary parts and append the results-saved-to footer.

        Args:
            parts: List of summary line strings (header, query, results, etc.)
            result_path: Path to saved results file (appended as last line)

        Returns:
            Newline-joined summary string
        """
        return "\n".join(parts + [f"Results saved to: {result_path}"])
