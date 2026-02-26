"""LangSmith API Client.

HTTP client for LangSmith API with retry logic and error handling.
"""

import logging
from typing import Any

import httpx
from mcp_common.exceptions import MCPServerError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langsmith_mcp.config import LangSmithSettings

logger = logging.getLogger(__name__)


class LangSmithAPIError(MCPServerError):
    """LangSmith API-specific error."""

    def __init__(self, message: str, status_code: int | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class LangSmithClient:
    """Async HTTP client for LangSmith API.

    Handles authentication, retries, and error handling for all LangSmith operations.
    """

    def __init__(self, settings: LangSmithSettings):
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "LangSmithClient":
        """Initialize async context."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup async context."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.api_endpoint,
                timeout=httpx.Timeout(self.settings.http_timeout),
                headers={
                    "x-api-key": self.settings.api_key,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            logger.info(
                "LangSmith client initialized",
                extra={"endpoint": self.settings.api_endpoint},
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("LangSmith client closed")

    def _get_client(self) -> httpx.AsyncClient:
        """Get the HTTP client, initializing if needed."""
        if self._client is None:
            raise MCPServerError("LangSmith client not initialized. Call initialize() first.")
        return self._client

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path (e.g., "/v1/prompts")
            params: Query parameters
            json_data: JSON body for POST/PUT requests

        Returns:
            Parsed JSON response

        Raises:
            LangSmithAPIError: On API errors
            MCPError: On unexpected errors
        """
        client = self._get_client()

        try:
            response = await client.request(
                method=method,
                url=path,
                params=params,
                json=json_data,
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            error_body = {}
            try:
                error_body = e.response.json()
            except Exception:
                error_body = {"error": e.response.text}

            logger.error(
                "LangSmith API error",
                extra={
                    "status_code": e.response.status_code,
                    "path": path,
                    "error": error_body,
                },
            )

            raise LangSmithAPIError(
                message=f"LangSmith API error: {error_body.get('error', str(e))}",
                status_code=e.response.status_code,
                details=error_body,
            ) from e

        except httpx.RequestError as e:
            logger.error(
                "LangSmith request error",
                extra={"path": path, "error": str(e)},
            )
            raise MCPServerError(f"Request failed: {e}") from e

        except Exception as e:
            logger.exception("Unexpected error in LangSmith request")
            raise MCPServerError(f"Unexpected error: {e}") from e

    # =====================
    # Conversation History
    # =====================

    async def get_thread_history(
        self,
        thread_id: str,
        project_name: str,
        page_number: int = 1,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve message history for a conversation thread.

        Args:
            thread_id: The thread ID to retrieve
            project_name: Project name containing the thread
            page_number: Page number for pagination
            max_chars: Maximum characters per page (uses settings default if None)

        Returns:
            Thread history with messages
        """
        max_chars = max_chars or self.settings.max_chars_per_page
        return await self._request(
            "GET",
            f"/v1/threads/{thread_id}/history",
            params={
                "project_name": project_name,
                "page_number": page_number,
                "max_chars_per_page": max_chars,
            },
        )

    # =====================
    # Prompts
    # =====================

    async def list_prompts(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List all prompts in the workspace.

        Args:
            limit: Maximum number of prompts to return
            offset: Offset for pagination

        Returns:
            List of prompts with metadata
        """
        return await self._request(
            "GET",
            "/v1/prompts",
            params={"limit": limit, "offset": offset},
        )

    async def get_prompt(
        self,
        prompt_identifier: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Get a specific prompt by identifier.

        Args:
            prompt_identifier: Prompt name or ID
            version: Optional specific version (latest if None)

        Returns:
            Prompt details with content
        """
        params = {}
        if version:
            params["version"] = version

        return await self._request(
            "GET",
            f"/v1/prompts/{prompt_identifier}",
            params=params,
        )

    async def push_prompt(
        self,
        prompt_identifier: str,
        prompt_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Push a new prompt version.

        Args:
            prompt_identifier: Prompt name or ID
            prompt_data: Prompt content and metadata

        Returns:
            Created prompt version details
        """
        return await self._request(
            "POST",
            f"/v1/prompts/{prompt_identifier}",
            json_data=prompt_data,
        )

    # =====================
    # Traces & Runs
    # =====================

    async def fetch_runs(
        self,
        project_id: str | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Fetch runs/traces from LangSmith.

        Args:
            project_id: Filter by project ID
            trace_id: Filter by trace ID
            run_id: Get specific run by ID
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Runs/traces data
        """
        params = {"limit": limit, "offset": offset}
        if project_id:
            params["project_id"] = project_id
        if trace_id:
            params["trace_id"] = trace_id
        if run_id:
            params["id"] = run_id

        return await self._request("GET", "/v1/runs", params=params)

    async def list_projects(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List all projects in the workspace.

        Args:
            limit: Maximum projects to return
            offset: Pagination offset

        Returns:
            List of projects
        """
        return await self._request(
            "GET",
            "/v1/projects",
            params={"limit": limit, "offset": offset},
        )

    # =====================
    # Datasets & Examples
    # =====================

    async def list_datasets(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List all datasets in the workspace.

        Args:
            limit: Maximum datasets to return
            offset: Pagination offset

        Returns:
            List of datasets
        """
        return await self._request(
            "GET",
            "/v1/datasets",
            params={"limit": limit, "offset": offset},
        )

    async def get_dataset(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:
        """Get a specific dataset by ID.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dataset details
        """
        return await self._request("GET", f"/v1/datasets/{dataset_id}")

    async def list_examples(
        self,
        dataset_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List examples in a dataset.

        Args:
            dataset_id: Dataset ID
            limit: Maximum examples to return
            offset: Pagination offset

        Returns:
            List of examples
        """
        return await self._request(
            "GET",
            f"/v1/datasets/{dataset_id}/examples",
            params={"limit": limit, "offset": offset},
        )

    async def create_dataset(
        self,
        name: str,
        description: str | None = None,
        data_type: str = "kv",
    ) -> dict[str, Any]:
        """Create a new dataset.

        Args:
            name: Dataset name
            description: Optional description
            data_type: Data type (kv, chat, etc.)

        Returns:
            Created dataset details
        """
        return await self._request(
            "POST",
            "/v1/datasets",
            json_data={
                "name": name,
                "description": description,
                "data_type": data_type,
            },
        )

    async def create_examples(
        self,
        dataset_id: str,
        examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Add examples to a dataset.

        Args:
            dataset_id: Dataset ID
            examples: List of example data

        Returns:
            Created examples
        """
        return await self._request(
            "POST",
            f"/v1/datasets/{dataset_id}/examples",
            json_data={"examples": examples},
        )

    # =====================
    # Experiments
    # =====================

    async def list_experiments(
        self,
        dataset_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List experiments.

        Args:
            dataset_id: Filter by dataset ID
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of experiments
        """
        params = {"limit": limit, "offset": offset}
        if dataset_id:
            params["dataset_id"] = dataset_id

        return await self._request("GET", "/v1/experiments", params=params)

    async def get_experiment(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        """Get a specific experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment details with results
        """
        return await self._request("GET", f"/v1/experiments/{experiment_id}")

    # =====================
    # Billing & Usage
    # =====================

    async def get_billing_usage(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Get billing and usage information.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            Billing and usage data
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return await self._request("GET", "/v1/billing/usage", params=params)
