import os
import dotenv
import httpx

from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool
from langchain_community.tools import TavilySearchResults
from pydantic import BaseModel, Field
import logging
from uuid import UUID

logger = logging.getLogger("uvicorn.error")

dotenv.load_dotenv()


class RetrieveContextInput(BaseModel):
    """Accept search text/count only; conversation scope is supplied by the backend."""

    model_config = {"extra": "forbid"}
    query: str = Field(
        min_length=1, max_length=8000, description="Text to find in this chat's ready documents."
    )
    k_returns: int = Field(default=4, ge=1, le=20, description="Maximum relevant chunks to return.")


def create_retrieval_tool(conversation_id: UUID) -> BaseTool:
    """Bind document search to one server-selected conversation before exposing it to the model.

    Args:
        conversation_id: Chat UUID from the validated request/run context, never
            from model-generated arguments. A missing/invalid scope is rejected.

    Returns:
        LangChain tool accepting only query/k_returns and returning context plus
        citation IDs. Backend errors return a generic message, never an unscoped search.

    Raises:
        ValueError: If conversation_id is not a UUID.
    """
    if not isinstance(conversation_id, UUID):
        raise ValueError("Document retrieval requires a conversation UUID")

    @tool(
        "retrieve_context", args_schema=RetrieveContextInput, response_format="content_and_artifact"
    )
    async def retrieve_context(query: str, k_returns: int = 4) -> tuple[str, dict]:
        """Search ready documents uploaded in this conversation for relevant context.

        Args:
            query: Text to match against ready document chunks in the bound chat.
            k_returns: Maximum chunks to return, from one to twenty.

        Returns:
            Context text and a citation artifact containing document UUIDs and
            original filenames. Return empty context when no ready documents match.
        """
        from src.backend.database.session import async_session_maker
        from src.backend.indexing.resources import get_document_vector_index
        from src.backend.indexing.retrieval import ConversationRetriever

        try:
            retriever = ConversationRetriever(async_session_maker, get_document_vector_index())
            documents = await retriever.search(conversation_id, query, limit=k_returns)
            source_ids = list(
                dict.fromkeys(document.metadata["source_file_id"] for document in documents)
            )
            context = "\n\n".join(
                f"Source: {document.metadata['source']}\nContent: {document.page_content}"
                for document in documents
            )
            return context, {
                "source_file_ids": source_ids,
                "raw_documents": [
                    {"source": document.metadata["source"], "content": document.page_content}
                    for document in documents
                ],
            }
        except Exception as error:
            logger.error(
                "Document retrieval failed conversation_id=%s exception_type=%s",
                conversation_id,
                type(error).__name__,
            )
            return "Document search is temporarily unavailable.", {}

    return retrieve_context


@tool
async def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Searches the internet for the given query and returns live web results
    including titles, text snippets, and source links.

    Use a higher 'max_results' (up to 10) if the query requires deep, comprehensive
    research or cross-referencing multiple viewpoints.
    """
    tavily_key = os.getenv("TAVILY_KEY")
    if not tavily_key:
        print("[Search Tool Warning]: TAVILY_API_KEY is missing from environment variables.")
        return "Error: Web search engine is currently unconfigured. Missing API credentials."

    try:
        # FIX: Explicitly passing the key into the tool constructor so it actually gets used!
        search_tool = TavilySearchResults(
            tavily_api_key=tavily_key, max_results=min(max(max_results, 1), 10)
        )

        results = await search_tool.ainvoke({"query": query})

        if not results:
            print(f"No results found for query: '{query}'")
            return f"Search completed, but no relevant results were found for: '{query}'"

        formatted_results = []
        for i, res in enumerate(results, start=1):
            title = res.get("title", f"Web Result {i}")
            snippet = res.get("content", "(No snippet text available)")
            link = res.get("url", "(No link available)")

            result_block = f"Title: {title}\nSnippet: {snippet}\nLink: {link}"
            formatted_results.append(result_block)

        final_output = "\n\n---\n\n".join(formatted_results)
        print(f"[Search Success] Captured {len(formatted_results)} results for query: '{query}'")

        return final_output

    except Exception as e:
        error_msg = f"An error occurred while executing the web search: {str(e)}"
        print(f"[Search Failure] {error_msg}")
        return error_msg


@tool
async def fetch_web_page(url: str) -> str:
    """
    Navigates to a specific URL link, downloads the webpage content,
    strips away layout clutter, and extracts the core readable text body.
    Useful when you need to read a full article, documentation page, or source link.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            html = response.text

        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style", "nav", "footer", "header", "form", "aside"]):
            element.decompose()

        main_content = soup.find("article") or soup.find("main") or soup.find("body")

        if not main_content:
            return "The webpage loaded successfully but contained no extractable textual content."

        chunks = (
            phrase.strip()
            for line in main_content.get_text().splitlines()
            for phrase in line.split("  ")
        )
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        if not clean_text:
            return "The webpage loaded successfully but contained no extractable textual content."

        MAX_CHARS = 12000
        if len(clean_text) > MAX_CHARS:
            cutoff = clean_text.rfind("\n", 0, MAX_CHARS)
            cutoff = cutoff if cutoff != -1 else MAX_CHARS
            return (
                f"[WARNING: Content truncated for size constraints]\n\n"
                f"{clean_text[:cutoff]}\n\n"
                f"... [Truncated remaining {len(clean_text) - cutoff} characters]"
            )

        return clean_text

    except httpx.TimeoutException:
        return "Error: The target server took too long to respond. Connection timed out."
    except httpx.HTTPStatusError as e:
        return f"HTTP error {e.response.status_code} returned while fetching page: {url}"
    except Exception as e:
        return f"An error occurred while attempting to parse the page link: {str(e)}"
