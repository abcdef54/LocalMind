"""Verify model tools cannot choose conversation scope and graph calls preserve citations."""

import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from src.backend.ai.langchain.tools import create_retrieval_tool
from src.backend.ai.langgraph.graph_builder import LocalMindAI
from src.backend.ai.langgraph.graph_nodes import _generate_sample_with_tools


class ScopedToolTests(unittest.IsolatedAsyncioTestCase):
    """Exercise actual tool validation and graph tool execution with fake retrieval/model calls."""

    async def test_tool_exposes_query_only_and_rejects_model_selected_scope(self):
        """Keep the backend chat in a closure rather than accepting conversation IDs from the LLM."""
        conversation_id = uuid4()
        file_id = str(uuid4())
        tool = create_retrieval_tool(conversation_id)
        self.assertEqual(set(tool.args_schema.model_fields), {"query", "k_returns"})
        with self.assertRaises(ValueError):
            create_retrieval_tool(None)
        retriever = AsyncMock()
        retriever.search.return_value = [
            Document(
                page_content="safe own text",
                metadata={
                    "source_file_id": file_id,
                    "source": "own.txt",
                },
            )
        ]
        with patch(
            "src.backend.indexing.resources.get_document_vector_index", return_value=object()
        ), patch("src.backend.indexing.retrieval.ConversationRetriever", return_value=retriever):
            output = await tool.ainvoke(
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "retrieve_context",
                    "args": {"query": "alpha", "k_returns": 2},
                }
            )
            retriever.search.assert_awaited_once_with(conversation_id, "alpha", limit=2)
            self.assertEqual(output.artifact["source_file_ids"], [file_id])
            with self.assertRaises(ValidationError):
                await tool.ainvoke({"query": "alpha", "conversation_id": str(uuid4())})

    async def test_graph_uses_validated_tool_message_and_keeps_citation_artifact(self):
        """Execute the tool through LangChain instead of bypassing schema validation via coroutine."""
        conversation_id = uuid4()
        file_id = str(uuid4())
        tool = create_retrieval_tool(conversation_id)
        retriever = AsyncMock()
        retriever.search.return_value = [
            Document(
                page_content="alpha",
                metadata={
                    "source_file_id": file_id,
                    "source": "own.txt",
                },
            )
        ]
        model = AsyncMock()
        model.bind_tools = lambda tools: model
        model.ainvoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call-1", "name": "retrieve_context", "args": {"query": "alpha"}}
                ],
            ),
            AIMessage(content="answer"),
        ]
        with patch(
            "src.backend.indexing.resources.get_document_vector_index", return_value=object()
        ), patch("src.backend.indexing.retrieval.ConversationRetriever", return_value=retriever):
            messages, citations = await _generate_sample_with_tools(
                model, [HumanMessage(content="question")], {"retrieve_context": tool}
            )
        self.assertEqual(citations, [file_id])
        self.assertEqual(messages[1].artifact["source_file_ids"], [file_id])
        self.assertEqual(messages[-1].content, "answer")

    async def test_ai_requires_and_forwards_backend_conversation_context(self):
        """Reject tool-enabled calls without scope and pass a validated UUID in graph config."""
        ai = LocalMindAI()
        ai._graph = AsyncMock()
        ai._graph.ainvoke.return_value = {"final_answer": "fake", "citations": []}
        with self.assertRaises(ValueError):
            await ai.chat([], "fake", "general", "low", 0.0)
        conversation_id = uuid4()
        await ai.chat([], "fake", "general", "low", 0.0, conversation_id=conversation_id)
        _, config = ai._graph.ainvoke.call_args.args
        self.assertEqual(config["configurable"]["conversation_id"], str(conversation_id))

    async def test_old_upload_and_unsafe_delete_routes_are_removed(self):
        """Keep the old upload-on-drop path absent instead of retaining a parallel indexer."""
        from src.backend.app import app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            self.assertEqual((await client.post("/uploads")).status_code, 404)
            self.assertEqual((await client.get("/uploads")).status_code, 404)
            self.assertEqual((await client.delete(f"/conversation/{uuid4()}")).status_code, 405)
