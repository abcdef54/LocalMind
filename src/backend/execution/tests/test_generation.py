"""Check graph failures and candidate concurrency without models, GPU or a database."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from src.backend.ai.errors import InferenceStateUnknown
from src.backend.ai.langgraph.graph_builder import LocalMindAI
from src.backend.ai.langgraph.graph_nodes import generate, _generate_sample_with_tools
from src.backend.ai.langgraph.graph_utils import invoke_model, _build_llm


class GenerationFailureTests(unittest.IsolatedAsyncioTestCase):
    """Reject fake answers and retain uncertainty until every candidate has finished."""

    async def test_waits_for_sibling_before_raising_and_prioritizes_uncertainty(self):
        """A fast failure cannot release the run while another candidate still computes."""
        sibling_started = asyncio.Event()
        sibling_finished = asyncio.Event()
        calls = 0

        async def sample(*args):
            """Fail one request while keeping the second active until the test releases it."""
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("ordinary rejection")
            sibling_started.set()
            await sibling_finished.wait()
            raise InferenceStateUnknown()

        with patch("src.backend.ai.langgraph.graph_nodes.get_llm", return_value=object()), patch(
            "src.backend.ai.langgraph.graph_nodes._seeded", side_effect=lambda model, seed: model
        ), patch(
            "src.backend.ai.langgraph.graph_nodes._generate_sample_with_tools", side_effect=sample
        ):
            task = asyncio.create_task(
                generate(
                    {
                        "width": 2,
                        "messages": [HumanMessage(content="question")],
                        "user_query": "question",
                    },
                    {"configurable": {"tools_enabled": False}},
                )
            )
            await sibling_started.wait()
            self.assertFalse(task.done())
            sibling_finished.set()
            with self.assertRaises(InferenceStateUnknown):
                await task

    async def test_empty_answer_and_tool_iteration_exhaustion_are_failures(self):
        """Never store a tool result or blank model output as the assistant's final answer."""
        model = AsyncMock()
        model.ainvoke.return_value = AIMessage(content=" ")
        with self.assertRaises(ValueError):
            await _generate_sample_with_tools(model, [], {})
        model.ainvoke.return_value = AIMessage(
            content="", tool_calls=[{"name": "missing", "args": {}, "id": "call-1"}]
        )
        with self.assertRaises(ValueError):
            await _generate_sample_with_tools(model, [], {})
        ai = LocalMindAI()
        ai._graph = AsyncMock()
        ai._graph.ainvoke.return_value = {}
        with self.assertRaises(ValueError):
            await ai.chat([], "fake", "general", "low", 0, tools_enabled=False)

    async def test_transport_failure_is_uncertain_and_requests_do_not_retry(self):
        """Prevent silent retries when a timeout may leave remote generation active."""
        model = AsyncMock()
        model.ainvoke.side_effect = httpx.ReadTimeout("private address")
        with self.assertRaises(InferenceStateUnknown):
            await invoke_model(model, [])
        client = _build_llm("fake", "http://127.0.0.1:1/v1", 0.0)
        self.assertEqual(client.max_retries, 0)
        model.ainvoke.assert_awaited_once()

    async def test_removed_direct_chat_endpoint_cannot_bypass_executor(self):
        """The root app must not expose the old synchronous generation route."""
        from src.backend.app import app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            self.assertEqual((await client.post("/chat", json={})).status_code, 404)

    async def test_high_mode_keeps_five_candidates_and_two_ranking_calls(self):
        """Preserve the research tournament behavior while exposing compact metrics."""
        model = AsyncMock()
        model.ainvoke.return_value = AIMessage(content="candidate answer")
        ranker = AsyncMock()
        ranker.ainvoke.return_value = AIMessage(content="A")
        model.bind = lambda **kwargs: ranker
        with patch("src.backend.ai.langgraph.graph_nodes.get_llm", return_value=model), patch(
            "src.backend.ai.langgraph.graph_nodes._seeded", side_effect=lambda model, seed: model
        ), patch("src.backend.ai.langgraph.graph_nodes._log_token_usage"):
            result = await LocalMindAI().chat(
                [{"role": "user", "content": "question"}],
                "fake",
                "general",
                "high",
                0.7,
                tools_enabled=False,
            )
        self.assertEqual(result["generation_metadata"]["candidate_count"], 5)
        self.assertEqual(result["generation_metadata"]["tournament_rank_calls"], 2)
        self.assertEqual(model.ainvoke.await_count, 5)
        self.assertEqual(ranker.ainvoke.await_count, 2)
