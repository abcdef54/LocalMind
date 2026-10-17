"""Check configurable match sizes without changing generated candidate pools."""

import math
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from src.backend.ai.langgraph.graph_builder import LocalMindAI
from src.backend.ai.langgraph.graph_nodes import _tournament_groups, tournament_rank
from src.backend.ai.langgraph.graph_states import Candidate
from src.backend.ai.langgraph.graph_configs import validate_ranking_group_size


class ConfigurableTournamentTests(unittest.IsolatedAsyncioTestCase):
    """Exercise uneven pools, byes and K-independent reasoning widths."""

    async def test_every_supported_size_finishes_without_losing_or_duplicating_candidates(self):
        """All pools reduce to one winner and each real match stays within its configured K."""
        for count in range(1, 18):
            for size in range(2, 8):
                pool = [Candidate(answer=f"answer-{index}") for index in range(count)]
                before = [candidate.id for candidate in pool]

                async def choose_first(query, candidates, model, idx):
                    """Select one actual group member without making a model call."""
                    self.assertGreaterEqual(len(candidates), 2)
                    self.assertLessEqual(len(candidates), size)
                    return candidates[0], 1, False

                ranker = AsyncMock(side_effect=choose_first)
                with patch("src.backend.ai.langgraph.graph_nodes.rank_no_reasoning", ranker), patch(
                    "src.backend.ai.langgraph.graph_nodes.random.shuffle",
                    side_effect=lambda values: values.reverse(),
                ), patch("builtins.print"):
                    winner, calls, failed, rounds, maximum = await tournament_rank(
                        "question", pool, object(), group_size=size
                    )
                self.assertIn(winner.id, before)
                self.assertEqual([candidate.id for candidate in pool], before)
                self.assertEqual(calls, ranker.await_count)
                self.assertFalse(failed)
                self.assertLessEqual(maximum, size)
                self.assertEqual(calls, math.ceil((count - 1) / (size - 1)))

    async def test_default_groups_preserve_existing_three_way_brackets(self):
        """The default grouping retains full groups of three and singleton byes."""
        for count, expected in ((1, [1]), (3, [3]), (5, [3, 1, 1]), (7, [3, 3, 1]), (9, [3, 3, 3])):
            pool = [Candidate() for _ in range(count)]
            groups = _tournament_groups(pool)
            self.assertEqual([len(group) for group in groups], expected)
            self.assertEqual(
                [candidate.id for group in groups for candidate in group],
                [candidate.id for candidate in pool],
            )

    async def test_high_mode_changes_matches_without_changing_five_generated_candidates(self):
        """K=2,3,5,7 uses 4,2,1,1 ranking calls while N stays five."""
        for size, expected_calls in ((2, 4), (3, 2), (5, 1), (7, 1)):
            model = AsyncMock()
            model.ainvoke.return_value = AIMessage(content="candidate")
            ranker = AsyncMock()
            ranker.ainvoke.return_value = AIMessage(content="A")
            model.bind = lambda **kwargs: ranker
            with patch("src.backend.ai.langgraph.graph_nodes.get_llm", return_value=model), patch(
                "src.backend.ai.langgraph.graph_nodes._seeded",
                side_effect=lambda model, seed: model,
            ), patch("src.backend.ai.langgraph.graph_nodes._log_token_usage"), patch(
                "builtins.print"
            ):
                result = await LocalMindAI().chat(
                    [{"role": "user", "content": "question"}],
                    "fake",
                    "general",
                    "high",
                    0.7,
                    tools_enabled=False,
                    ranking_group_size=size,
                )
            self.assertEqual(model.ainvoke.await_count, 5)
            self.assertEqual(ranker.ainvoke.await_count, expected_calls)
            self.assertEqual(result["generation_metadata"]["ranking_group_size"], size)
            self.assertEqual(result["generation_metadata"]["candidate_count"], 5)

    async def test_invalid_group_size_is_rejected_before_model_generation(self):
        """Python entry points reject the same invalid K values as the HTTP schema."""
        ai = LocalMindAI()
        ai._graph = AsyncMock()
        for size in (True, False, 0, 1, 8, 2.5, "3", None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                await ai.chat(
                    [], "fake", "general", "low", 0, tools_enabled=False, ranking_group_size=size
                )
        ai._graph.ainvoke.assert_not_awaited()
