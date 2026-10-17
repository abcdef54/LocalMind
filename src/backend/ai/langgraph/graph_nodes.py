import asyncio
import random
import re
import time
from uuid import UUID
from typing import Optional
from langchain_core.runnables import RunnableConfig
from langchain.messages import HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from src.backend.ai.langgraph.graph_states import AgentState, Candidate
from src.backend.ai.langgraph.graph_configs import get_config, MAX_RANK_GROUP
from src.backend.ai.langgraph.graph_utils import (
    get_llm,
    _seeded,
    _sample_idx,
    _log_token_usage,
    invoke_model,
)
from src.backend.ai.errors import InferenceStateUnknown
from src.backend.ai.langchain.tools import create_retrieval_tool, fetch_web_page, tavily_search


def setup(state: AgentState, config: RunnableConfig) -> dict:
    """Pure-Python entry (NO llm call). Reads mode + governor settings, seeds the run state.
    RETURNS: user_query, reasoning_mode, width, pool."""
    cfg = config["configurable"]
    mode = cfg["reasoning_mode"]
    loop_cfg = get_config(mode)

    return {
        "user_query": state["messages"][-1].content,
        "reasoning_mode": mode,
        "width": loop_cfg,
        "pool": [],
    }


async def _generate_sample_with_tools(llm, messages: list, tools_map: dict) -> tuple[list, list]:
    """Generate one candidate using validated LangChain tool calls and scoped retrieval.

    Args:
        llm: Candidate's model client, optionally bound to the supplied tools.
        messages: Conversation context copied before adding generated/tool messages.
        tools_map: Server-built tool instances; retrieve_context must already be
            bound to the chat, so model arguments cannot select another conversation.

    Returns:
        Produced messages and unique cited file IDs after at most five model rounds.
        ToolMessage artifacts are preserved instead of bypassing tool validation.

    Raises:
        ValueError: A model reply is blank/unsupported or five rounds end without
            a final answer. Tool output must never be saved as the assistant answer.
        InferenceStateUnknown: A model request may still be running remotely.
    """
    bound_llm = llm.bind_tools(list(tools_map.values())) if tools_map else llm
    current_messages = list(messages)
    messages_produced = []
    citations_extracted = []
    max_iterations = 5

    for _ in range(max_iterations):
        response = await invoke_model(bound_llm, current_messages)
        messages_produced.append(response)
        current_messages.append(response)

        if not response.tool_calls:
            if not isinstance(response.content, str) or not response.content.strip():
                raise ValueError("Model returned an empty or unsupported answer")
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool_fn = tools_map.get(tool_name)
            if not tool_fn:
                err_msg = ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found.",
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                messages_produced.append(err_msg)
                current_messages.append(err_msg)
                continue

            try:
                tool_msg = await tool_fn.ainvoke(
                    {
                        "type": "tool_call",
                        "id": tool_id,
                        "name": tool_name,
                        "args": tool_args,
                    }
                )
                artifact = tool_msg.artifact
                if artifact and isinstance(artifact, dict):
                    citations_extracted.extend(artifact.get("source_file_ids", []))
                messages_produced.append(tool_msg)
                current_messages.append(tool_msg)
            except Exception as e:
                err_msg = ToolMessage(
                    content="The tool could not complete this request.",
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                messages_produced.append(err_msg)
                current_messages.append(err_msg)

    else:
        raise ValueError("Model exceeded the tool-call round limit without an answer")
    return messages_produced, list(set(citations_extracted))


async def generate(state: AgentState, config: RunnableConfig) -> dict:
    """Generate independent candidate replies using the supplied conversation context.

    Args:
        state: Messages, user query and candidate count prepared by setup.
        config: Saved model settings and server-selected conversation UUID for tools.

    Returns:
        Candidate pool with each answer's produced messages and citation IDs.
        Every sample receives a fresh seed; prompts and bracket rules are unchanged.

    Raises:
        Exception: Any candidate failure after sibling requests have finished. Remote
            inference uncertainty takes priority so admission is not released early.
    """
    width = state["width"]
    llm = get_llm(config)  # plain client; temperature from config (0.0 for low, >0 for wider modes)

    idx = _sample_idx(state, config)
    print(
        f"\n[IDX {idx}] ==================== Generate Node (Self-Consistency) ====================\n"
    )
    print(f"[IDX {idx}] Question: {state['user_query']}")
    print(f"[IDX {idx}] Width (N samples): {width}")

    cfg = config["configurable"]
    tools_enabled = cfg.get("tools_enabled", True)
    if tools_enabled:
        conversation_id = UUID(str(cfg["conversation_id"]))
        tools_map = {
            "retrieve_context": create_retrieval_tool(conversation_id),
            "fetch_web_page": fetch_web_page,
            "tavily_search": tavily_search,
        }
    else:
        tools_map = {}

    # Run the generator with tools in parallel for width
    tasks = [
        _generate_sample_with_tools(
            _seeded(llm, random.randint(0, 2**31 - 1)), state["messages"], tools_map
        )
        for _ in range(width)
    ]
    # Wait for every sibling request even if one fails. Returning early would let
    # the executor release its slot while another candidate is still generating.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [value for value in results if isinstance(value, BaseException)]
    if failures:
        uncertain = next(
            (error for error in failures if isinstance(error, InferenceStateUnknown)), None
        )
        raise uncertain or failures[0]

    pool = []
    _gen_prompt_text = " ".join(
        m.content for m in state["messages"] if isinstance(getattr(m, "content", None), str)
    )
    for i, (messages_produced, citations) in enumerate(results):
        final_msg = messages_produced[-1]
        content = (
            final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content)
        )
        pool.append(Candidate(answer=content, messages=messages_produced, citations=citations))
        print(f"[IDX {idx}] Sample {i}: id={pool[-1].id[:8]}")
        # TEMPORARY token-usage debug (one block per generation call)
        _log_token_usage(
            idx,
            "Generate Call",
            msg=final_msg,
            prompt_text=_gen_prompt_text,
            completion_text=content,
        )
    print(
        f"\n[IDX {idx}] ==========================================================================\n"
    )

    return {
        "pool": pool,
    }


# ───────────────────── One-shot ranking selector ─────────────────────
# A DOMAIN-GENERAL selector. Unlike majority vote — which can only tally when candidates share a
# normalizable answer (identical GSM8K final numbers) and is therefore undefined for open-ended QA,
# summarization, code generation, RAG, research agents, or essay generation — an LLM ranker compares
# candidates on their merits and picks one with no equality assumption. GSM8K is used only because
# its exact-match answers let us benchmark this ranker head-to-head against majority vote and Pass@N.


def _candidate_labels(n: int) -> list[str]:
    """Stable A, B, C, … labels (benchmark widths 1..9 stay within A..I)."""
    return [chr(ord("A") + i) for i in range(n)]


def _build_ranking_block(user_query: str, candidates: list[Candidate], labels: list[str]) -> str:
    """Question + each candidate's FULL plain-text answer under a letter label. Deliberately does
    NOT inject extracted numbers or vote counts, so the ranker judges the candidates themselves and
    stays independent of majority voting. Showing whole answers (not a normalized key) is also what
    keeps this selector TASK-AGNOSTIC — it works the same for prose, code, or summaries, where no
    comparable answer key exists."""
    parts = [f"Question:\n{user_query}\n"]
    for label, c in zip(labels, candidates):
        parts.append(f"Candidate {label}:\n{c.answer}\n")
    return "\n".join(parts)


def _no_reasoning_prompt(user_query: str, candidates: list[Candidate], labels: list[str]) -> str:
    """Ranking prompt: choose one label, plain text only, no explanation / no reasoning field."""
    opts = " or ".join(labels)
    return (
        _build_ranking_block(user_query, candidates, labels)
        + "\nDetermine which answer is most likely correct.\n"
        + "Consider:\n"
        + "- arithmetic correctness\n"
        + "- logical consistency\n"
        + "- whether the answer fully addresses the question\n"
        + f"Respond ONLY with a single letter: {opts}.\n"
    )


def _parse_letter_choice(text: Optional[str], labels: list[str]) -> Optional[str]:
    """Map a free-text response to one of `labels`. Tolerates 'B', 'Candidate B', 'B.', 'The answer
    is B', etc. Reading-order scan so the FIRST stated label wins. Returns None if nothing matches
    (caller decides the fallback — never falls back to majority voting)."""
    if not text:
        return None
    label_set = set(labels)
    up = text.strip().upper()
    if up in label_set:  # exact single-letter reply
        return up
    m = re.search(r"\b([A-Z])\b", up)  # first standalone letter token
    if m and m.group(1) in label_set:
        return m.group(1)
    for ch in up:  # first valid label char in reading order
        if ch in label_set:
            return ch
    return None


async def rank_no_reasoning(
    user_query: str, candidates: list[Candidate], llm, idx: str = "?"
) -> tuple[Candidate, int, bool]:
    """ONE plain-text ranking call over ALL candidates; parse the letter; return the winning
    Candidate. Domain-general: it compares the candidate outputs themselves and needs no answer-equality key, so it applies equally to
    non-numeric tasks (QA, summaries, code, essays) where majority vote is undefined.
    RETURNS: (winner, rank_calls=1, parse_failed). `parse_failed` is True when the reply maps to no
    valid label — a fallback candidate is still returned. Independent of majority."""
    labels = _candidate_labels(len(candidates))
    label_to_cand = dict(zip(labels, candidates))
    prompt = _no_reasoning_prompt(user_query, candidates, labels)

    resp = await invoke_model(
        llm.bind(max_tokens=16), [HumanMessage(content=prompt)]
    )  # exactly ONE call
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    # TEMPORARY token-usage debug
    _log_token_usage(idx, "Rank Call", msg=resp, prompt_text=prompt, completion_text=text)

    label = _parse_letter_choice(text, labels)
    parse_failed = label is None
    if parse_failed:
        print(
            f"[IDX {idx}] [rank_no_reasoning] unparseable response {text!r}; falling back to candidate {labels[0]}"
        )
        label = labels[0]
    return label_to_cand[label], 1, parse_failed


# ───────────────────────── Tournament selector (hierarchical bracket ranking) ─────────────────────────
# Pure ORCHESTRATION over the one-shot plain-text ranker (rank_no_reasoning): NO new prompt, NO
# regeneration, NO verifier/critic, NO abstention, NO extra generation round — the generator above is
# untouched and ONLY the selector changes. Rather than one N-way ranking call, run a balanced tournament
# where every ranking call compares at most MAX_RANK_GROUP (=3) candidates.


def _tournament_groups(candidates: list["Candidate"]) -> list[list[Candidate]]:
    """One round's bracket split — the MINIMUM-rank-call rule. Rank floor(n/3) full
    groups of MAX_RANK_GROUP (=3); the trailing n%3 candidates each take a BYE (a size-1 group, no
    ranking call) rather than being ranked as a sub-3 group. A group of 3 removes 2 candidates per
    call, so this hits the theoretical minimum ceil((n-1)/2) calls for every n. Byeing the remainder
    (instead of ranking it) is what lets the survivor count drop straight to <=3 and finish in one
    final call — e.g. N=5: rank one [3], bye the other two -> 3 survivors -> 1 final call = 2 calls
    (NOT [3,2] -> 3 calls). Structures: 3 -> [3] (1 call); 5 -> [3] + 2 byes (2 calls); 7 -> [3,3]
    + 1 bye (3 calls); 9 -> [3,3,3] (4 calls); all reach exactly 3 survivors after round 1, then a
    single 3-way final. The list is already shuffled by `tournament_rank`, so which candidates fill
    the ranked groups vs. take byes is random; the A/B/C… labels are display-only (input order)."""
    n = len(candidates)
    if n <= MAX_RANK_GROUP:
        return [list(candidates)]  # final group, ranked in one call
    full = (n // MAX_RANK_GROUP) * MAX_RANK_GROUP  # candidates that fill complete size-3 groups
    groups = [list(candidates[i : i + MAX_RANK_GROUP]) for i in range(0, full, MAX_RANK_GROUP)]
    groups.extend([candidates[j]] for j in range(full, n))  # trailing remainder -> singleton byes
    return groups


async def tournament_rank(
    user_query: str,
    candidates: list[Candidate],
    llm,
    idx: str = "?",
) -> tuple[Candidate, int, bool, int, int]:
    """Hierarchical tournament selector — an ORCHESTRATION layer over the existing one-shot ranker;
    it does NOT introduce a new ranking prompt. Each ranking call sees at most MAX_RANK_GROUP
    candidates, decomposing the N-way comparison that degrades at larger N.
    Algorithm: shuffle once -> `_tournament_groups` ranks floor(n/3) groups of 3 and byes the
    trailing n%3 (minimum rank calls) -> rank each group with the one-shot plain-text ranker ->
    advance winners (a byed singleton carries over with no call) -> recurse until one candidate remains.

    RETURNS: (global_winner, total_rank_calls, parse_failed, tournament_rounds,
              tournament_max_group_size).
      - total_rank_calls: real LLM ranking calls only (groups of >= 2); a bye (size 1) costs nothing.
      - parse_failed: True if ANY group call failed to parse a label (a fallback candidate was used).
      - tournament_max_group_size: the largest group actually submitted to a ranking call (0 if none).
    Independent of majority voting; needs no answer-equality key, so it carries over to non-numeric
    tasks exactly like the one-shot ranker it wraps."""
    group_ranker = rank_no_reasoning

    survivors = list(candidates)

    # Global display labels (A, B, C, …) are pinned to the INPUT order — the same order the
    # tournament_select node prints under "Candidates:" — so the bracket log is reconcilable with
    # that listing and with the winner id printed downstream. Assigned BEFORE the shuffle on purpose:
    # the shuffle only randomizes bracket SEEDING (who meets whom), never a candidate's identity
    # label. LOG-ONLY, and independent of the local A/B/C labels each one-shot ranking call assigns
    # internally to its own 2–3 group members. (Pinning to the shuffled order instead made the log
    # impossible to reconcile with the pool listing — a candidate listed first could print as "B".)
    labels = {c.id: lbl for c, lbl in zip(candidates, _candidate_labels(len(candidates)))}

    def lbl(c: "Candidate") -> str:
        return labels.get(c.id, "?")

    random.shuffle(survivors)  # randomize bracket seeding only; identity labels already fixed above

    print(f"\n[IDX {idx}] ================ Tournament Ranking ================\n")

    total_calls = 0
    rounds = 0
    max_group_size = 0
    parse_failed_any = False

    while len(survivors) > 1:
        rounds += 1
        groups = _tournament_groups(survivors)
        print(f"[IDX {idx}] Round {rounds}")
        next_survivors: list[Candidate] = []
        for gi, group in enumerate(groups, start=1):
            if len(group) == 1:
                # Bye: a lone candidate advances with no ranking call. Under 3B's min-call grouping
                # this is the EXPECTED handling of the trailing n%3 (e.g. N=5 byes 2 -> 3 survivors;
                # N=7 byes 1 -> 3 survivors; N=9 byes none), which is what keeps the call count minimal.
                print(f"[IDX {idx}] Group {gi}: {lbl(group[0])} (bye)")
                print(f"[IDX {idx}] Winner: {lbl(group[0])}\n")
                next_survivors.append(group[0])
                continue

            max_group_size = max(max_group_size, len(group))
            print(f"[IDX {idx}] Group {gi}:")
            if len(group) == 2:
                print(f"[IDX {idx}] {lbl(group[0])} vs {lbl(group[1])}")
            else:  # 3-way (MAX_RANK_GROUP); list vertically
                for c in group:
                    print(f"[IDX {idx}] {lbl(c)}")

            winner, calls, parse_failed = await group_ranker(user_query, group, llm, idx)
            total_calls += calls
            parse_failed_any = parse_failed_any or parse_failed
            print(f"[IDX {idx}] Winner: {lbl(winner)}\n")
            next_survivors.append(winner)
        survivors = next_survivors

    global_winner = survivors[0]

    print(f"[IDX {idx}] Global Winner: {lbl(global_winner)} (id={global_winner.id[:8]})\n")
    print(f"[IDX {idx}] Tournament Rounds: {rounds}")
    print(f"[IDX {idx}] Tournament Rank Calls: {total_calls}")
    print(f"[IDX {idx}] Tournament Max Group Size: {max_group_size}")
    print(f"[IDX {idx}] ====================================================\n")

    return global_winner, total_calls, parse_failed_any, rounds, max_group_size


async def tournament_select(state: AgentState, config: RunnableConfig) -> dict:
    """Tournament selector node. Generation already produced `pool`; here we select Top-1 via a
    HIERARCHICAL TOURNAMENT (`tournament_rank`) instead of one N-way ranking call, to avoid the
    degradation one-shot ranking shows as candidate count grows. Pure orchestration over the
    one-shot plain-text ranker; the generator is untouched.
    Width=1 short-circuits (one candidate, no call, 0 rounds). Preserves EVERY majority/one-shot metric
    and adds tournament_rounds / tournament_rank_calls / tournament_max_group_size. `rank_calls`
    equals `tournament_rank_calls` (both count the tournament's ranking LLM calls), so the existing
    eval aggregation keeps working unchanged.
    RETURNS: final_answer, messages, the selector-log fields, and the three tournament metrics."""
    cfg = config["configurable"]
    pool = state.get("pool", [])
    user_query = state.get("user_query", "")
    idx = _sample_idx(state, config)

    sampled_answers = [c.answer for c in pool]

    print(f"\n[IDX {idx}] ============ Tournament-Select Node ============\n")
    print(f"[IDX {idx}] Rank mode: Tournament")
    print(f"[IDX {idx}] Candidates: {len(pool)}")
    for label, candidate in zip(_candidate_labels(len(pool)), pool):
        print(f"[IDX {idx}] {label}: id={candidate.id[:8]}")

    # sampled_answers, sampled_numbers, vote_distribution, unique_answers = _self_consistency_metrics(pool)

    if not pool:
        print(f"[IDX {idx}] Empty pool.")
        print(f"\n[IDX {idx}] ================================================================\n")
        return {
            "final_answer": "",
            "messages": [AIMessage(content="")],
            "sampled_answers": [],
            "rank_latency": 0.0,
            "rank_calls": 0,
            "winner_candidate_id": None,
            "rank_parse_failed": False,
            "tournament_rounds": 0,
            "tournament_rank_calls": 0,
            "tournament_max_group_size": 0,
        }

    rank_latency = 0.0
    rank_calls = 0
    rank_parse_failed = False
    tournament_rounds = 0
    tournament_max_group_size = 0

    if len(pool) <= 1:
        # Single candidate: nothing to select (no LLM call, no bracket).
        winner = pool[0]
        print(f"[IDX {idx}] Single candidate; no tournament needed.")
    else:
        cfg_copy = config.copy()
        cfg_copy["rank_temperature"] = 0.0  # greedy ranker by default -> reproducible
        llm = get_llm(cfg_copy)
        t0 = time.perf_counter()
        winner, rank_calls, rank_parse_failed, tournament_rounds, tournament_max_group_size = (
            await tournament_rank(user_query, pool, llm, idx)
        )
        rank_latency = time.perf_counter() - t0

    answer = winner.answer

    print(f"[IDX {idx}] Winner: id={winner.id[:8]}")
    print(
        f"[IDX {idx}] Tournament rounds: {tournament_rounds} | rank calls: {rank_calls} | max group: {tournament_max_group_size}"
    )
    print(f"[IDX {idx}] Rank latency: {rank_latency:.3f}s")
    print(f"[IDX {idx}] Parse failed: {rank_parse_failed}")
    print(f"\n[IDX {idx}] ================================================================\n")

    return {
        "final_answer": answer,
        "messages": winner.messages,
        "citations": winner.citations,
        "sampled_answers": sampled_answers,
        "rank_latency": rank_latency,
        "rank_calls": rank_calls,
        "winner_candidate_id": winner.id,
        "rank_parse_failed": rank_parse_failed,
        "tournament_rounds": tournament_rounds,
        "tournament_rank_calls": rank_calls,
        "tournament_max_group_size": tournament_max_group_size,
    }
