from uuid import UUID

from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableConfig
from src.backend.ai.langgraph.graph_states import AgentState
from src.backend.ai.langgraph.graph_nodes import setup, generate, tournament_select
from src.backend.ai.system_prompt import get_localmind_system_instruction_with_personality

builder = StateGraph(AgentState)
builder.add_node("setup", setup)
builder.add_node("generate", generate)
builder.add_node("tournament_select", tournament_select)

builder.add_edge(START, "setup")
builder.add_edge("setup", "generate")
builder.add_edge("generate", "tournament_select")
builder.add_edge("tournament_select", END)

GRAPH = builder.compile()


class LocalMindAI:
    """Generate and rank candidate replies with server-bound document retrieval.

    No model is loaded by construction; the compiled graph selects its configured
    client at invocation. Every tool-enabled call must provide a conversation UUID.
    """

    def __init__(self) -> None:
        self._graph = GRAPH

    async def chat(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        personality: str,
        reasoning_mode: str,
        temperature: float,
        tools_enabled: bool = True,
        conversation_id: UUID | None = None,
    ) -> dict:
        """Run generation/ranking with a document tool restricted to the selected chat.

        Args:
            messages: Role/content conversation context, already selected by the backend.
            model_name: Inference model name registered in llama-swap.
            personality: System instruction variant, such as general or researcher.
            reasoning_mode: Mode selecting the candidate count and ranking behavior.
            temperature: Sampling temperature forwarded to the model.
            tools_enabled: Whether document/web tools may be called during generation.
            conversation_id: Backend-selected UUID required when tools are enabled;
                omitted for tool-free summarization. Never supplied by model arguments.

        Returns:
            Final answer, winning citation IDs and compact tournament metadata;
            raw candidate answers and prompts are not included in persisted metadata.

        Raises:
            ValueError: If a tool-enabled call lacks a conversation UUID or generation
                finishes without a nonblank answer.
            Exception: Graph/model errors propagate to the calling run handler.
        """
        if tools_enabled and not isinstance(conversation_id, UUID):
            raise ValueError("Tool-enabled chat requires a server-selected conversation UUID")
        system_instruction = get_localmind_system_instruction_with_personality(personality)

        # Prepend the system prompt at the beginning of history, merging if summary is already present
        if messages and messages[0].get("role") == "system":
            merged_content = f"{system_instruction}\n\n{messages[0]['content']}"
            messages_with_system = [{"role": "system", "content": merged_content}] + messages[1:]
        else:
            messages_with_system = [{"role": "system", "content": system_instruction}] + messages

        init: AgentState = {
            "messages": messages_with_system,
        }

        config: RunnableConfig = {
            "configurable": {
                "model_name": model_name,
                "reasoning_mode": reasoning_mode,
                "temperature": temperature,
                "personality": personality,
                "tools_enabled": tools_enabled,
                "conversation_id": str(conversation_id) if conversation_id is not None else None,
            }
        }

        state = await self._graph.ainvoke(init, config)
        answer = state.get("final_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Generation finished without an answer")
        metadata = {
            name: state[name]
            for name in (
                "winner_candidate_id",
                "rank_latency",
                "rank_calls",
                "rank_parse_failed",
                "tournament_rounds",
                "tournament_rank_calls",
                "tournament_max_group_size",
            )
            if name in state
        }
        metadata["candidate_count"] = len(state.get("pool", []))
        return {
            "final_answer": answer,
            "citations": state.get("citations", []),
            "generation_metadata": metadata,
        }

    async def baseline_chat(
        self,
        messages,
        model_name: str,
    ) -> str:
        """Generate one tool-free low-reasoning reply for summary updates.

        Args:
            messages: Backend-prepared summary instructions and conversation context.
            model_name: Inference model name used for the summary.

        Returns:
            Plain final answer text, without document/web retrieval or citations.
        """
        result = await self(messages, model_name, "general", "low", 0.0, tools_enabled=False)
        return result["final_answer"]

    async def __call__(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        personality: str,
        reasoning_mode: str,
        temperature: float,
        tools_enabled: bool = True,
        conversation_id: UUID | None = None,
    ) -> dict:
        """Call chat with the same arguments and preserve the server's conversation scope.

        Args:
            messages: Backend-selected role/content context.
            model_name: Inference model name.
            personality: System instruction variant.
            reasoning_mode: Candidate generation/ranking mode.
            temperature: Sampling temperature.
            tools_enabled: Whether tools can be invoked.
            conversation_id: Backend chat UUID, required for document retrieval.

        Returns:
            The answer/citation dictionary returned by chat; its errors propagate.
        """
        return await self.chat(
            messages,
            model_name,
            personality,
            reasoning_mode,
            temperature,
            tools_enabled,
            conversation_id,
        )
