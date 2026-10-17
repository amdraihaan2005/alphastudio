"""
Multi-Query Decomposer powered by GroqCloud and LangChain Expression Language (LCEL).

Decomposes complex financial queries (such as cross-document comparisons, multi-metric evaluations,
and acronym-dense inquiries) into 2 to 4 focused, atomic sub-queries for parallel hybrid retrieval.
Exposes `MultiQueryDecomposer(RunnableSerializable)` with full LCEL support.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_groq import ChatGroq
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert financial research assistant.
Your task is to analyze a user's query and decompose it into 2 to 4 distinct, highly focused search queries.

Guidelines:
1. Multi-Entity / Comparative Queries: If comparing two companies or years (e.g., "Compare NVDA 2023 vs 2024 gross margin"), create separate sub-queries for each entity/period.
2. Metric Isolation: If multiple financial metrics are requested (e.g., "operating cash flow and long term debt"), isolate each metric into its own query.
3. Terminology Expansion: Expand common financial jargon and abbreviations (e.g., "COGS" -> "cost of goods sold", "CapEx" -> "capital expenditures", "R&D" -> "research and development").
4. Keep sub-queries concise and keyword-rich for dense vector and full-text retrieval.
5. Always generate at least 2 distinct sub-queries. Do NOT repeat the exact same query."""


class MultiQueryPlan(BaseModel):
    sub_queries: list[str] = Field(
        default_factory=list,
        description="A list of 2 to 4 distinct, focused search queries targeting specific facts, metrics, or time periods.",
    )


_groq_llm: ChatGroq | None = None


def get_groq_llm(temperature: float = 0.0) -> ChatGroq:
    """Singleton getter for the official GroqCloud ChatGroq client."""
    global _groq_llm
    if _groq_llm is None:
        if not settings.GROQ_API_KEY or not settings.GROQ_API_KEY.strip():
            raise ValueError("GROQ_API_KEY is not configured in settings/environment.")
        _groq_llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_LLM_MODEL,
            temperature=temperature,
        )
    return _groq_llm


class MultiQueryDecomposer(RunnableSerializable[str, list[str]]):
    """
    Production-grade LangChain Runnable for Multi-Query Decomposition.
    Inherits `RunnableSerializable[str, list[str]]` with full LCEL support.
    """

    llm: Any | None = Field(default=None, description="LangChain ChatModel for query decomposition")
    max_sub_queries: int = Field(default=4, ge=1, le=10, description="Maximum sub-queries to emit")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    system_prompt: str = Field(default=SYSTEM_PROMPT, description="System decomposition prompt")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_active_llm(self) -> Any:
        if self.llm is not None:
            return self.llm
        return get_groq_llm(temperature=self.temperature)

    def _build_prompt(self, query: str) -> list[Any]:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "User Query: {query}\n\nDecompose this query into sub-queries:"),
            ]
        )
        return prompt.format_messages(query=query)

    def _process_result(self, cleaned_query: str, plan: MultiQueryPlan) -> list[str]:
        sub_queries = [q.strip() for q in plan.sub_queries if q and q.strip()]
        if cleaned_query not in sub_queries:
            sub_queries.insert(0, cleaned_query)
        logger.info(
            f"Decomposed query '{cleaned_query}' into {len(sub_queries)} sub-queries: {sub_queries}"
        )
        return sub_queries[: self.max_sub_queries]

    def invoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Synchronous query decomposition conforming to Runnable interface."""
        cleaned_query = input.strip()
        if not cleaned_query:
            return []

        try:
            active_llm = self._get_active_llm()
            if hasattr(active_llm, "with_structured_output"):
                structured_llm = active_llm.with_structured_output(MultiQueryPlan)
                messages = self._build_prompt(cleaned_query)
                plan: MultiQueryPlan = structured_llm.invoke(messages, config=config)
                return self._process_result(cleaned_query, plan)
        except Exception as exc:
            logger.warning(
                f"Query decomposition failed ({exc}). Gracefully falling back to original query.",
                exc_info=False,
            )
            return [cleaned_query]

        return [cleaned_query]

    async def ainvoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Asynchronous query decomposition conforming to Runnable interface."""
        cleaned_query = input.strip()
        if not cleaned_query:
            return []

        try:
            active_llm = self._get_active_llm()
            if hasattr(active_llm, "with_structured_output"):
                structured_llm = active_llm.with_structured_output(MultiQueryPlan)
                messages = self._build_prompt(cleaned_query)
                plan: MultiQueryPlan = await structured_llm.ainvoke(messages, config=config)
                return self._process_result(cleaned_query, plan)
        except Exception as exc:
            logger.warning(
                f"Async query decomposition failed ({exc}). Gracefully falling back to original query.",
                exc_info=False,
            )
            return [cleaned_query]

        return [cleaned_query]


_default_decomposer: MultiQueryDecomposer | None = None


def get_default_decomposer() -> MultiQueryDecomposer:
    """Singleton getter for the default MultiQueryDecomposer Runnable instance."""
    global _default_decomposer
    if _default_decomposer is None:
        _default_decomposer = MultiQueryDecomposer()
    return _default_decomposer


async def decompose_query(
    query: str,
    max_sub_queries: int = 4,
    llm: BaseLanguageModel | Any | None = None,
) -> list[str]:
    """
    Decomposes a user prompt into focused sub-queries using GroqCloud or an injected LLM.
    Guarantees that the original query is included and returns a resilient
    fallback to [query] if the model call fails or returns empty.
    """
    if llm is not None or max_sub_queries != 4:
        decomposer = MultiQueryDecomposer(llm=llm, max_sub_queries=max_sub_queries)
        return await decomposer.ainvoke(query)
    return await get_default_decomposer().ainvoke(query)


def decompose_query_sync(
    query: str,
    max_sub_queries: int = 4,
    llm: BaseLanguageModel | Any | None = None,
) -> list[str]:
    """
    Synchronous query decomposition executing directly via LCEL .invoke().
    Eliminates threadpool allocations and event-loop blocking.
    """
    if llm is not None or max_sub_queries != 4:
        decomposer = MultiQueryDecomposer(llm=llm, max_sub_queries=max_sub_queries)
        return decomposer.invoke(query)
    return get_default_decomposer().invoke(query)
