"""
RAG Generator — sends retrieved context to an LLM and gets a cited answer.

Flow:
  1. Receive user query + retrieved context chunks
  2. Build a structured prompt (system + context + question)
  3. Call GPT-4o-mini via LangChain ChatOpenAI
  4. Parse the response to extract citation references [1], [2], etc.
  5. Return a GeneratedAnswer: text + structured citations + confidence

Key design decisions:
  - temperature=0: deterministic answers (no creativity, maximum factuality)
  - Pydantic response model: structured output, not raw text
  - Citation extraction: maps [1] [2] references back to source Documents
  - "I don't know" handling: LLM is instructed to say so when unsure

Interview talking point:
  "I set temperature=0 for the RAG generator because we want deterministic,
  fact-based answers — not creative writing. The LLM is given a strict
  instruction to only use the provided context and cite every claim."
"""

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import settings

# ─────────────────────────────────────────────
# System prompt — the most important prompt in the project
# Every word here shapes answer quality
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert company research analyst with access to a curated knowledge base.

INSTRUCTIONS:
1. Answer the user's question using ONLY the information provided in the context below.
2. Cite every factual claim using numbered references like [1], [2], [3].
3. If the context does not contain enough information to answer, say exactly:
   "I don't have enough information in my knowledge base to answer this question."
4. Never fabricate facts, statistics, or quotes.
5. Be concise but complete — a good answer is 2-4 paragraphs.
6. After your answer, list the sources you cited under "Sources:".

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
<answer>
Your answer with inline citations like [1] and [2].
</answer>

Sources:
[1] Title of source — URL or filename
[2] Title of source — URL or filename

CONTEXT:
{context}"""


# ─────────────────────────────────────────────
# Response data models
# ─────────────────────────────────────────────

@dataclass
class Citation:
    """A single source citation."""
    number: int           # The [1] reference number
    source: str           # URL or filename
    title: str            # Document title
    page: int | None      # Page number (for PDFs)
    relevance_score: float  # How relevant this chunk was

    def format(self) -> str:
        page_info = f", page {self.page}" if self.page else ""
        return f"[{self.number}] {self.title}{page_info} — {self.source}"


@dataclass
class GeneratedAnswer:
    """
    Structured output from the generator.

    This is what gets returned to the API and displayed in the UI.
    Having a structured model (not raw text) makes the frontend simple:
    just render answer_text and loop over citations.
    """
    answer_text: str                  # The main answer with [1] [2] markers
    citations: list[Citation]         # Structured citation objects
    confidence: float                 # 0.0–1.0 based on retrieval scores
    source_documents: list[Document]  # Raw chunks (for debugging/evaluation)
    had_enough_context: bool          # False if LLM said "I don't know"

    def format_with_sources(self) -> str:
        """Format the full answer + citations for display."""
        lines = [self.answer_text, "", "Sources:"]
        for citation in self.citations:
            lines.append(citation.format())
        return "\n".join(lines)


class Generator:
    """
    Generates cited answers using an LLM + retrieved context.

    Usage:
        generator = Generator()
        answer = generator.generate(
            query="Who founded Danone?",
            context_docs=[doc1, doc2],
            context_string="[1] Source: ...\nDanone was founded by..."
        )
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.llm_model

        # temperature=0: no randomness — same question always gets same answer
        # This is critical for a research tool where consistency matters
        self._llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        logger.info(f"Generator initialized with model: {self.model_name}")

    def generate(
        self,
        query: str,
        context_docs: list[Document],
        context_string: str,
    ) -> GeneratedAnswer:
        """
        Generate a cited answer from retrieved context.

        Args:
            query: The user's question
            context_docs: Raw Document objects (for citation metadata)
            context_string: Pre-formatted context string (from retriever)

        Returns:
            GeneratedAnswer with text, citations, and confidence score
        """
        if not context_docs:
            return self._no_context_answer(query)

        # Build the prompt
        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(context=context_string))
        human_msg = HumanMessage(content=query)

        logger.info(f"Calling {self.model_name} with {len(context_docs)} context chunks...")

        # Make the LLM call
        response = self._llm.invoke([system_msg, human_msg])
        raw_text = response.content

        logger.info(f"Response received ({len(raw_text)} chars)")

        # Parse the response into structured output
        answer_text = self._extract_answer(raw_text)
        citations = self._build_citations(context_docs)
        confidence = self._calculate_confidence(context_docs)
        had_context = "don't have enough information" not in answer_text.lower()

        return GeneratedAnswer(
            answer_text=answer_text,
            citations=citations,
            confidence=confidence,
            source_documents=context_docs,
            had_enough_context=had_context,
        )

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _extract_answer(self, raw_text: str) -> str:
        """
        Extract the answer from between <answer> tags.
        Falls back to the full response if tags are missing.
        """
        match = re.search(r"<answer>(.*?)</answer>", raw_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: return everything before "Sources:" section
        if "Sources:" in raw_text:
            return raw_text.split("Sources:")[0].strip()

        return raw_text.strip()

    def _build_citations(self, context_docs: list[Document]) -> list[Citation]:
        """
        Build Citation objects from the retrieved documents.

        The citations are numbered [1], [2], ... matching the order
        the context was presented to the LLM.
        """
        citations = []
        for i, doc in enumerate(context_docs, start=1):
            meta = doc.metadata
            citation = Citation(
                number=i,
                source=meta.get("source", "Unknown"),
                title=meta.get("title", meta.get("source", "Unknown")),
                page=meta.get("page"),
                relevance_score=meta.get("relevance_score", 0.0),
            )
            citations.append(citation)
        return citations

    def _calculate_confidence(self, context_docs: list[Document]) -> float:
        """
        Estimate answer confidence from retrieval scores.

        Uses the average relevance score of retrieved chunks.
        Higher relevance = LLM had better context = more confident answer.

        Scale: 0.0 (no match) → 1.0 (perfect match)
        Typical good retrieval: 0.5–0.8
        """
        if not context_docs:
            return 0.0
        scores = [
            doc.metadata.get("relevance_score", 0.0)
            for doc in context_docs
        ]
        avg_score = sum(scores) / len(scores)
        # Clamp to [0, 1]
        return round(min(max(avg_score, 0.0), 1.0), 3)

    def _no_context_answer(self, query: str) -> GeneratedAnswer:
        """Return a graceful answer when no context was retrieved."""
        return GeneratedAnswer(
            answer_text="I don't have enough information in my knowledge base to answer this question. Please ingest the company data first.",
            citations=[],
            confidence=0.0,
            source_documents=[],
            had_enough_context=False,
        )
