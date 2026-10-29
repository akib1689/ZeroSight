"""QA retrieval tools: BM25 ``search`` and ``lookup`` over per-question context.

Proper ``BaseTool`` subclasses that hold the per-question BM25 index as instance
state. Instantiated per-question by ``make_qa_tools(context_paragraphs)``.
"""

import string
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Normalize for BM25: lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


class SearchInput(BaseModel):
    query: str = Field(..., description="Search query to find relevant paragraphs")


class LookupInput(BaseModel):
    term: str = Field(..., description="Term or phrase to find in the context")


class SearchTool(BaseTool):
    """BM25 search over per-question context paragraphs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "search"
    description: str = (
        "Search the context paragraphs for information relevant to the query. "
        "Returns the top matching paragraphs with their titles."
    )
    args_schema: type[BaseModel] = SearchInput

    bm25: Optional[Any] = None
    titles: list[str] = Field(default_factory=list)
    full_texts: list[str] = Field(default_factory=list)

    def _run(self, query: str) -> str:
        if self.bm25 is None:
            return "No context available."

        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, _score in ranked[:3]:
            results.append(f"[{self.titles[idx]}]\n{self.full_texts[idx][:500]}")

        return "\n\n---\n\n".join(results) if results else "No matching paragraphs found."


class LookupTool(BaseTool):
    """Find a term in the context paragraphs with surrounding text."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "lookup"
    description: str = (
        "Look up a term or phrase in the context paragraphs. "
        "Returns the first paragraph containing the term with surrounding context."
    )
    args_schema: type[BaseModel] = LookupInput

    titles: list[str] = Field(default_factory=list)
    full_texts: list[str] = Field(default_factory=list)

    def _run(self, term: str) -> str:
        term_lower = term.lower()
        for i, text in enumerate(self.full_texts):
            pos = text.lower().find(term_lower)
            if pos != -1:
                start = max(0, pos - 150)
                end = min(len(text), pos + len(term) + 350)
                return f"[{self.titles[i]}]\n...{text[start:end]}..."
        return f"Term '{term}' not found in any paragraph."


def make_qa_tools(context_paragraphs: list[dict]) -> list[BaseTool]:
    """Create search + lookup tools bound to per-question context.

    Args:
        context_paragraphs: List of dicts with keys ``title`` (str) and
            ``sentences`` (list[str]) or ``paragraph_text`` (str).

    Returns:
        List of BaseTool instances: [SearchTool, LookupTool].
    """
    titles = []
    full_texts = []
    for para in context_paragraphs:
        title = para.get("title", "")
        text = para.get("paragraph_text", "")
        if not text and "sentences" in para:
            text = " ".join(para["sentences"])
        titles.append(title)
        full_texts.append(f"{title}. {text}")

    tokenized = [_tokenize(t) for t in full_texts]
    bm25: Optional[Any] = BM25Okapi(tokenized) if tokenized else None

    return [
        SearchTool(bm25=bm25, titles=titles, full_texts=full_texts),
        LookupTool(titles=titles, full_texts=full_texts),
    ]
