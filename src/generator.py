"""
Answer generation: takes the reranked passages and the user's question,
builds a grounded prompt, and calls the Gemini API for the final answer.
"""
import google.generativeai as genai

from src import config
from src.retriever import RetrievedPassage

SYSTEM_INSTRUCTIONS = """You are DocuChat, an assistant that answers questions using ONLY the
provided context passages from the user's own documents. Rules:
- If the answer is not contained in the context, say you don't know based on the
  provided documents. Do not use outside knowledge to fill gaps.
- Be concise and directly answer the question.
- When you use a fact from a passage, refer to it by its [source] tag inline,
  e.g. "... as noted in [notes.pdf, p.2]."
"""


def _configure():
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    genai.configure(api_key=config.GEMINI_API_KEY)


def _build_prompt(question: str, passages: list[RetrievedPassage]) -> str:
    context_blocks = []
    for p in passages:
        tag = f"{p.source}" + (f", p.{p.page}" if p.page else "")
        context_blocks.append(f"[{tag}]\n{p.text}")
    context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no relevant passages found)"

    return f"""{SYSTEM_INSTRUCTIONS}

Context passages:
{context_str}

Question: {question}

Answer:"""


def generate_answer(question: str, passages: list[RetrievedPassage]) -> str:
    """Calls Gemini with the grounded prompt and returns the answer text."""
    _configure()
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    prompt = _build_prompt(question, passages)
    response = model.generate_content(prompt)
    return response.text.strip()
