"""
High-level pipeline: query -> retrieve + rerank -> generate grounded answer.
This is the single entry point the Streamlit app calls.
"""
from dataclasses import dataclass

from src.retriever import Retriever, RetrievedPassage
from src.generator import generate_answer


@dataclass
class Answer:
    text: str
    passages: list[RetrievedPassage]


def answer_question(retriever: Retriever, question: str) -> Answer:
    passages = retriever.retrieve(question)
    answer_text = generate_answer(question, passages)
    return Answer(text=answer_text, passages=passages)
