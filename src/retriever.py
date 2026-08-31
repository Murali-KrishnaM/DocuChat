"""
Retrieval: embed the query, do FAISS semantic search for candidate chunks,
then rerank those candidates with a cross-encoder for better relevance
before handing the top passages to the generator.
"""
import pickle
from dataclasses import dataclass

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from src import config
from src.ingest import Chunk


@dataclass
class RetrievedPassage:
    text: str
    source: str
    page: int
    score: float  # cross-encoder relevance score


class Retriever:
    """Loads the FAISS index + metadata once and serves queries against them."""

    def __init__(self):
        if not config.FAISS_INDEX_PATH.exists() or not config.METADATA_PATH.exists():
            raise FileNotFoundError(
                "No index found. Run `python -m src.ingest` first to build the FAISS index."
            )

        with open(config.METADATA_PATH, "rb") as f:
            meta = pickle.load(f)
        self.chunks: list[Chunk] = meta["chunks"]
        embedding_model_name = meta["embedding_model"]

        self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        self.embedder = SentenceTransformer(embedding_model_name)
        self.cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)

    def _semantic_search(self, query: str, top_k: int) -> list[Chunk]:
        query_vec = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)
        return [self.chunks[i] for i in indices[0] if i != -1]

    def _rerank(self, query: str, candidates: list[Chunk], top_k: int) -> list[RetrievedPassage]:
        if not candidates:
            return []
        pairs = [[query, c.text] for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [
            RetrievedPassage(text=c.text, source=c.source, page=c.page, score=float(s))
            for c, s in ranked[:top_k]
        ]

    def retrieve(
        self,
        query: str,
        top_k_retrieve: int = config.TOP_K_RETRIEVE,
        top_k_rerank: int = config.TOP_K_RERANK,
    ) -> list[RetrievedPassage]:
        """End-to-end: FAISS candidate search -> cross-encoder rerank -> top passages."""
        candidates = self._semantic_search(query, top_k_retrieve)
        return self._rerank(query, candidates, top_k_rerank)
