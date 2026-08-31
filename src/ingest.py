"""
Ingestion pipeline: load raw documents -> chunk -> embed -> build FAISS index.

Supports .pdf and .txt/.md files dropped into data/raw_docs/.
Persists:
  - storage/faiss_index/index.faiss   (the vector index)
  - storage/metadata.pkl              (chunk text + source/page for citations)
"""
import pickle
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src import config


@dataclass
class Chunk:
    text: str
    source: str   # filename
    page: int     # 1-indexed page number (0 for non-paginated text files)
    chunk_id: int


def _load_pdf(path: Path) -> list[tuple[str, int]]:
    """Returns list of (page_text, page_number)."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i))
    return pages


def _load_text(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(text, 0)] if text.strip() else []


def load_documents(raw_docs_dir: Path = config.RAW_DOCS_DIR) -> dict[str, list[tuple[str, int]]]:
    """Loads every supported file in raw_docs_dir. Returns {filename: [(text, page), ...]}."""
    docs = {}
    supported = {".pdf", ".txt", ".md"}
    files = [f for f in raw_docs_dir.iterdir() if f.suffix.lower() in supported]

    if not files:
        raise FileNotFoundError(
            f"No supported documents found in {raw_docs_dir}. "
            "Add .pdf, .txt, or .md files there first."
        )

    for f in files:
        if f.suffix.lower() == ".pdf":
            docs[f.name] = _load_pdf(f)
        else:
            docs[f.name] = _load_text(f)
    return docs


def chunk_documents(docs: dict[str, list[tuple[str, int]]]) -> list[Chunk]:
    """Splits each page/document into overlapping chunks using LangChain's splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    chunk_id = 0
    for filename, pages in docs.items():
        for page_text, page_num in pages:
            for piece in splitter.split_text(page_text):
                if piece.strip():
                    chunks.append(Chunk(text=piece.strip(), source=filename, page=page_num, chunk_id=chunk_id))
                    chunk_id += 1
    return chunks


def build_index(chunks: list[Chunk], model_name: str = config.EMBEDDING_MODEL) -> None:
    """Embeds all chunks and writes a FAISS index + metadata file to disk."""
    if not chunks:
        raise ValueError("No chunks to index. Check your source documents.")

    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    config.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))

    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "embedding_model": model_name}, f)

    print(f"Indexed {len(chunks)} chunks from {len(set(c.source for c in chunks))} document(s).")
    print(f"Saved index -> {config.FAISS_INDEX_PATH}")
    print(f"Saved metadata -> {config.METADATA_PATH}")


def run_ingestion() -> None:
    docs = load_documents()
    chunks = chunk_documents(docs)
    build_index(chunks)


if __name__ == "__main__":
    run_ingestion()
