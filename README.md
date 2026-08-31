# DocuChat

A Retrieval-Augmented Generation (RAG) app for querying your own PDFs/notes,
running entirely on local compute except for the final answer generation call.

**Pipeline:** Sentence-Transformer embeddings → local FAISS vector search →
cross-encoder reranking → grounded answer generation via the Gemini API →
Streamlit UI with source-passage citations.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and paste your Gemini API key
# (get a free one at https://aistudio.google.com/apikey)
```

## Usage

```bash
streamlit run app.py
```

1. In the sidebar, upload one or more `.pdf`, `.txt`, or `.md` files.
2. Click **Build / Rebuild Index** — this chunks the text, embeds it with
   Sentence-Transformers, and builds a local FAISS index.
3. Ask questions in the chat box. Each answer includes an expandable
   **Sources** section showing which passages (and page numbers) were used.

## How it works

1. **Ingest** (`src/ingest.py`): documents are loaded, split into overlapping
   chunks with LangChain's `RecursiveCharacterTextSplitter`, embedded with a
   Sentence-Transformer model (`all-MiniLM-L6-v2` by default), and stored in
   a local FAISS `IndexFlatIP` index (cosine similarity via normalized vectors).
2. **Retrieve** (`src/retriever.py`): the query is embedded and FAISS returns
   the top-N candidate chunks. A cross-encoder
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) then rescoring the (query, chunk)
   pairs directly, which is more accurate than embedding similarity alone,
   and the top-K are kept.
3. **Generate** (`src/generator.py`): the reranked passages are inserted into
   a grounded prompt and sent to the Gemini API, which is instructed to
   answer only from the given context and cite sources inline.
4. **UI** (`app.py`): Streamlit handles uploads, indexing, chat history, and
   rendering of the cited source passages.

## Rebuilding the index

Re-run indexing any time you add or change documents in `data/raw_docs/` —
either via the sidebar button or:

```bash
python -m src.ingest
```

## Notes

- Everything except the final Gemini call runs locally — no cloud vector DB,
  no external embedding API.
- Swap `EMBEDDING_MODEL`, `CROSS_ENCODER_MODEL`, `GEMINI_MODEL`, chunk sizes,
  and top-k values in `.env` without touching code.
