import os
import json
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF
import markdown

CHUNK_SIZE = 500
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store/index.faiss"
CHUNKS_PATH = "vector_store/chunks.json"


def load_txt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])


def load_md_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
    return html.replace("<p>", "").replace("</p>", "\n")


def load_documents(directory: str) -> List[str]:
    texts = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            texts.append(load_txt_file(path))
        elif filename.endswith(".pdf"):
            texts.append(load_pdf_file(path))
        elif filename.endswith(".md"):
            texts.append(load_md_file(path))
    return texts


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def build_vector_store(chunks: List[str]):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, VECTOR_STORE_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)


if __name__ == "__main__":
    docs = load_documents("data")
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    build_vector_store(all_chunks)
    print(f"[INFO] Built vector store with {len(all_chunks)} chunks.")
