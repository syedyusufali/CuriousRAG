# scripts/query.py
import os
import json
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

try:
    import openai
except ImportError:
    openai = None

from transformers import pipeline

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store/index.faiss"
CHUNKS_PATH = "vector_store/chunks.json"


class RAGQueryEngine:
    def __init__(self, k: int = 5):
        self.k = k
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index(VECTOR_STORE_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.use_openai = openai is not None and os.getenv("OPENAI_API_KEY")
        self.llm = (
            lambda prompt: self.query_openai(prompt)
            if self.use_openai else
            self.query_huggingface()
        )
        if not self.use_openai:
            self.hf_pipeline = pipeline("text-generation", model="gpt2", max_length=300)

    def retrieve(self, query: str) -> List[str]:
        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, self.k)
        return [self.chunks[i] for i in indices[0]]

    def query_openai(self, prompt: str) -> str:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You answer questions based on provided context."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.2
        )
        return response["choices"][0]["message"]["content"].strip()

    def query_huggingface(self):
        def inner(prompt: str) -> str:
            return self.hf_pipeline(prompt)[0]["generated_text"][len(prompt):].strip()
        return inner

    def generate(self, query: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)
        prompt = (
            f"You are a helpful assistant. Use the following context to answer the question.\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        return self.llm(prompt)

    def query(self, question: str) -> str:
        context_chunks = self.retrieve(question)
        return self.generate(question, context_chunks)


if __name__ == "__main__":
    import sys
    engine = RAGQueryEngine()
    question = " ".join(sys.argv[1:]) or "What is this document about?"
    print(engine.query(question))
