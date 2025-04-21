# AskMyDocs: RAG-powered Document Chatbot

**AskMyDocs** is a Retrieval-Augmented Generation (RAG) chatbot that enables intelligent querying of your personal text files. It retrieves relevant context using vector search and generates factual answers using OpenAI or HuggingFace models.

## 🚀 Features

- Text, PDF, and Markdown ingestion
- Semantic chunking using SentenceTransformers
- Vector search with FAISS
- LLM-powered generation via OpenAI or HuggingFace
- Streamlit-based user interface
- Dockerized and Kubernetes-ready for deployment
- Modular, testable architecture with `pytest` support

---

## 🧱 Project Structure

```
askmydocs/
├── app.py                  # Streamlit frontend
├── Dockerfile              # Docker container setup
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── data/                   # Your .txt/.pdf/.md files go here
├── scripts/                # Core processing modules
│   ├── ingest.py           # Chunking + embedding + FAISS
│   └── query.py            # Retrieval + LLM generation
├── vector_store/           # Vector index and chunk cache
├── k8s/                    # Kubernetes manifests
│   ├── askmydocs-deployment.yaml
│   └── openai-secret.yaml
└── tests/                  # Unit tests
    └── test_ingest.py
```

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/yourname/askmydocs.git
cd askmydocs

# 2. Place .txt/.pdf/.md files into data/
mkdir -p data
cp your_docs.txt data/

# 3. Ingest documents into vector DB
python scripts/ingest.py

# 4. Launch app
streamlit run app.py
```

---

## 🔐 OpenAI API Key

To use OpenAI's GPT model for answering questions, set the key via environment variable:

```bash
export OPENAI_API_KEY=your-api-key
```

---

## 🐳 Docker Usage

```bash
docker build -t askmydocs .
docker run -p 8501:8501 -e OPENAI_API_KEY=your-api-key askmydocs
```

---

## ☸️ Kubernetes Deployment

```bash
# Start Minikube
minikube start

# Add OpenAI key as secret
kubectl apply -f k8s/openai-secret.yaml

# Deploy app
kubectl apply -f k8s/askmydocs-deployment.yaml

# Port forward to access app
kubectl port-forward service/askmydocs-service 8501:80
```

Visit: `http://localhost:8501`

---

## ✅ Tests

```bash
pip install pytest
pytest tests/
```

---

## 📚 Use Cases

- Research document Q&A
- Personal note summarization
- Internal company knowledge assistant
- Scientific literature retrieval

---

## 📄 License

MIT License. Built by Syed Yusuf Ali.
