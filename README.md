# 📚 Chroma Knowledge Search — RAG System

**AI-powered semantic document search** platform enabling **context-aware Q&A** across uploaded PDFs/DOCX/TXT documents.

## Uses

- 🧠 **OpenAI** — embeddings + LLM responses + moderation  
- 🔎 **ChromaDB** — vector storage & retrieval  
- ⚙️ **FastAPI** — secure backend RAG service  
- 🎛️ **Streamlit** — interactive upload + question UI  
- 🔐 **API Key Authentication**  
- 🗄️ **SQLite metadata DB**  
- 🐳 **Docker Compose** — fully containerized  
- 📦 **Poetry** — dependency management

## 🚀 Features

| Capability | Status |
|-----------|:-----:|
| Upload PDF, DOCX, TXT | ✅ |
| Automatic text extraction | ✅ |
| Chunking with embeddings | ✅ |
| Secure API-Key isolation | ✅ |
| Contextual answers (RAG) | ✅ |
| Safe content moderation | ✅ |
| Storage in ChromaDB | ✅ |
| Metadata persistence | ✅ |
| Multi-doc querying | ✅ |
| Containerized deployment | ✅ |
| Fully local execution | ✅ |

## ▶️ Run the App

```bash
docker compose up --build
```

| Service          | URL                                                                              |
| ---------------- | -------------------------------------------------------------------------------- |
| ✅ App UI         | [http://localhost:8501](http://localhost:8501)                                   |
| ✅ Backend Docs   | [http://localhost:8000/docs](http://localhost:8000/docs)                         |
| ✅ Backend Health | [http://localhost:8000/health](http://localhost:8000/health)                     |
| ✅ ChromaDB REST  | [http://localhost:8001/api/v2/heartbeat](http://localhost:8001/api/v2/heartbeat) |
