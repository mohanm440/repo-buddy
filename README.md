# 🤖 Repo Buddy

> **Understand any GitHub repository with AI.**

Repo Buddy is a production-grade, AI-powered GitHub Repository Intelligence and Retrieval-Augmented Generation (RAG) platform. It allows developers, recruiters, and engineers to explore, query, and analyze any public GitHub repository using hybrid code search, live token streaming, and cross-file reasoning.

---

## 🚀 Live Demo

[**Try Repo Buddy Live Demo**](YOUR_STREAMLIT_APP_URL)

---

## 💻 Source Code

The complete source code is available in this GitHub repository.

---

## ✨ Key Features

- 🔍 **Hybrid RAG Retrieval**: Combines BM25 lexical keyword search with dense FAISS vector embeddings and Reciprocal Rank Fusion (RRF) scoring.
- ⚡ **Instant Token Streaming**: Live line-by-line response rendering using Groq's high-throughput LLM inference API.
- 🔀 **Commit & Diff Intelligence**: Fetches recent commits and pull requests directly via GitHub REST APIs to answer *"What changed in the latest commit?"*.
- 📚 **Unified Source Citations**: Deduplicated, interactive source chips with direct line ranges, file type badges, and GitHub links.
- 🕸️ **Repository Pipeline Graph**: Visual architecture graph rendering confirmed component flows (`train.py` ➔ `realtime.py` ➔ `capture.py` / `classifier.py`).
- 💬 **Smart Conversational Handler**: Zero-latency instant responses for pure greetings (`hello`, `hi`, `thanks`) without triggering unnecessary RAG retrieval.
- 🛡️ **8 Free Shared Questions**: Shared daily query quota powered by Groq, plus a seamless **Bring Your Own Key (BYO Key)** option for multi-provider support (OpenAI, Gemini, Claude, Mistral, DeepSeek, etc.).
- 📱 **Fully Responsive 3D Cockpit**: Translucent glass surfaces, custom CSS micro-interactions, and adaptive layouts for desktop, tablet, and mobile displays.

---

## 🏗️ Architecture & RAG Pipeline

```
┌─────────────────┐       ┌────────────────────┐       ┌──────────────────────┐
│  GitHub Repo    │ ────> │  Clone & Chunking  │ ────> │  BGE Vector Model    │
└─────────────────┘       └────────────────────┘       └──────────┬───────────┘
                                                                  │
┌─────────────────┐       ┌────────────────────┐       ┌──────────▼───────────┐
│   User Query    │ ────> │ Hybrid Search      │ <──── │ BM25 + FAISS Index   │
└────────┬────────┘       │ (RRF Re-ranking)   │       └──────────────────────┘
         │                └─────────┬──────────┘
         │                          │ Top-K Context
         ▼                          ▼
┌──────────────────────────────────────────────┐
│       LLM Generation (Groq / BYO Key)        │
└──────────────────────┬───────────────────────┘
                       │ Streaming Tokens
                       ▼
┌──────────────────────────────────────────────┐
│   Repo Buddy UI (Markdown + Source Chips)    │
└──────────────────────────────────────────────┘
```

1. **Repository Ingestion & Chunking**: Safe shallow clone (`git clone --depth 1`) with multi-file extension filtering (`.py`, `.js`, `.ts`, `.java`, `.go`, `.md`, `.toml`, etc.).
2. **Dual Indexing**:
   - **Dense Vectors**: `BAAI/bge-small-en-v1.5` embeddings stored in an in-memory `FAISS` index.
   - **Lexical Tokens**: `rank_bm25` index for exact code symbol and function name matches.
3. **Hybrid RAG & RRF**: Reciprocal Rank Fusion combines top candidates from both search algorithms for optimal context precision.
4. **Contextual LLM Generation**: Prompts Groq models (`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`) or user-supplied LLMs with retrieved context and chat history.

---

## 🛠️ Tech Stack

- **Frontend / Framework**: Streamlit, Custom HTML5/CSS3 (3D Glassmorphic Design System)
- **RAG & Search**: FAISS (`faiss-cpu`), `rank_bm25`, PyTorch, HuggingFace `transformers` (`BAAI/bge-small-en-v1.5`)
- **LLM API Provider**: Groq API (`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`) via official `openai` Python SDK
- **GitHub API**: REST v3 API (`urllib`) for commit, PR, and issue metadata retrieval
- **Language**: Python 3.10+

---

## 📁 Project Structure

```
D:/rag system/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies for deployment
├── README.md                   # Project documentation
├── .gitignore                  # Git exclusions for secrets, caches, and logs
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── generation/
│   └── llm.py                  # Prompt templates, token streaming, and overview generators
├── ingestion/
│   ├── chunker.py              # File, commit, issue, and PR chunking logic
│   ├── github_api.py           # GitHub REST API client for commits & diffs
│   └── repository.py           # Git cloning, file tree parsing, and security limits
├── retrieval/
│   ├── embeddings.py           # BAAI/bge-small-en-v1.5 PyTorch embedding model
│   └── hybrid_search.py        # Hybrid FAISS + BM25 + RRF retrieval engine
└── utils/
    └── github_links.py         # Link generator for line-level GitHub source URLs
```

---

## 💻 Local Installation

### Prerequisites
- Python 3.10 or higher
- Git installed on your system

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/repo-buddy.git
   cd repo-buddy
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variable** (Optional for local testing):
   ```bash
   # On Windows PowerShell:
   $env:GROQ_API_KEY="your_groq_api_key_here"

   # On Linux/macOS:
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

5. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

6. Open your browser at `http://localhost:8501`.

---

## 🔑 Streamlit Secrets Configuration

When deploying on **Streamlit Community Cloud**, configure your shared API key in your app's **Secrets**:

1. Go to your app dashboard on Streamlit Community Cloud.
2. Click **Settings** ➔ **Secrets**.
3. Paste the following configuration:

```toml
GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
```

---

## ☁️ Streamlit Community Cloud Deployment

1. Push your repository code to GitHub (ensure secrets are ignored by `.gitignore`).
2. Log in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **New app**.
4. Select your GitHub repository, branch (`main`), and set the main file path to:
   ```
   app.py
   ```
5. Add your `GROQ_API_KEY` under **Advanced Settings** ➔ **Secrets**.
6. Click **Deploy!**

---

## 💡 Example Questions

- `Give me an overview of this repository`
- `Where is packet capture implemented?`
- `How does a live packet flow through the system?`
- `What changed in the latest commit?`
- `Explain the anomaly detection autoencoder model.`

---

## 🔒 Security & Privacy Notes

- **API Key Protection**: API keys and tokens are **never** logged, printed, or committed to Git repository history.
- **Repository Safety**: Only public GitHub repositories are cloned into temporary isolated directories and purged immediately after indexing.
- **BYO Key Security**: User-entered API keys are stored strictly in-memory within the visitor's private Streamlit session state and are never persisted to disk or server logs.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
