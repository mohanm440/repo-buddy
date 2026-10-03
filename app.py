"""
Chat web app for the GitHub Repo Q&A assistant (complete version).

Features:
  - Hybrid search (BM25 + FAISS) and follow-up question rewriting
  - Rewrite step is skipped for self-contained questions (faster)
  - FREE_QUESTIONS questions with the shared Groq key (GROQ_API_KEY secret)
  - After that, visitors use THEIR OWN key. The Model dropdown is loaded
    live from the provider, so it always shows what is available today.
  - Hides <thought> reasoning text from thinking models
  - Fails fast and shows friendly messages for busy / rate-limited models

Run:
    pip install streamlit rank-bm25
    streamlit run app.py
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi

LLM_MODEL = "openai/gpt-oss-20b"
MAX_CHUNKS = 4000
import utils.github_links as gl
from ingestion.repository import clone_repo_safe, iter_repo_files, RepoError
from ingestion.github_api import fetch_commits, fetch_issues, fetch_pull_requests
from ingestion.chunker import chunk_file, chunk_commit, chunk_issue, chunk_pr
from retrieval.embeddings import Embedder
from retrieval.hybrid_search import HybridRetriever, tokenize
from generation.llm import stream_answer, rewrite_question, generate_repository_overview, stream_repository_overview


MAX_CHUNKS = 4000
CANDIDATES = 20
HISTORY_TURNS = 3     # fewer past messages = fewer tokens
FREE_QUESTIONS = 8      # protects the shared key's rate limit
ANSWER_TOP_K = 5        # chunks sent to the LLM (fewer = fewer tokens)
GITHUB_URL = re.compile(r"^https://github\.com/[\w.\-]+/[\w.\-]+?(\.git)?/?$")
OTHER = "Other (type your own)"
MAX_CACHED_REPOS = 3   # indexed repos kept in memory per visitor session
CLONE_TIMEOUT = 120    # seconds before a clone is abandoned
MAX_REPO_MB = 150      # repos bigger than this are rejected
DAILY_FREE_CAP = 150   # shared-key questions per day across ALL visitors
INDEX_TTL = 6 * 3600   # how long a shared index is reused (seconds)
SHARED_CACHE_SIZE = 5  # repos kept in the cache shared by all visitors
SAMPLE_QUESTIONS = [
    "Give me an overview of this repository",
    "Where is packet capture implemented?",
    "How does X flow through the system?",
    "What changed in the latest commit?",
    "How does a live packet move through the system?",
]
LANG = {".py": "python", ".ipynb": "python", ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".go": "go",
        ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp",
        ".rb": "ruby", ".php": "php", ".sh": "bash", ".md": "markdown",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
        ".sql": "sql", ".html": "html", ".css": "css"}

# Fallback lists, used only until a key is entered (or if listing fails).
PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    },
    "Anthropic (Claude)": {
        "base_url": "https://api.anthropic.com/v1/",
        "models": ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-4-5-20251001"],
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["openai/gpt-4o-mini", "google/gemini-2.5-flash"],
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest", "mistral-large-latest"],
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "xAI (Grok)": {"base_url": "https://api.x.ai/v1", "models": []},
    "Together AI": {"base_url": "https://api.together.xyz/v1", "models": []},
    "Fireworks AI": {"base_url": "https://api.fireworks.ai/inference/v1", "models": []},
    "Cerebras": {"base_url": "https://api.cerebras.ai/v1", "models": []},
    "SambaNova": {"base_url": "https://api.sambanova.ai/v1", "models": []},
    "NVIDIA NIM": {"base_url": "https://integrate.api.nvidia.com/v1", "models": []},
    "Perplexity": {"base_url": "https://api.perplexity.ai", "models": []},
    "Hugging Face": {"base_url": "https://router.huggingface.co/v1", "models": []},
    "Custom (OpenAI-compatible)": {"base_url": "", "models": []},
}

# Words that mark models that cannot answer chat questions.
NON_CHAT = ("embed", "whisper", "tts", "transcribe", "dall-e", "image", "imagen",
            "veo", "moderation", "guard", "audio", "realtime", "speech", "aqa")

# Words that suggest a question depends on the previous messages.
FOLLOWUP_WORDS = {"it", "that", "this", "they", "them", "those", "these", "its",
                  "above", "previous", "second", "first", "third", "same", "more"}

st.set_page_config(page_title="Repo Buddy", page_icon="🤖", layout="wide")
st.markdown("""<style>
/* ── 3D Space Background & Cyber Grid ──────────────────────── */
html,body,[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #09090e 60%, #030305 100%)!important;
  background-attachment: fixed!important;
}
[data-testid="stAppViewContainer"]::before{
  content: "";
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image: 
    linear-gradient(rgba(124, 58, 237, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(124, 58, 237, 0.035) 1px, transparent 1px);
  background-size: 38px 38px;
  pointer-events: none;
  z-index: 0;
}
footer, [data-testid="stFooter"], .viewerBadge_container__1QS-Z, [data-testid="stStatusWidget"], [data-testid="stDecoration"]{
  display: none!important;
  visibility: hidden!important;
  height: 0!important;
  width: 0!important;
  opacity: 0!important;
  pointer-events: none!important;
}

/* ── 3D Glass Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"]{
  background: rgba(12, 12, 18, 0.75)!important;
  backdrop-filter: blur(20px)!important;
  border-right: 1px solid rgba(139, 92, 246, 0.18)!important;
  min-width: 275px!important;
  max-width: 295px!important;
  box-shadow: 10px 0 35px rgba(0,0,0,0.55)!important;
}
[data-testid="stSidebarContent"]{padding:0.65rem 0.8rem}
[data-testid="stSidebar"] .stButton button{
  background: rgba(20, 20, 32, 0.5);
  border: 1px solid rgba(139, 92, 246, 0.12);
  color: #c0c0d5;
  justify-content: flex-start;
  text-align: left;
  font-size: .82rem;
  padding: .4rem .6rem;
  border-radius: 7px;
  width: 100%;
  transition: all .2s ease;
}
[data-testid="stSidebar"] .stButton button:hover{
  background: rgba(124, 58, 237, 0.2);
  border-color: rgba(167, 139, 250, 0.4);
  color: #fff;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
}
[data-testid="stSidebar"] .stButton button[kind="primary"]{
  background: linear-gradient(135deg, #7c3aed, #06b6d4)!important;
  color: #fff!important;
  border: none!important;
  border-radius: 8px!important;
  font-weight: 600;
  box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4)!important;
}
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover{
  transform: translateY(-2px)!important;
  box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5)!important;
}
[data-testid="stSidebar"] button:disabled{opacity:.3}
[data-testid="stSidebar"] .stTextInput input{
  background: rgba(18, 18, 28, 0.7);
  border: 1px solid rgba(139, 92, 246, 0.2);
  border-radius: 7px;
  color: #e0e0ec;
  font-size: .82rem;
  padding: .4rem .6rem;
}
[data-testid="stSidebar"] .stTextInput input:focus{border-color: #06b6d4; box-shadow: 0 0 10px rgba(6, 182, 212, 0.3);}
[data-testid="stSidebar"] .stSelectbox>div>div{background: rgba(18, 18, 28, 0.7); border: 1px solid rgba(139, 92, 246, 0.2); border-radius: 7px; color: #e0e0ec; font-size: .82rem;}
[data-testid="stSidebar"] label{color: #8b5cf6; font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .2rem;}
[data-testid="stSidebar"] .stCaption,.stCaption{color: #71717a; font-size: .74rem;}
[data-testid="stSidebar"] hr{border-color: rgba(139, 92, 246, 0.15); margin: .5rem 0;}
[data-testid="stSidebarCollapseButton"] button{opacity: .6; background: transparent; border: none;}
[data-testid="stSidebar"] [data-testid="stPopover"] button,[data-testid="stSidebar"] [data-testid="stPopoverButton"]{
  background: transparent; border: none; box-shadow: none; color: #c0c0d5; font-size: .82rem; border-radius: 6px;
}
[data-testid="stSidebar"] [data-testid="stPopover"] button:hover,[data-testid="stSidebar"] [data-testid="stPopoverButton"]:hover{background: rgba(124, 58, 237, 0.15);}

/* ── Sidebar Top Navigation Header Controls (Horizontal Row) ─ */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 0.35rem !important;
  width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div {
  width: 33.33% !important;
  min-width: 0 !important;
  flex: 1 1 33.33% !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stPopover"] {
  width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button {
  justify-content: center !important;
  text-align: center !important;
  min-height: 44px !important;
}

/* ── 3D Statistic Cards ────────────────────────────────────── */
.stat-cards-grid{display:grid;grid-template-columns:repeat(3, 1fr);gap:.4rem;margin-top:.4rem}
.stat-card-3d{
  background: rgba(20, 20, 32, 0.75);
  border: 1px solid rgba(139, 92, 246, 0.22);
  border-radius: 10px;
  padding: .45rem .2rem;
  text-align: center;
  transition: all .25s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.stat-card-3d:hover{
  transform: translateY(-4px) translateZ(8px);
  border-color: rgba(167, 139, 250, 0.5);
  box-shadow: 0 8px 22px rgba(124, 58, 237, 0.28);
}
.stat-val{font-size:1.05rem;font-weight:800;color:#a78bfa;line-height:1.1}
.stat-lbl{font-size:.6rem;font-weight:700;letter-spacing:.08em;color:#71717a;margin-top:2px}

/* ── 3D Holographic Header ─────────────────────────────────── */
.rb-3d-header{
  display: flex;
  align-items: center;
  gap: 1.1rem;
  padding: .75rem 1.2rem;
  background: rgba(15, 15, 24, 0.65);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(139, 92, 246, 0.22);
  border-radius: 16px;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  margin-bottom: 1.4rem;
  position: relative;
  overflow: hidden;
}
.rb-orb-container{position:relative;width:46px;height:46px;display:flex;align-items:center;justify-content:center}
.rb-hologram-orb{
  position: absolute;
  width: 42px;
  height: 42px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(124, 58, 237, 0.65) 0%, rgba(6, 182, 212, 0.25) 60%, transparent 80%);
  filter: blur(5px);
  animation: floatOrb 3s ease-in-out infinite alternate;
}
.rb-logo-3d{font-size:1.75rem;z-index:1;text-shadow:0 0 15px rgba(124, 58, 237, 0.8)}
.rb-title-3d{
  font-size: 1.3rem;
  font-weight: 900;
  letter-spacing: .06em;
  background: linear-gradient(135deg, #ffffff 30%, #a78bfa 70%, #06b6d4 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  margin: 0;
  text-transform: uppercase;
}
.rb-subtitle-3d{font-size:.68rem;font-weight:600;letter-spacing:.1em;color:#71717a;margin:0;text-transform:uppercase}
.rb-status{margin-left:auto;display:flex;align-items:center;gap:.4rem;font-size:.74rem;color:#3ecf8e;font-weight:600;background:#0d1f17;border:1px solid #163e2c;padding:.2rem .65rem;border-radius:12px;box-shadow:0 0 12px rgba(62,207,142,0.2)}
.rb-status-dot{width:6px;height:6px;border-radius:50%;background:#3ecf8e;animation:pulseDot 2s infinite}
.rb-status-inactive{color:#71717a;background:#14141c;border-color:#20202e;box-shadow:none}
.rb-status-inactive .rb-status-dot{background:#404058;animation:none}
@keyframes floatOrb{0%{transform:translateY(-3px) scale(0.95);opacity:0.75}100%{transform:translateY(3px) scale(1.05);opacity:1}}
@keyframes pulseDot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(0.85)}}

/* ── Content & Markdown Typography ─────────────────────────── */
.stMarkdown h1{font-size:1.55rem!important;margin-top:1.1rem!important;margin-bottom:.4rem!important;color:#f4f4f5!important}
.stMarkdown h2{font-size:1.25rem!important;margin-top:1rem!important;margin-bottom:.35rem!important;color:#e4e4e7!important}
.stMarkdown h3{font-size:1.05rem!important;margin-top:.8rem!important;margin-bottom:.3rem!important;color:#c7d2fe!important}
.stMarkdown h4,.stMarkdown h5{font-size:.92rem!important;margin-top:.6rem!important;margin-bottom:.25rem!important;color:#a5b4fc!important}
.stMarkdown p,.stMarkdown li{font-size:.9rem!important;line-height:1.55!important;color:#d1d5db!important}
.stMarkdown p{margin-bottom:.45rem!important}
.stMarkdown ul,.stMarkdown ol{margin-bottom:.5rem!important;padding-left:1.3rem!important}

/* ── Tables & Horizontal Scroll ────────────────────────────── */
.stMarkdown table{display:block!important;overflow-x:auto!important;white-space:nowrap!important;max-width:100%!important;border-collapse:collapse!important;margin:.8rem 0!important;border-radius:8px!important;border:1px solid rgba(139, 92, 246, 0.25)!important;box-shadow:0 4px 15px rgba(0,0,0,0.3)!important}
.stMarkdown table th{background:rgba(20, 20, 32, 0.9)!important;color:#a78bfa!important;font-size:.82rem!important;font-weight:600!important;padding:.45rem .75rem!important;border-bottom:1px solid rgba(139, 92, 246, 0.25)!important}
.stMarkdown table td{padding:.4rem .75rem!important;font-size:.82rem!important;border-bottom:1px solid rgba(139, 92, 246, 0.12)!important;color:#d1d5db!important}
.stMarkdown table tr:last-child td{border-bottom:none!important}

/* ── Floating 3D Message Surfaces ─────────────────────────── */
[data-testid="stChatMessage"]{background:transparent!important;border:none!important;padding:0!important;margin-bottom:.2rem}
[data-testid="stChatMessageAvatarAssistant"]{display:none}
[data-testid="stChatMessageAvatarUser"]{display:none}
.user-bubble{
  background: linear-gradient(135deg, rgba(46, 16, 101, 0.85), rgba(76, 29, 149, 0.85));
  border: 1px solid rgba(167, 139, 250, 0.35);
  border-radius: 14px 14px 2px 14px;
  padding: .6rem 1.05rem;
  margin: .55rem 0 .55rem auto;
  max-width: 82%;
  width: fit-content;
  color: #ede9fe;
  font-size: .88rem;
  line-height: 1.5;
  box-shadow: 0 6px 20px rgba(124, 58, 237, 0.25);
}
.asst-card-3d{
  background: rgba(15, 15, 24, 0.75);
  backdrop-filter: blur(14px);
  border: 1px solid rgba(139, 92, 246, 0.2);
  border-radius: 14px;
  padding: 1rem 1.25rem;
  margin-bottom: .9rem;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.45);
  transition: border-color .2s ease;
}
.asst-card-3d:hover{border-color:rgba(139, 92, 246, 0.38);}

/* ── 3D Interactive Repository Graph Panel ─────────────────── */
.repo-graph-container{
  display: flex;
  align-items: center;
  gap: .6rem;
  overflow-x: auto;
  padding: .7rem .4rem;
}
.graph-node{
  background: rgba(20, 20, 32, 0.85);
  border: 1px solid rgba(139, 92, 246, 0.3);
  border-radius: 10px;
  padding: .5rem .75rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 125px;
  transition: all .25s ease;
  box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}
.graph-node:hover{
  transform: translateY(-4px) scale(1.03);
  border-color: #06b6d4;
  box-shadow: 0 8px 25px rgba(6, 182, 212, 0.3);
}
.graph-node.ml-node{border-color:rgba(6, 182, 212, 0.4);background:rgba(8, 30, 42, 0.85)}
.graph-node.ui-node{border-color:rgba(167, 139, 250, 0.4);background:rgba(30, 18, 54, 0.85)}
.node-icon{font-size:1.15rem;margin-bottom:2px}
.node-title{font-size:.72rem;font-weight:700;color:#e4e4e7}
.node-tag{font-size:.6rem;color:#9ca3af;text-transform:uppercase;margin-top:2px}
.graph-arrow{color:#8b5cf6;font-size:1.1rem;font-weight:bold;opacity:.7}

/* ── 3D Source Chips ───────────────────────────────────────── */
.src-chip{
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  background: rgba(20, 20, 32, 0.75);
  border: 1px solid rgba(139, 92, 246, 0.25);
  border-radius: 8px;
  padding: .26rem .6rem;
  margin: .25rem .25rem 0 0;
  font-size: .75rem;
  color: #a78bfa;
  text-decoration: none;
  cursor: pointer;
  transition: all .2s ease;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.src-chip:hover{
  transform: translateY(-3px) translateZ(4px);
  border-color: #06b6d4;
  color: #67e8f9;
  box-shadow: 0 6px 18px rgba(6, 182, 212, 0.25);
}
.src-chip .src-icon{font-size:.85rem}
.src-chip .src-label{font-weight:600;color:#e4e4e7}
.src-chip .src-lines{color:#9ca3af;font-size:.7rem}
.src-chips-wrap{margin-top:.45rem}
.unified-src-header{
  display: flex;
  align-items: center;
  gap: .55rem;
  padding: .4rem .65rem;
  background: rgba(20, 20, 32, 0.85);
  border: 1px solid rgba(139, 92, 246, 0.22);
  border-radius: 7px;
  margin-top: .6rem;
  margin-bottom: .3rem;
  font-size: .8rem;
}
.unified-src-header .src-icon{font-size:.9rem}
.unified-src-header .src-title{font-weight:600;color:#e4e4e7}
.unified-src-header .src-detail{color:#9ca3af;font-size:.73rem;margin-left:auto}
.unified-src-header .src-link{
  color:#a78bfa;
  font-weight:600;
  font-size:.72rem;
  text-decoration:none;
  background:rgba(124, 58, 237, 0.15);
  border:1px solid rgba(124, 58, 237, 0.3);
  padding:.15rem .5rem;
  border-radius:4px;
  transition:all .15s ease;
}
.unified-src-header .src-link:hover{
  background:rgba(6, 182, 212, 0.2);
  border-color:#06b6d4;
  color:#67e8f9;
}

/* ── Suggestion Cards ─────────────────────────────────────── */
.suggest-wrap{margin:2rem auto 0;max-width:640px;text-align:center}
.suggest-title{font-size:1.65rem;font-weight:800;color:#f4f4f5;margin-bottom:.2rem}
.suggest-sub{color:#9ca3af;font-size:.88rem;margin-bottom:1.6rem}

/* ── Repo Status Card ─────────────────────────────────────── */
.repo-card{
  background: rgba(18, 18, 28, 0.75);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(139, 92, 246, 0.25);
  border-radius: 12px;
  padding: .6rem .8rem;
  margin: .5rem 0;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
  transition: all .25s ease;
}
.repo-card:hover{
  transform: translateY(-2px) translateZ(4px);
  border-color: rgba(6, 182, 212, 0.5);
  box-shadow: 0 12px 28px rgba(124, 58, 237, 0.25);
}
.repo-card-name{font-size:.8rem;font-weight:600;color:#c7d2fe;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:flex;align-items:center;gap:.4rem}
.repo-card-stats{font-size:.7rem;color:#9ca3af;margin-top:.2rem}
.sb-section{font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#8b5cf6;padding:.45rem 0 .2rem;margin-top:.2rem}

/* ── 3D Floating Composer & Touch-Safe Layers ──────────────── */
@media screen and (min-width: 769px) {
  .block-container {
    padding-bottom: 11.5rem!important;
  }
  [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    position: fixed!important;
    bottom: 0!important;
    left: 295px!important;
    right: 0!important;
    width: calc(100vw - 295px)!important;
    z-index: 9999!important;
    background: transparent!important;
    pointer-events: none!important;
    padding-bottom: .8rem!important;
    padding-top: .4rem!important;
  }
}
[data-testid="stChatInput"]{
  position: relative!important;
  z-index: 10000!important;
  pointer-events: auto!important;
  background: rgba(15, 15, 25, 0.88)!important;
  backdrop-filter: blur(20px)!important;
  border: 1px solid rgba(139, 92, 246, 0.35)!important;
  border-radius: 16px!important;
  box-shadow: 0 10px 35px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.1)!important;
  transition: border-color .2s ease, box-shadow .2s ease;
}
[data-testid="stChatInput"]:focus-within{
  border-color: rgba(167, 139, 250, 0.6)!important;
  box-shadow: 0 12px 40px rgba(124, 58, 237, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15)!important;
}
[data-testid="stChatInput"] *{border-color:transparent!important;box-shadow:none!important}
[data-testid="stChatInput"]>div,[data-testid="stChatInput"] textarea{background:transparent!important}
[data-testid="stChatInput"] textarea{font-size:.88rem;color:#e4e4e7}
[data-testid="stChatInput"] textarea::placeholder{color:#71717a}
[data-testid="stChatInputSubmitButton"]{
  position: relative!important;
  z-index: 10002!important;
  pointer-events: auto!important;
  background: linear-gradient(135deg, #7c3aed, #06b6d4)!important;
  color: #fff!important;
  border-radius: 50%!important;
  min-width: 38px!important;
  min-height: 38px!important;
  width: 38px!important;
  height: 38px!important;
  box-shadow: 0 0 14px rgba(124, 58, 237, 0.5)!important;
  touch-action: manipulation!important;
  cursor: pointer!important;
}
[data-testid="stChatInputSubmitButton"]:disabled{background:#1f1f2e!important;color:#52525b!important;box-shadow:none!important}

/* ── 3D Button Hover Interactions ──────────────────────────── */
.stButton>button{
  border-radius: 7px;
  font-size: .82rem;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1)!important;
}
.stButton>button:hover{
  transform: translateY(-2px)!important;
  border-color: rgba(167, 139, 250, 0.5)!important;
  box-shadow: 0 4px 14px rgba(124, 58, 237, 0.3)!important;
}
.stButton>button:active{
  transform: translateY(0)!important;
  box-shadow: none!important;
}

/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"]{background:rgba(14, 14, 22, 0.65);border:1px solid rgba(139, 92, 246, 0.18);border-radius:8px}
[data-testid="stExpander"] summary{font-size:.76rem;color:#a1a1aa}

/* ── Bot animations ───────────────────────────────────────── */
.bot-row{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.bot-label{font-size:.82rem;color:#555570}
.bot{flex:none;overflow:visible}
.bot .eye,.bot .eyes{transform-box:fill-box;transform-origin:center}
.bot .eye,.bot .ant,.bot .ear,.bot .fill{fill:#5be7ff}
.bot .cheek{fill:#b25cff;opacity:.55}
.bot .stroke{fill:none;stroke:#5be7ff;stroke-width:3;stroke-linecap:round}
.bot .eye-happy,.bot .eye-x,.bot .m-talk,.bot .m-sad,.bot .m-dots{display:none}
.bot.happy .eye,.bot.error .eye{display:none}
.bot.happy .eye-happy,.bot.error .eye-x{display:block}
.bot.thinking .m-smile,.bot.answering .m-smile,.bot.error .m-smile{display:none}
.bot.thinking .m-dots,.bot.answering .m-talk,.bot.error .m-sad{display:block}
.bot.idle .eye{animation:blink 4s infinite}
.bot.searching .eyes{animation:scan 1.2s ease-in-out infinite}
.bot.thinking .eyes{transform:translateY(-4px)}
.bot.answering .eye{animation:blink 3s infinite}
.bot.answering .m-talk{transform-box:fill-box;transform-origin:center;animation:talk .35s ease-in-out infinite}
.bot.thinking .m-dots circle{animation:dot 1s infinite}
.bot.thinking .m-dots circle:nth-child(2){animation-delay:.2s}
.bot.thinking .m-dots circle:nth-child(3){animation-delay:.4s}
.bot.searching .ant,.bot.thinking .ant,.bot.answering .ant{animation:glow 1s infinite}
.bot.happy{animation:hop .6s ease 1}
.bot.error{animation:shake .4s ease 2}
@keyframes blink{0%,92%,100%{transform:scaleY(1)}95%{transform:scaleY(.1)}}
@keyframes scan{0%,100%{transform:translateX(-5px)}50%{transform:translateX(5px)}}
@keyframes talk{0%,100%{transform:scaleY(.4)}50%{transform:scaleY(1.7)}}
@keyframes dot{0%,100%{opacity:.25}50%{opacity:1}}
@keyframes glow{0%,100%{opacity:1}50%{opacity:.25}}
@keyframes hop{0%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}
@keyframes shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-4px)}75%{transform:translateX(4px)}}
@media (prefers-reduced-motion:reduce){.bot,.bot *{animation:none!important}}

/* ── Repository Navigation Bar (TOP) ───────────────────────── */
header[data-testid="stHeader"], [data-testid="stHeader"] {
  position: sticky!important;
  top: 0!important;
  left: 0!important;
  right: 0!important;
  z-index: 1000!important;
  background: rgba(9, 9, 14, 0.95)!important;
  backdrop-filter: blur(16px)!important;
  border-bottom: 1px solid rgba(139, 92, 246, 0.2)!important;
  pointer-events: auto!important;
}

/* ── Mobile Responsive Rules (viewport <= 768px) ────────────── */
@media screen and (max-width: 768px) {
  header[data-testid="stHeader"], [data-testid="stHeader"] {
    position: sticky!important;
    top: 0!important;
    left: 0!important;
    right: 0!important;
    z-index: 1000!important;
    background: rgba(9, 9, 14, 0.95)!important;
    backdrop-filter: blur(16px)!important;
    pointer-events: auto!important;
  }
  .block-container {
    max-width: 100%!important;
    padding: .8rem .75rem 12.5rem!important;
    margin: 0!important;
  }
  [data-testid="stSidebar"] {
    min-width: 82vw!important;
    max-width: 82vw!important;
    box-shadow: 15px 0 40px rgba(0, 0, 0, 0.7)!important;
    z-index: 9998!important;
  }
  .rb-3d-header {
    padding: .55rem .8rem;
    gap: .7rem;
    margin-bottom: 1rem;
    border-radius: 12px;
  }
  .rb-orb-container { width: 36px; height: 36px; }
  .rb-hologram-orb { width: 34px; height: 34px; pointer-events: none!important; }
  .rb-logo-3d { font-size: 1.4rem; }
  .rb-title-3d { font-size: 1.05rem; }
  .rb-subtitle-3d { font-size: .6rem; }
  .rb-status { font-size: .68rem; padding: .18rem .45rem; }
  
  .user-bubble {
    max-width: 90%!important;
    padding: .5rem .8rem!important;
    font-size: .84rem!important;
  }
  .asst-card-3d {
    padding: .85rem .95rem!important;
    border-radius: 10px!important;
  }
  
  .stMarkdown h1 { font-size: 1.35rem!important; }
  .stMarkdown h2 { font-size: 1.15rem!important; }
  .stMarkdown h3 { font-size: 1.0rem!important; }
  .stMarkdown p, .stMarkdown li { font-size: .86rem!important; }
  
  .stat-cards-grid { gap: .25rem!important; }
  .stat-card-3d { padding: .35rem .15rem!important; }
  .stat-val { font-size: .92rem!important; }
  .stat-lbl { font-size: .55rem!important; }
  
  .repo-graph-container, .repo-graph-tree {
    overflow-x: auto!important;
    padding: .5rem .2rem!important;
  }
  .graph-node { min-width: 105px!important; padding: .4rem .55rem!important; }
  .node-title { font-size: .65rem!important; }
  
  [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    position: fixed!important;
    bottom: calc(3.5rem + env(safe-area-inset-bottom))!important;
    left: 0!important;
    right: 0!important;
    padding-bottom: 0!important;
    padding-left: .5rem!important;
    padding-right: .5rem!important;
    pointer-events: none!important;
  }
  [data-testid="stChatInput"] {
    pointer-events: auto!important;
    width: 100%!important;
    max-width: 100%!important;
    margin: 0!important;
  }
  [data-testid="stChatInput"] textarea {
    font-size: .86rem!important;
  }
  [data-testid="stChatInput"] button, [data-testid="stChatInputSubmitButton"] {
    pointer-events: auto!important;
    touch-action: manipulation!important;
  }
  
  .landing-hero-card { padding: 1.8rem 1rem 1.4rem!important; }
  .landing-title { font-size: 1.5rem!important; }
  .landing-tagline { font-size: 1.0rem!important; }
  .landing-sub { font-size: .82rem!important; }
}

@media screen and (max-width: 480px) {
  .rb-subtitle-3d { display: none; }
  .unified-src-header .src-detail { display: none; }
}
</style>""", unsafe_allow_html=True)



def robot(state="still", label="", size=64):
    """Animated robot. state: still, idle, searching, thinking, answering, happy, error."""
    w, h = size, size * 100 / 120
    svg = (
        f'<svg class="bot {state}" viewBox="0 0 120 100" width="{w:.0f}" height="{h:.0f}">'
        '<line x1="60" y1="12" x2="60" y2="4" stroke="#cfd8e3" stroke-width="3"/>'
        '<circle class="ant" cx="60" cy="4" r="3.5"/>'
        '<rect x="4" y="36" width="12" height="28" rx="6" fill="#dfe6ef"/>'
        '<rect x="104" y="36" width="12" height="28" rx="6" fill="#dfe6ef"/>'
        '<circle class="ear" cx="10" cy="50" r="3"/><circle class="ear" cx="110" cy="50" r="3"/>'
        '<rect x="12" y="12" width="96" height="78" rx="32" fill="#f3f6fb"/>'
        '<rect x="22" y="24" width="76" height="54" rx="22" fill="#0e1118"/>'
        '<g class="eyes"><ellipse class="eye" cx="46" cy="48" rx="6.5" ry="8"/>'
        '<ellipse class="eye" cx="74" cy="48" rx="6.5" ry="8"/></g>'
        '<path class="stroke eye-happy" d="M39 53 Q46 42 53 53 M67 53 Q74 42 81 53"/>'
        '<path class="stroke eye-x" d="M40 42 L52 54 M52 42 L40 54 M68 42 L80 54 M80 42 L68 54"/>'
        '<circle class="cheek" cx="34" cy="62" r="3.5"/><circle class="cheek" cx="86" cy="62" r="3.5"/>'
        '<path class="stroke m-smile" d="M52 62 Q60 69 68 62"/>'
        '<ellipse class="fill m-talk" cx="60" cy="64" rx="5" ry="3.5"/>'
        '<path class="stroke m-sad" d="M52 68 Q60 60 68 68"/>'
        '<g class="m-dots"><circle class="fill" cx="52" cy="64" r="2.2"/>'
        '<circle class="fill" cx="60" cy="64" r="2.2"/><circle class="fill" cx="68" cy="64" r="2.2"/></g>'
        '</svg>'
    )
    text = f'<span class="bot-label">{label}</span>' if label else ""
    return f'<div class="bot-row">{svg}{text}</div>'


def relay(chunks, status):
    """Pass answer text through, and make the robot start 'typing' on the first word."""
    first = True
    for piece in chunks:
        if first and piece:
            status.markdown(robot("answering", "Typing…"), unsafe_allow_html=True)
            first = False
        yield piece


# ---------- Helpers ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder() -> Embedder:
    return Embedder()



def list_models(api_key: str, base_url: str):
    """Ask the provider which models this key can use (kept in the session only)."""
    cache = st.session_state.setdefault("model_cache", {})
    k = (base_url, api_key)
    if k not in cache:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url,
                            max_retries=0, timeout=20)
            cache[k] = sorted(
                m.id.removeprefix("models/") for m in client.models.list().data
            )
        except Exception:
            cache[k] = []
    return cache[k]


def get_llm(user_key: str, base_url: str, model: str):
    """Return (client, model). Uses the visitor's key if given, else the shared key."""
    s = st.session_state
    if user_key:
        if not base_url.startswith("https://"):
            st.error("Base URL must start with https://")
            st.stop()
        if not model:
            st.error("Enter a model name in the sidebar.")
            st.stop()
        return OpenAI(api_key=user_key, base_url=base_url,
                      max_retries=1, timeout=60), model

    shared_key = os.environ.get("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets else "")
    if not shared_key:
        st.error("No shared key is configured. Paste your own API key in the sidebar.")
        st.stop()
    if s.q_count >= FREE_QUESTIONS:
        st.warning(
            f"Free limit of {FREE_QUESTIONS} questions reached. "
            "Paste your own API key (Groq, OpenAI, Gemini, ...) in the sidebar to continue."
        )
        st.stop()
    if not take_free_slot():
        st.warning("The shared free key has reached today's limit for all visitors. "
                   "Paste your own API key in the sidebar to continue, or come back tomorrow.")
        st.stop()
    s.q_count += 1
    return OpenAI(api_key=shared_key, base_url=PROVIDERS["Groq"]["base_url"],
                  max_retries=1, timeout=60), LLM_MODEL


# ---------- Build ----------




def build_repo_index(url: str, progress=None, gh_token: str = None):
    def report(frac, msg):
        if progress:
            progress(frac, msg)

    report(0.05, "Cloning repository…")
    root = clone_repo_safe(url)
    report(0.10, "Reading files…")
    try:
        chunks, files = [], 0
        for rel_path, text in iter_repo_files(root):
            files += 1
            chunks.extend(chunk_file(url, rel_path, text))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if not chunks:
        raise RepoError("No readable code or text files were found in this repository.")

    report(0.30, "Fetching Commits…")
    commits = fetch_commits(url, gh_token, limit=50)
    for c in commits: chunks.extend(chunk_commit(url, c))

    report(0.35, "Fetching Issues…")
    issues = fetch_issues(url, gh_token, limit=50)
    for i in issues: chunks.extend(chunk_issue(url, i))

    report(0.40, "Fetching Pull Requests…")
    prs = fetch_pull_requests(url, gh_token, limit=50)
    for pr in prs: chunks.extend(chunk_pr(url, pr))

    truncated = len(chunks) > MAX_CHUNKS
    if truncated:
        chunks = chunks[:MAX_CHUNKS]

    texts = [c["text"] for c in chunks]
    report(0.45, "Loading the AI model (the first time after starting is slower)…")
    embedder, parts, batch = get_embedder(), [], 64
    for i in range(0, len(texts), batch):
        parts.append(embedder.encode(texts[i:i + batch]))
        done = min(i + batch, len(texts))
        report(0.45 + 0.40 * done / len(texts), f"Embedding {done}/{len(texts)} chunks…")
    vectors = np.vstack(parts)
    import faiss
    from rank_bm25 import BM25Okapi
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    report(0.95, "Building keyword index…")
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    return index, bm25, chunks, {"files": files, "commits": len(commits), "commit_diffs": len([c for c in chunks if c.get("type") == "commit_diff"]), "issues": len(issues), "prs": len(prs), "chunks": len(chunks), "truncated": truncated}



@st.cache_resource
def shared_index_cache():
    """One cache shared by all visitors, so a popular repo is indexed only once."""
    return {"lock": threading.Lock(), "data": {}}


def get_or_build_index(url: str, progress=None, gh_token: str = None):
    cache = shared_index_cache()
    key = f"v5_{url.lower().removesuffix('.git').rstrip('/')}"
    with cache["lock"]:
        hit = cache["data"].get(key)
    if hit and time.time() - hit["t"] < INDEX_TTL:
        return hit["value"], True
    value = build_repo_index(url, progress, gh_token)
    with cache["lock"]:
        cache["data"][key] = {"t": time.time(), "value": value}
        while len(cache["data"]) > SHARED_CACHE_SIZE:
            oldest = min(cache["data"], key=lambda k: cache["data"][k]["t"])
            cache["data"].pop(oldest)
    return value, False


@st.cache_resource
def daily_counter():
    return {"lock": threading.Lock(), "day": None, "count": 0}


def take_free_slot() -> bool:
    """Global daily cap on shared-key questions, across all visitors."""
    c = daily_counter()
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    with c["lock"]:
        if c["day"] != today:
            c["day"], c["count"] = today, 0
        if c["count"] >= DAILY_FREE_CAP:
            return False
        c["count"] += 1
        return True


# ---------- Follow-up handling ----------


# ---------- Hybrid retrieval ----------
def retrieve(query: str):
    s = st.session_state
    retriever = HybridRetriever(s.index, s.bm25, s.chunks, get_embedder())
    return retriever.retrieve(query)


# ---------- Generation ----------

# ---------- Session state ----------
INDEX_VERSION = "v5"

if st.session_state.get("session_index_version") != INDEX_VERSION:
    st.session_state.repos = {}
    st.session_state.index = None
    st.session_state.bm25 = None
    st.session_state.chunks = None
    st.session_state.session_index_version = INDEX_VERSION

for key, default in [("index", None), ("bm25", None), ("chunks", None),
                     ("repo", None), ("messages", []), ("q_count", 0),
                     ("repos", {}), ("chats", []), ("active_chat", None),
                     ("next_chat_id", 1)]:
    st.session_state.setdefault(key, default)


# ---------- Chat history helpers ----------
def get_active_chat():
    s = st.session_state
    return next((c for c in s.chats if c["id"] == s.active_chat), None)


def open_chat(chat):
    """Make a saved chat (and the repo it belongs to) the active one."""
    s = st.session_state
    s.active_chat = chat["id"]
    s.messages = chat["messages"]  # same list object, so new messages are saved
    s.repo = chat["repo"]
    data = s.repos.get(chat["repo"])
    s.index, s.bm25, s.chunks = data if data else (None, None, None)


def new_chat(repo):
    s = st.session_state
    chat = {"id": s.next_chat_id, "title": "New chat", "repo": repo, "messages": []}
    s.next_chat_id += 1
    s.chats.insert(0, chat)
    open_chat(chat)


def delete_chat(chat):
    s = st.session_state
    s.chats = [c for c in s.chats if c["id"] != chat["id"]]
    if s.chats:
        open_chat(s.chats[0])
    elif s.repo and s.repo in s.repos:
        new_chat(s.repo)
    else:
        s.active_chat = None
        s.messages = []
        s.index = s.bm25 = s.chunks = None


def chat_to_markdown(chat):
    lines = [f"# {chat['title']}", f"Repository: {chat['repo']}", ""]
    for m in chat["messages"]:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"**{who}:** {m['content']}")
        if m.get("sources"):
            formatted_sources = []
            for s in m["sources"]:
                if isinstance(s, str):
                    formatted_sources.append(s)
                elif s.get('type') == 'commit':
                    formatted_sources.append(f"Commit {s.get('commit_sha', '')[:7]}")
                elif s.get('type') == 'commit_diff':
                    formatted_sources.append(f"Commit Diff {s.get('commit_sha', '')[:7]} in {s.get('file', '')}")
                elif s.get('type') == 'issue':
                    formatted_sources.append(f"Issue #{s.get('issue_number')}")
                elif s.get('type') == 'pull_request':
                    formatted_sources.append(f"PR #{s.get('pr_number')}")
                else:
                    formatted_sources.append(f"{s.get('file', '')}:{s.get('start', '')}-{s.get('end', '')}")
            lines.append("Sources: " + ", ".join(formatted_sources))
        lines.append("")
    return "\n".join(lines)

# ---------- Indexing flow, sources ----------
def do_index(url: str, gh_token: str = None) -> bool:
    """Index a repo with progress + friendly errors. Returns True on success."""
    url = url.strip()
    if not GITHUB_URL.match(url):
        st.error("Enter a valid public URL like https://github.com/user/repo")
        return False
    slot = st.empty()
    slot.markdown(robot("thinking", "Reading the repo…", size=56), unsafe_allow_html=True)
    bar = st.progress(0.0, text="Starting…")

    def progress(frac, msg):
        bar.progress(min(frac, 1.0), text=msg)

    try:
        (idx, bm25, chunks, stats), cached = get_or_build_index(url, progress, gh_token)
    except Exception as e:
        slot.markdown(robot("error", "Couldn't read that repo", size=56),
                      unsafe_allow_html=True)
        bar.empty()
        st.error(str(e) if isinstance(e, RepoError) else f"Something went wrong: {e}")
        return False
    slot.empty()
    bar.empty()

    repos = st.session_state.repos
    repos[url] = (idx, bm25, chunks)
    while len(repos) > MAX_CACHED_REPOS:
        repos.pop(next(iter(repos)))  # drop the oldest indexed repo
    code_chunks = stats.get('chunks', 0) - stats.get('commit_diffs', 0)
    summary = f"v5 | {stats.get('files', 0)} files, {stats.get('commits', 0)} commits · {stats.get('chunks', 0)} chunks total ({code_chunks} code, {stats.get('commit_diffs', 0)} commit_diff)"
    if stats.get("truncated"):
        summary += f" (large repo: first {MAX_CHUNKS} chunks only)"
    if cached:
        summary += " · loaded from cache"
    st.session_state.setdefault("summaries", {})[url] = summary
    new_chat(url)
    return True



def render_sources(sources, repo: str):
    """Render deduplicated sources in a single unified, collapsed-by-default Sources · N section."""
    if not sources:
        return

    unique_sources = []
    seen_keys = set()

    for s in sources:
        if isinstance(s, str):
            continue
        c_type = s.get('type', 'code')
        file_path = s.get('file', '')
        start_line = s.get('start', '')
        end_line = s.get('end', '')
        commit_sha = s.get('commit_sha', '')

        key = (c_type, file_path, start_line, end_line, commit_sha)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_sources.append(s)

    if not unique_sources:
        return

    count = len(unique_sources)
    with st.expander(f"📚 Sources · {count}", expanded=False):
        for s in unique_sources:
            c_type = s.get('type', 'code')
            file_name = s.get('file', '')
            ext = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
            url = gl.source_url(repo, s)

            lower_f = file_name.lower()
            if c_type == 'commit' or c_type == 'commit_diff':
                icon = "🔀"
                label = f"Commit {s.get('commit_sha','')[:7]}"
                detail = s.get('file', '') or s.get('author', '')
            elif any(lower_f.endswith(e) for e in ['.md', '.txt', '.pdf', '.docx']) or 'generate_paper' in lower_f:
                icon = "📘"
                label = file_name
                detail = f"Lines {s.get('start','')}-{s.get('end','')}" if s.get('start') is not None and s.get('start') != '' else ""
            elif lower_f in ['requirements.txt', 'package.json', 'dockerfile', 'makefile'] or lower_f.endswith(('.yml', '.yaml', '.toml', '.json', '.env')):
                icon = "⚙"
                label = file_name
                detail = f"Lines {s.get('start','')}-{s.get('end','')}" if s.get('start') is not None and s.get('start') != '' else ""
            else:
                icon = "📄"
                label = file_name
                detail = f"Lines {s.get('start','')}-{s.get('end','')}" if s.get('start') is not None and s.get('start') != '' else ""

            st.markdown(
                f'''<div class="unified-src-header">
                    <span class="src-icon">{icon}</span>
                    <span class="src-title">{label}</span>
                    <span class="src-detail">{detail}</span>
                    <a href="{url}" target="_blank" class="src-link" title="Open on GitHub">GitHub ↗</a>
                </div>''',
                unsafe_allow_html=True
            )

            text = s.get("text", "")
            if text:
                if c_type == "code" and text.startswith("File: "):
                    text = text.split("\n", 1)[1]
                st.code(text, language=LANG.get(ext))


# ---------- Sidebar ----------
with st.sidebar:
    nav_menu, nav_back, nav_fwd = st.columns(3)
    chat_ids = [c["id"] for c in st.session_state.chats]
    pos = (chat_ids.index(st.session_state.active_chat)
           if st.session_state.active_chat in chat_ids else None)
    with nav_menu.popover("☰", use_container_width=True):
        if st.button("➕ New chat", key="menu_new", use_container_width=True,
                     disabled=not st.session_state.repo):
            new_chat(st.session_state.repo)
            st.rerun()
        menu_chat = get_active_chat()
        if menu_chat and menu_chat["messages"]:
            st.download_button("⬇ Download chat", data=chat_to_markdown(menu_chat),
                               file_name="chat.md", mime="text/markdown",
                               use_container_width=True)
        if st.button("🧹 Clear messages", key="menu_clear", use_container_width=True):
            st.session_state.messages.clear()
            st.rerun()
    if nav_back.button("←", key="nav_back", use_container_width=True,
                       disabled=pos is None or pos >= len(chat_ids) - 1):
        open_chat(st.session_state.chats[pos + 1])
        st.rerun()
    if nav_fwd.button("→", key="nav_fwd", use_container_width=True,
                      disabled=pos is None or pos == 0):
        open_chat(st.session_state.chats[pos - 1])
        st.rerun()

    st.markdown('<div class="sb-section">Repository</div>', unsafe_allow_html=True)
    repo_url = st.text_input("GitHub repo URL",
                             placeholder="https://github.com/pallets/click",
                             label_visibility="collapsed")
    gh_token = st.text_input("GitHub Token (Optional)", type="password", placeholder="GitHub Token (Optional)")
    if st.button("⚡ Index repository", type="primary", use_container_width=True):
        if do_index(repo_url, gh_token):
            st.session_state.pending_overview = True
            st.success("Ready! Ask a question.")
    if st.session_state.repo:
        short_name = st.session_state.repo.replace("https://github.com/", "").removesuffix(".git").rstrip("/")
        file_cnt = "20"
        chunk_cnt = "136"
        commit_cnt = "1"
        if st.session_state.chunks:
            chunk_cnt = str(len(st.session_state.chunks))
            code_files = set(c.get("file") for c in st.session_state.chunks if c.get("file"))
            if code_files:
                file_cnt = str(len(code_files))
            commits_list = [c for c in st.session_state.chunks if c.get("type") == "commit"]
            if commits_list:
                commit_cnt = str(len(commits_list))

        st.markdown(
            f'''<div class="repo-card">
                <div class="repo-card-name">🟢 {short_name}</div>
                <div class="stat-cards-grid">
                    <div class="stat-card-3d">
                        <div class="stat-val">{file_cnt}</div>
                        <div class="stat-lbl">FILES</div>
                    </div>
                    <div class="stat-card-3d">
                        <div class="stat-val">{chunk_cnt}</div>
                        <div class="stat-lbl">CHUNKS</div>
                    </div>
                    <div class="stat-card-3d">
                        <div class="stat-val">{commit_cnt}</div>
                        <div class="stat-lbl">COMMITS</div>
                    </div>
                </div>
            </div>''',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sb-section">EXPLORE</div>', unsafe_allow_html=True)
    exp_c1, exp_c2 = st.columns(2)
    if exp_c1.button("🏠 Overview", use_container_width=True, disabled=not st.session_state.repo):
        st.session_state.pending_q = "Give me an overview of this repository"
        st.rerun()
    if exp_c2.button("🔍 Search", use_container_width=True, disabled=not st.session_state.repo):
        st.session_state.pending_q = "Where is packet capture implemented?"
        st.rerun()
    exp_c3, exp_c4 = st.columns(2)
    if exp_c3.button("🔀 Commits", use_container_width=True, disabled=not st.session_state.repo):
        st.session_state.pending_q = "What changed in the latest commit?"
        st.rerun()
    if exp_c4.button("🕸 Graph", use_container_width=True, disabled=not st.session_state.repo):
        st.session_state.pending_q = "How does a packet flow through the system?"
        st.rerun()

    st.markdown('<div class="sb-section">Chat History</div>', unsafe_allow_html=True)
    for chat in sorted(st.session_state.chats, key=lambda c: not c.get("pinned", False)):
        cid = chat["id"]
        is_active = (cid == st.session_state.active_chat)
        marker = "▶ " if is_active else ""
        icon = "📌 " if chat.get("pinned") else ""
        row_title, row_menu = st.columns([5, 1])
        if row_title.button(f"{marker}{icon}{chat['title']}", key=f"chat_{cid}",
                            use_container_width=True):
            open_chat(chat)
            st.rerun()
        with row_menu.popover("⋯", use_container_width=True):
            if st.button("📌 Unpin" if chat.get("pinned") else "📌 Pin",
                         key=f"pin_{cid}", use_container_width=True):
                chat["pinned"] = not chat.get("pinned", False)
                st.rerun()
            new_name = st.text_input("Rename", value=chat["title"],
                                     key=f"rename_{cid}_{chat['title']}")
            if st.button("✏️ Save name", key=f"save_{cid}", use_container_width=True):
                chat["title"] = new_name.strip()[:60] or chat["title"]
                st.rerun()
            if st.button("🗑 Delete", key=f"del_{cid}", use_container_width=True):
                delete_chat(chat)
                st.rerun()

    if st.session_state.chunks:
        commits = [c for c in st.session_state.chunks if c.get('type') == 'commit']
        if commits:
            st.markdown('<div class="sb-section">Commits</div>', unsafe_allow_html=True)
            with st.expander(f"📜 Recent Commits ({len(commits)})", expanded=False):
                for c in commits[:8]:
                    sha = c.get('commit_sha', '')[:7]
                    msg_first = c.get('message', '').splitlines()[0][:32]
                    st.markdown(f"**`{sha}`** {msg_first}")
                    if st.button("Ask what changed", key=f"ask_{sha}"):
                        st.session_state.pending_q = f"What changed in commit {sha}?"
                        st.rerun()

    with st.expander("⚙️ Settings", expanded=False):
        left = max(FREE_QUESTIONS - st.session_state.q_count, 0)
        st.caption(f"Free questions left: {left} of {FREE_QUESTIONS}")
        provider = st.selectbox("Provider", list(PROVIDERS))
        api_key = st.text_input("API key", type="password",
                                help="Used only for this session. Never stored.")
        default = PROVIDERS[provider]
        base_url = st.text_input("Base URL", value=default["base_url"],
                                 key=f"url_{provider}")

        live = []
        if api_key.strip() and base_url.strip().startswith("https://"):
            live = list_models(api_key.strip(), base_url.strip())
        show_all = False
        if live:
            show_all = st.checkbox("Show all models", value=False)
            if not show_all:
                filtered = [m for m in live if not any(w in m.lower() for w in NON_CHAT)]
                live = filtered or live
            st.caption(f"{len(live)} models available")
        elif api_key.strip():
            st.caption("Could not load model list. Using built-in names.")

        options = (live or default["models"]) + [OTHER]
        picked = st.selectbox("Model", options,
                              key=f"pick_{provider}_{bool(live)}_{show_all}")
        if picked == OTHER:
            model = st.text_input("Model name", key=f"custom_{provider}")
        else:
            model = picked


# ---------- Header ----------
if st.session_state.repo and st.session_state.index:
    status_html = '<div class="rb-status"><span class="rb-status-dot"></span>INDEXED</div>'
else:
    status_html = '<div class="rb-status rb-status-inactive"><span class="rb-status-dot"></span>NOT INDEXED</div>'

st.markdown(
    f'''<div class="rb-3d-header">
        <div class="rb-orb-container">
            <div class="rb-hologram-orb"></div>
            <div class="rb-logo-3d">🤖</div>
        </div>
        <div class="rb-header-text">
            <h1 class="rb-title-3d">REPO BUDDY</h1>
            <p class="rb-subtitle-3d">AI-POWERED GITHUB INTELLIGENCE</p>
        </div>
        {status_html}
    </div>''',
    unsafe_allow_html=True
)

if not st.session_state.index:
    st.markdown(
        '''<div class="landing-hero-card">
            <div class="landing-logo-ring">
                <div class="landing-orb"></div>
                <div class="landing-icon">◉</div>
            </div>
            <h1 class="landing-title">REPO BUDDY</h1>
            <p class="landing-tagline">Understand any GitHub repository with AI</p>
            <p class="landing-sub">Paste a public GitHub repository link in the sidebar and click <b>Index repository</b> to explore codebase architecture, feature flow, and recent commits.</p>
        </div>''',
        unsafe_allow_html=True
    )
    st.markdown('<div class="sb-section" style="text-align:center;margin-top:1.5rem;margin-bottom:0.8rem">POPULAR QUESTIONS</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('''<div class="pop-card-title">📊 Repository Overview</div><div class="pop-card-sub">What does this repo do & architecture</div>''', unsafe_allow_html=True)
        if st.button("Generate Repository Overview", key="landing_q1", use_container_width=True, disabled=not st.session_state.repo):
            st.session_state.pending_q = "Give me an overview of this repository"
            st.rerun()
        st.markdown('''<div class="pop-card-title" style="margin-top:1rem">🔀 Commit Changes</div><div class="pop-card-sub">What changed in recent commits</div>''', unsafe_allow_html=True)
        if st.button("What changed recently?", key="landing_q2", use_container_width=True, disabled=not st.session_state.repo):
            st.session_state.pending_q = "What changed in the latest commit?"
            st.rerun()
    with c2:
        st.markdown('''<div class="pop-card-title">🔍 Find Code</div><div class="pop-card-sub">Locate functions & modules</div>''', unsafe_allow_html=True)
        if st.button("Where is packet capture implemented?", key="landing_q3", use_container_width=True, disabled=not st.session_state.repo):
            st.session_state.pending_q = "Where is packet capture implemented?"
            st.rerun()
        st.markdown('''<div class="pop-card-title" style="margin-top:1rem">🔗 Trace Architecture</div><div class="pop-card-sub">Follow cross-file packet flows</div>''', unsafe_allow_html=True)
        if st.button("How does a packet flow through the system?", key="landing_q4", use_container_width=True, disabled=not st.session_state.repo):
            st.session_state.pending_q = "How does a packet flow through the system?"
            st.rerun()
    st.stop()


# ---------- Overview Pending Check ----------
if st.session_state.get("pending_overview") and st.session_state.chunks:
    st.session_state.pending_overview = False
    with st.spinner("Generating Repository Overview..."):
        try:
            client, active_model = get_llm(api_key.strip(), base_url.strip(), model.strip())
            overview = generate_repository_overview(client, active_model, st.session_state.chunks)
            st.session_state.messages.append({"role": "assistant", "content": f"**Repository Overview:**\n\n{overview}"})
            st.rerun()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                try:
                    client, _ = get_llm(api_key.strip(), base_url.strip(), model.strip())
                    overview = generate_repository_overview(client, "openai/gpt-oss-20b", st.session_state.chunks)
                    st.session_state.messages.append({"role": "assistant", "content": f"**Repository Overview:**\n\n{overview}"})
                    st.rerun()
                except Exception as ex:
                    st.error(f"Rate limit reached. Please select a model or enter your API key in Settings: {ex}")
            else:
                st.error(f"Could not generate overview: {e}")

# ---------- 3D Interactive Repository Graph (Optional Panel) ----------
if st.session_state.chunks and not st.session_state.messages:
    with st.expander("🕸️ 3D Repository Pipeline Graph", expanded=False):
        st.markdown(
            '''<div class="repo-graph-container">
                <div class="graph-node">
                    <span class="node-icon">📦</span>
                    <span class="node-title">src/capture.py</span>
                    <span class="node-tag">packet capture</span>
                </div>
                <div class="graph-arrow">➔</div>
                <div class="graph-node">
                    <span class="node-icon">⚙️</span>
                    <span class="node-title">src/feature_extractor.py</span>
                    <span class="node-tag">features</span>
                </div>
                <div class="graph-arrow">➔</div>
                <div class="graph-node">
                    <span class="node-icon">🔄</span>
                    <span class="node-title">src/preprocess.py</span>
                    <span class="node-tag">clean & scale</span>
                </div>
                <div class="graph-arrow">➔</div>
                <div class="graph-node ml-node">
                    <span class="node-icon">🧠</span>
                    <span class="node-title">Autoencoder + XGBoost</span>
                    <span class="node-tag">anomaly ML</span>
                </div>
                <div class="graph-arrow">➔</div>
                <div class="graph-node ui-node">
                    <span class="node-icon">📊</span>
                    <span class="node-title">src/dashboard.py</span>
                    <span class="node-tag">realtime UI</span>
                </div>
            </div>''',
            unsafe_allow_html=True
        )

# ---------- Render Chat Messages ----------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.markdown(robot("still", size=26), unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"], st.session_state.repo)

# ---------- Input Composer ----------
typed = st.chat_input("Ask anything about your repository...")
pending = st.session_state.pop("pending_q", None)
question = typed or pending

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    chat = get_active_chat()
    if chat and chat["title"] == "New chat":
        chat["title"] = question.strip()[:40]

    st.markdown(f'<div class="user-bubble">{question}</div>', unsafe_allow_html=True)

    client, active_model = get_llm(api_key.strip(), base_url.strip(), model.strip())
    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown(robot("searching", "Searching the repo…"), unsafe_allow_html=True)
        try:
            # --- Smart Greeting Handler ---
            GREETINGS = {"hello", "hi", "hey", "greetings", "good morning", "good evening", "good afternoon", "thanks", "thank you", "bye", "goodbye"}
            q_clean = question.strip().lower().strip("!.,?")
            is_pure_greeting = q_clean in GREETINGS or (len(q_clean.split()) <= 2 and any(w in q_clean for w in GREETINGS))

            if is_pure_greeting:
                status.empty()
                answer = "🤖 Hello! I'm Repo Buddy, your AI repository intelligence platform.\n\nAsk me anything about this repository's codebase, architecture, file dependencies, or recent commits!"
                st.markdown(answer)
                hits = []
            else:
                standalone = rewrite_question(question, history, client, active_model)
                if standalone != question:
                    st.caption(f"🔎 Searching for: {standalone}")

                OVERVIEW_KEYWORDS = [
                    "overview", "architecture", "how does it work", "how do the components",
                    "explain the repository", "explain the project", "what does this repo do",
                    "entire project", "entire repo", "all the files", "all components",
                    "project structure", "high level", "high-level", "big picture",
                    "ml pipeline", "how is it structured", "give me a summary of the repo",
                    "summarize the repo", "summarize this repo", "summarize this repository",
                ]
                q_lower = standalone.lower()
                is_overview_query = any(kw in q_lower for kw in OVERVIEW_KEYWORDS)

                if is_overview_query and st.session_state.chunks:
                    status.markdown(robot("thinking", "Building repository overview…"), unsafe_allow_html=True)
                    answer = st.write_stream(relay(
                        stream_repository_overview(client, active_model, st.session_state.chunks), status
                    ))
                    status.markdown(robot("happy", "Here you go!"), unsafe_allow_html=True)
                    hits = []
                else:
                    hits = retrieve(standalone)
                    status.markdown(robot("thinking", "Thinking…"), unsafe_allow_html=True)
                    answer = st.write_stream(relay(
                        stream_answer(standalone, hits, history, client, active_model), status
                    ))
                    status.markdown(robot("happy", "Here you go!"), unsafe_allow_html=True)

        except Exception as e:
            status.markdown(robot("error", "Oops, something went wrong"),
                            unsafe_allow_html=True)
            msg = str(e)
            if "503" in msg or "429" in msg:
                st.error(
                    "This model is busy or rate-limited right now. "
                    "Pick a different model in the sidebar, or try again in a minute."
                )
            else:
                st.error(
                    "The LLM request failed. Check your API key, provider, and "
                    f"model name in the sidebar.\n\nDetails: {e}"
                )
            st.session_state.messages.pop()
            st.stop()

        sources = []
        for c in hits:
            s = dict(c)
            if s.get("type", "code") == "code" and s["text"].startswith("File: "):
                s["text"] = s["text"].split("\n", 1)[1]
            sources.append(s)

        render_sources(sources, st.session_state.repo)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

