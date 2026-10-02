import argparse
import json
import os
from pathlib import Path
import faiss
from openai import OpenAI

from ingestion.repository import clone_repo_safe, iter_repo_files, RepoError
from ingestion.github_api import fetch_commits, fetch_issues, fetch_pull_requests
from ingestion.chunker import chunk_file, chunk_commit, chunk_issue, chunk_pr
from retrieval.embeddings import Embedder
from retrieval.hybrid_search import HybridRetriever, tokenize
from generation.llm import stream_answer
from rank_bm25 import BM25Okapi
import numpy as np
import shutil

INDEX_DIR = Path("repo_index")
LLM_MODEL = "openai/gpt-oss-120b"
TOP_K = 6

def build_index(repo_url: str, gh_token: str = None):
    print("Cloning repository...")
    root = clone_repo_safe(repo_url)
    
    chunks = []
    try:
        print("Reading files...")
        for rel_path, text in iter_repo_files(root):
            chunks.extend(chunk_file(repo_url, rel_path, text))
    finally:
        shutil.rmtree(root, ignore_errors=True)
        
    print("Fetching Commits...")
    for c in fetch_commits(repo_url, gh_token): chunks.extend(chunk_commit(repo_url, c))
    
    print("Fetching Issues...")
    for i in fetch_issues(repo_url, gh_token): chunks.extend(chunk_issue(repo_url, i))
    
    print("Fetching Pull Requests...")
    for pr in fetch_pull_requests(repo_url, gh_token): chunks.extend(chunk_pr(repo_url, pr))

    if not chunks:
        raise SystemExit("No indexable files or data found.")
    print(f"Embedding {len(chunks)} chunks...")

    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts)
    
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    
    bm25 = BM25Okapi([tokenize(t) for t in texts])

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks))
    # Note: we can't easily serialize bm25 to JSON, but for this CLI we might just rebuild it on load, 
    # or skip hybrid search for the CLI. To keep CLI working, we rebuild bm25 on load.
    print(f"Saved index to {INDEX_DIR}/")

class CLI_Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text())
        self.embedder = Embedder()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.retriever = HybridRetriever(self.index, self.bm25, self.chunks, self.embedder)

    def search(self, query: str, k: int = TOP_K):
        return self.retriever.retrieve(query, top_k=k)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    build_p = sub.add_parser("build")
    build_p.add_argument("repo_url")
    build_p.add_argument("--token", help="GitHub API Token")
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("chat")
    args = ap.parse_args()

    if args.cmd == "build":
        build_index(args.repo_url, gh_token=args.token)
        return

    retriever = CLI_Retriever()
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY", "dummy"),
        base_url="https://api.groq.com/openai/v1",
    )
    
    if args.cmd == "ask":
        hits = retriever.search(args.question)
        for chunk in stream_answer(args.question, hits, [], client, LLM_MODEL):
            print(chunk, end="")
        print()
    else:
        history = []
        while (q := input("\\nQuestion (blank to quit): ").strip()):
            hits = retriever.search(q)
            history.append({"role": "user", "content": q})
            print("\\nAnswer: ", end="")
            full_ans = ""
            for chunk in stream_answer(q, hits, history, client, LLM_MODEL):
                print(chunk, end="", flush=True)
                full_ans += chunk
            history.append({"role": "assistant", "content": full_ans})
            print()

if __name__ == "__main__":
    main()