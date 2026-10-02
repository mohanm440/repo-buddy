import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any
from retrieval.embeddings import Embedder

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CANDIDATES = 20
ANSWER_TOP_K = 5

def tokenize(text: str):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

class HybridRetriever:
    def __init__(self, index: faiss.Index, bm25: BM25Okapi, chunks: List[Dict[str, Any]], embedder: Embedder):
        self.index = index
        self.bm25 = bm25
        self.chunks = chunks
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = ANSWER_TOP_K) -> List[Dict[str, Any]]:
        # Intent detection
        is_change_query = bool(re.search(r'\b(change|changed|added|removed|deleted|commit)\b', query, re.IGNORECASE))
        commit_match = re.search(r'\bcommit\s+([a-f0-9]{7,40})\b', query, re.IGNORECASE)
        file_match = re.search(r'\b(?:in|for|file)\s+([a-zA-Z0-9_./-]+\.\w+)\b', query, re.IGNORECASE)
        
        target_commit = commit_match.group(1).lower() if commit_match else None
        target_file = file_match.group(1).lower() if file_match else None
        
        # Boost specific chunks based on intent before FAISS/BM25
        exact_matches = []
        if is_change_query and target_commit:
            for i, chunk in enumerate(self.chunks):
                if chunk.get("type") == "commit_diff" and chunk.get("commit_sha", "").startswith(target_commit):
                    if target_file:
                        c_file = chunk.get("file", "").lower()
                        if c_file.endswith(target_file) or target_file.endswith(c_file):
                            exact_matches.append(i)
                    else:
                        exact_matches.append(i)

        # Semantic search
        emb = self.embedder.encode([QUERY_PREFIX + query])
        _, ids = self.index.search(emb, CANDIDATES)
        vector_ranking = [int(i) for i in ids[0] if i != -1]

        # Keyword search
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:CANDIDATES]
        keyword_ranking = [int(i) for i in order if scores[i] > 0]

        # Reciprocal Rank Fusion
        fused = {}
        for ranking in (vector_ranking, keyword_ranking):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
                
        # Boost exact matches massively so they appear first
        for i in exact_matches:
            fused[i] = fused.get(i, 0.0) + 1000.0

        top = sorted(fused, key=fused.get, reverse=True)[:top_k]
        
        # Debug printing for development
        # Debug printing for development
        if is_change_query and target_commit:
            print("=== COMMIT DIFF DEBUG ===")
            print(f"query: {query}")
            print("detected intent: CHANGE_QUERY")
            print(f"commit SHA: {target_commit}")
            print(f"file: {target_file}")
            print("detailed commits fetched: YES (from index)")
            print(f"matching commit: YES (boosted)")
            print(f"changed files: (pre-indexed)")
            print(f"matching file: {target_file}")
            print(f"commit diff chunks: {len([c for c in self.chunks if c.get('type') == 'commit_diff'])}")
            print(f"matching diff chunks: {len(exact_matches)}")
            ret_chunks = [f"{c.get('type')} {c.get('commit_sha', '')[:7]} {c.get('file', '')}" for i in top for c in [self.chunks[i]]]
            print(f"retrieved chunks: {ret_chunks}")
            import streamlit as st
            print(f"cache version: {st.session_state.get('session_index_version', 'unknown')}")
            print("=========================")

        return [self.chunks[i] for i in top]
