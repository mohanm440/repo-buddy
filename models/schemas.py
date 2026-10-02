from typing import TypedDict, Optional, Union

class BaseChunk(TypedDict):
    type: str
    repository: str
    text: str # Text to be embedded

class CodeChunk(BaseChunk):
    file: str
    start: int
    end: int

class CommitChunk(BaseChunk):
    commit_sha: str
    author: str
    date: str
    message: str

class IssueChunk(BaseChunk):
    issue_number: int
    title: str
    author: str
    state: str
    created_at: str

class PRChunk(BaseChunk):
    pr_number: int
    title: str
    author: str
    state: str

class CommitDiffChunk(BaseChunk):
    commit_sha: str
    file: str
    status: str
    additions: int
    deletions: int
    changes: int
    chunk_index: int
    total_chunks: int

ChunkType = Union[CodeChunk, CommitChunk, IssueChunk, PRChunk, CommitDiffChunk]
