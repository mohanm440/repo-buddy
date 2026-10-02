from typing import List, Dict, Any
from models.schemas import CodeChunk, CommitChunk, IssueChunk, PRChunk

CHUNK_LINES = 40
CHUNK_OVERLAP = 8

def chunk_file(repo_url: str, rel_path: str, text: str) -> List[CodeChunk]:
    """Sliding window over lines for code files."""
    chunks = []
    lines = text.splitlines()
    step = CHUNK_LINES - CHUNK_OVERLAP
    for start in range(0, len(lines), step):
        window = lines[start:start + CHUNK_LINES]
        body = "\n".join(window).strip()
        if not body:
            continue
        chunks.append({
            "type": "code",
            "repository": repo_url,
            "file": rel_path,
            "start": start + 1,
            "end": start + len(window),
            "text": f"File: {rel_path}\n{body}",
        })
        if start + CHUNK_LINES >= len(lines):
            break
    return chunks

def chunk_commit(repo_url: str, commit: Dict[str, Any]) -> List[Any]:
    chunks = []
    
    # 1. Base Commit Metadata Chunk
    text = f"Commit: {commit['sha']}\nAuthor: {commit['author']}\nDate: {commit['date']}\nMessage: {commit['message']}"
    chunks.append({
        "type": "commit",
        "repository": repo_url,
        "commit_sha": commit['sha'],
        "author": commit['author'],
        "date": commit['date'],
        "message": commit['message'],
        "text": text
    })
    
    # 2. File-level Diff Chunks
    files = commit.get('files', [])
    for f in files:
        filename = f.get('filename', '')
        status = f.get('status', '')
        additions = f.get('additions', 0)
        deletions = f.get('deletions', 0)
        changes = f.get('changes', 0)
        patch = f.get('patch', '')
        
        # If no patch is available (e.g. binary or huge), we still index the file status change
        if not patch:
            patch_text = f"The file was {status} but the patch was unavailable."
            file_chunks = [patch_text]
        else:
            # Split large patches safely (e.g. by newlines, 100 lines at a time)
            patch_lines = patch.splitlines()
            file_chunks = ["\n".join(patch_lines[i:i+100]) for i in range(0, len(patch_lines), 100)]
            
        for i, p_chunk in enumerate(file_chunks):
            chunk_text = (
                f"Commit Diff: {commit['sha'][:7]}\n"
                f"Message: {commit.get('message', '').splitlines()[0] if commit.get('message') else ''}\n"
                f"File: {filename}\n"
                f"Status: {status}\n"
                f"Additions: {additions} | Deletions: {deletions} | Changes: {changes}\n"
                f"Patch ({i+1}/{len(file_chunks)}):\n{p_chunk}"
            )
            chunks.append({
                "type": "commit_diff",
                "repository": repo_url,
                "commit_sha": commit['sha'],
                "short_sha": commit['sha'][:7],
                "commit_message": commit.get('message', ''),
                "file": filename,
                "status": status,
                "additions": additions,
                "deletions": deletions,
                "changes": changes,
                "patch": p_chunk,
                "chunk_index": i + 1,
                "total_chunks": len(file_chunks),
                "text": chunk_text
            })
            
    return chunks

def chunk_issue(repo_url: str, issue: Dict[str, Any]) -> List[IssueChunk]:
    # Very long issues might need further chunking, but for now we keep it simple as a single chunk
    # Truncate body if it's excessively long
    body = issue['body'][:3000] if issue['body'] else ""
    text = f"Issue #{issue['number']}: {issue['title']}\nState: {issue['state']}\nAuthor: {issue['author']}\nDate: {issue['created_at']}\n\n{body}"
    return [{
        "type": "issue",
        "repository": repo_url,
        "issue_number": issue['number'],
        "title": issue['title'],
        "author": issue['author'],
        "state": issue['state'],
        "created_at": issue['created_at'],
        "text": text
    }]

def chunk_pr(repo_url: str, pr: Dict[str, Any]) -> List[PRChunk]:
    body = pr['body'][:3000] if pr['body'] else ""
    text = f"Pull Request #{pr['number']}: {pr['title']}\nState: {pr['state']}\nAuthor: {pr['author']}\n\n{body}"
    return [{
        "type": "pull_request",
        "repository": repo_url,
        "pr_number": pr['number'],
        "title": pr['title'],
        "author": pr['author'],
        "state": pr['state'],
        "text": text
    }]
