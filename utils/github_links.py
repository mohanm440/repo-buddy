def source_url(repo_url: str, chunk: dict) -> str:
    """Generate a GitHub URL for a given chunk."""
    base = repo_url.removesuffix('.git').rstrip('/')
    chunk_type = chunk.get("type", "code")

    if chunk_type == "code":
        file_path = chunk.get("file", "")
        start = chunk.get("start", 1)
        end = chunk.get("end", 1)
        url = f"{base}/blob/HEAD/{file_path}"
        if not file_path.lower().endswith(".ipynb"):
            url += f"#L{start}-L{end}"
        return url
    elif chunk_type == "commit":
        sha = chunk.get("commit_sha", "")
        return f"{base}/commit/{sha}"
    elif chunk_type == "commit_diff":
        sha = chunk.get("commit_sha", "")
        return f"{base}/commit/{sha}"
    elif chunk_type == "issue":
        num = chunk.get("issue_number", "")
        return f"{base}/issues/{num}"
    elif chunk_type == "pull_request":
        num = chunk.get("pr_number", "")
        return f"{base}/pull/{num}"
    
    return base
