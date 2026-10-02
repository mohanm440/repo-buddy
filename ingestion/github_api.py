import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any

def _fetch_github_api(url: str, token: str = None) -> List[Dict[str, Any]]:
    headers = {"User-Agent": "Repo-Buddy-App"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"GitHub API Error for {url}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected Error for {url}: {e}")
        return []

def get_owner_repo(repo_url: str):
    parts = repo_url.removesuffix(".git").rstrip("/").split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "", ""

def fetch_commits(repo_url: str, token: str = None, limit: int = 50, detail_limit: int = 15) -> List[Dict]:
    owner, repo = get_owner_repo(repo_url)
    if not owner or not repo: return []
    url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page={limit}"
    data = _fetch_github_api(url, token)
    commits = []
    
    for idx, c in enumerate(data):
        if not isinstance(c, dict): continue
        commit_sha = c.get("sha", "")
        commit_info = c.get("commit", {})
        
        commit_dict = {
            "sha": commit_sha,
            "author": commit_info.get("author", {}).get("name", "Unknown"),
            "date": commit_info.get("author", {}).get("date", ""),
            "message": commit_info.get("message", ""),
            "files": []
        }
        
        # Fetch detailed patch/diff data for the latest `detail_limit` commits
        if idx < detail_limit and commit_sha:
            detail_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
            detail_data = _fetch_github_api(detail_url, token)
            if isinstance(detail_data, dict):
                commit_dict["files"] = detail_data.get("files", [])
                
        commits.append(commit_dict)
        
    return commits

def fetch_issues(repo_url: str, token: str = None, limit: int = 50) -> List[Dict]:
    owner, repo = get_owner_repo(repo_url)
    if not owner or not repo: return []
    url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=all&per_page={limit}"
    data = _fetch_github_api(url, token)
    issues = []
    for i in data:
        if not isinstance(i, dict): continue
        if "pull_request" in i: continue # Skip PRs which are also returned in issues API
        issues.append({
            "number": i.get("number", 0),
            "title": i.get("title", ""),
            "author": i.get("user", {}).get("login", "Unknown"),
            "state": i.get("state", ""),
            "created_at": i.get("created_at", ""),
            "body": i.get("body", "") or ""
        })
    return issues

def fetch_pull_requests(repo_url: str, token: str = None, limit: int = 50) -> List[Dict]:
    owner, repo = get_owner_repo(repo_url)
    if not owner or not repo: return []
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all&per_page={limit}"
    data = _fetch_github_api(url, token)
    prs = []
    for pr in data:
        if not isinstance(pr, dict): continue
        prs.append({
            "number": pr.get("number", 0),
            "title": pr.get("title", ""),
            "author": pr.get("user", {}).get("login", "Unknown"),
            "state": pr.get("state", ""),
            "body": pr.get("body", "") or ""
        })
    return prs
