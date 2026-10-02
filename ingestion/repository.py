import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CLONE_TIMEOUT = 120
MAX_REPO_MB = 150
MAX_FILE_BYTES = 300_000
INCLUDE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".cs", ".rb", ".php", ".sh", ".md", ".rst", ".txt", ".toml", ".yaml",
    ".yml", ".json", ".ini", ".cfg", ".sql", ".html", ".css",
}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist",
             "build", ".idea", ".vscode", "vendor"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "poetry.lock"}

class RepoError(Exception):
    pass

def clone_repo_safe(url: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="repo_"))
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", url, str(tmp)],
            check=True, capture_output=True, text=True, timeout=CLONE_TIMEOUT, env=env,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RepoError("Cloning took too long. Try a smaller repository.")
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        err = (e.stderr or "").lower()
        if any(w in err for w in ("not found", "authentication", "could not read",
                                  "terminal prompts disabled")):
            raise RepoError("Repository not found, or it is private. Only public repositories work.")
        raise RepoError("Could not clone this repository.")
    except FileNotFoundError:
        raise RepoError("Git is not installed on this server.")

    size_mb = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file()) / 1e6
    if size_mb > MAX_REPO_MB:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RepoError(f"This repository is too large ({size_mb:.0f} MB, limit {MAX_REPO_MB} MB).")
    return tmp

def iter_repo_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        ext = path.suffix.lower()
        if any(p in SKIP_DIRS for p in rel.parts) or path.name in SKIP_FILES:
            continue
        if ext not in INCLUDE_EXT and ext != ".ipynb":
            continue
        if path.stat().st_size > MAX_FILE_BYTES * (3 if ext == ".ipynb" else 1):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if ext == ".ipynb":
                cells = json.loads(text).get("cells", [])
                text = "\n\n".join("".join(c.get("source", [])) for c in cells)
        except (UnicodeDecodeError, OSError, ValueError):
            continue
        if text.strip():
            yield rel.as_posix(), text
