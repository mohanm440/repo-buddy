import re
from openai import OpenAI
from typing import List, Dict, Any
from generation.prompt import SYSTEM_PROMPT, OVERVIEW_PROMPT

FOLLOWUP_WORDS = {"it", "that", "this", "they", "them", "those", "these", "its",
                  "above", "previous", "second", "first", "third", "same", "more"}
HISTORY_TURNS = 3

def needs_rewrite(question: str) -> bool:
    words = re.findall(r"[a-z']+", question.lower())
    return len(words) < 8 or any(w in FOLLOWUP_WORDS for w in words)

def rewrite_question(question: str, history: List[Dict[str, str]], client: OpenAI, model: str) -> str:
    if not history or not needs_rewrite(question):
        return question
    convo = "\n".join(
        f"{m['role']}: {m['content'][:500]}" for m in history[-HISTORY_TURNS:]
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": (
                    "Rewrite the user's last question as a complete standalone "
                    "question about the repository, using the conversation for "
                    "context. If it is already standalone, return it unchanged. "
                    "Return ONLY the question.")},
                {"role": "user",
                 "content": f"Conversation:\n{convo}\n\nLast question: {question}"},
            ],
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.S).strip()
        return text or question
    except Exception:
        return question

def stream_answer(question: str, hits: List[Dict[str, Any]], history: List[Dict[str, str]], client: OpenAI, model: str):
    context_parts = []
    for c in hits:
        c_type = c.get("type", "code")
        try:
            if c_type == "code":
                context_parts.append(f"[{c['file']}:{c['start']}-{c['end']}]\n{c['text'][:1800]}")
            elif c_type == "commit":
                context_parts.append(f"[commit:{c['commit_sha'][:7]}]\n{c['text'][:1800]}")
            elif c_type == "commit_diff":
                context_parts.append(f"[COMMIT DIFF]\nCitation: [commit:{c['commit_sha'][:7]}:file:{c['file']}]\n{c['text'][:1800]}")
            elif c_type == "issue":
                context_parts.append(f"[issue:#{c['issue_number']}]\n{c['text'][:1800]}")
            elif c_type == "pull_request":
                context_parts.append(f"[PR:#{c['pr_number']}]\n{c['text'][:1800]}")
        except KeyError as e:
            print(f"\n[ERROR] KeyError {e} in chunk: {c.keys()} type: {c_type}")
            
    context = "\n\n---\n\n".join(context_parts)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [
        {"role": m["role"], "content": m["content"][:800]}
        for m in history[-HISTORY_TURNS:]
    ]
    messages.append(
        {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}
    )
    stream = client.chat.completions.create(
        model=model, temperature=0.1, stream=True, messages=messages
    )

    buf, in_thought = "", False
    for event in stream:
        if not (event.choices and event.choices[0].delta.content):
            continue
        buf += event.choices[0].delta.content
        while buf:
            if in_thought:
                end = buf.find("</thought>")
                if end == -1:
                    buf = buf[-10:]
                    break
                buf = buf[end + len("</thought>"):]
                in_thought = False
            else:
                start = buf.find("<thought>")
                if start == -1:
                    if len(buf) > 9:
                        yield buf[:-9]
                        buf = buf[-9:]
                    break
                yield buf[:start]
                buf = buf[start + len("<thought>"):]
                in_thought = True
    if buf and not in_thought:
        yield buf

def stream_repository_overview(client: OpenAI, model: str, chunks: List[Dict[str, Any]]):
    all_files = list(set(c.get("file") for c in chunks if c.get("type", "code") == "code" and c.get("file")))
    
    core_files, doc_files, test_files, config_files = [], [], [], []
    for f in all_files:
        lower_f = f.lower()
        if any(x in lower_f for x in ['test', 'spec', '__tests__']):
            test_files.append(f)
        elif lower_f in ['requirements.txt', 'package.json', 'setup.py', 'dockerfile', 'makefile', '.env.example'] or lower_f.endswith('.yml') or lower_f.endswith('.yaml'):
            config_files.append(f)
        elif any(lower_f.endswith(ext) for ext in ['.md', '.txt', '.docx', '.pdf']) or 'generate_paper' in lower_f or 'generate_enhanced_paper' in lower_f:
            doc_files.append(f)
        else:
            core_files.append(f)
            
    core_files.sort(key=lambda x: (0 if any(d in x.lower() for d in ['src/', 'models/', 'lib/', 'app/', 'realtime']) else 1, x))
    selected_files = core_files[:10] + doc_files[:3] + config_files[:2]
    
    overview_chunks = []
    for f in selected_files:
        file_chunks = [c for c in chunks if c.get("type", "code") == "code" and c.get("file") == f]
        if file_chunks:
            overview_chunks.append(file_chunks[0])
            if len(file_chunks) > 1:
                overview_chunks.append(file_chunks[len(file_chunks)//2])
                
    if not overview_chunks:
        overview_chunks = [c for c in chunks if c.get("type", "code") == "code"][:15]
    
    file_tree_section = (
        f"CORE_IMPLEMENTATION_FILES:\n" +
        "\n".join(f"  - {f}" for f in core_files) +
        f"\n\nDOCUMENTATION_FILES:\n" +
        "\n".join(f"  - {f}" for f in doc_files) +
        f"\n\nCONFIGURATION_FILES:\n" +
        "\n".join(f"  - {f}" for f in config_files) +
        f"\n\nTEST_DEBUG_FILES:\n" +
        "\n".join(f"  - {f}" for f in test_files)
    )

    excerpts_section = "\n\n---\n\n".join(
        f"[{c['file']}:{c['start']}-{c['end']}]\n{c['text'][:800]}" for c in overview_chunks
    )

    user_message = (
        f"=== REPOSITORY FILE CATEGORIES ===\n{file_tree_section}\n\n"
        f"=== SOURCE CODE EXCERPTS (sampled from CORE_IMPLEMENTATION first) ===\n{excerpts_section}\n\n"
        "Please generate the repository overview."
    )

    stream = client.chat.completions.create(
        model=model,
        temperature=0.2,
        stream=True,
        messages=[
            {"role": "system", "content": OVERVIEW_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    buf, in_thought = "", False
    for event in stream:
        if not (event.choices and event.choices[0].delta.content):
            continue
        buf += event.choices[0].delta.content
        while buf:
            if in_thought:
                end = buf.find("</thought>")
                if end == -1:
                    buf = buf[-10:]
                    break
                buf = buf[end + len("</thought>"):]
                in_thought = False
            else:
                start = buf.find("<thought>")
                if start == -1:
                    if len(buf) > 9:
                        yield buf[:-9]
                        buf = buf[-9:]
                    break
                yield buf[:start]
                buf = buf[start + len("<thought>"):]
                in_thought = True
    if buf and not in_thought:
        yield buf

def generate_repository_overview(client: OpenAI, model: str, chunks: List[Dict[str, Any]]) -> str:
    return "".join(stream_repository_overview(client, model, chunks))

