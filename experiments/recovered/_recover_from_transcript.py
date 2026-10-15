# -*- coding: utf-8 -*-
"""Recover scratchpad scripts from the Claude Code session transcript (JSONL).
Replays, in order, every successful Write (full content), Edit (string replace on the current
content), and full-file Read (cat -n output) for each file under the scratchpad, and reports the
final reconstructed state plus how trustworthy it is. Writes recovered files to OUT_DIR."""
import json, os, re, sys
from collections import defaultdict

TRANSCRIPT = sys.argv[1]
OUT_DIR = sys.argv[2]
MARK = "scratchpad"                       # only files that lived under the scratchpad

def norm(p):
    p = p.replace("\\", "/")
    i = p.lower().find(MARK)
    return p[i + len(MARK) + 1:] if i >= 0 else None

uses = {}                                 # tool_use_id -> (name, input)
events = []                               # ordered: (tool_use_id)
results = {}                              # tool_use_id -> (is_error, text)

with open(TRANSCRIPT, encoding="utf-8") as fh:
    for line in fh:
        try: rec = json.loads(line)
        except Exception: continue
        msg = rec.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list): continue
        for b in content:
            if not isinstance(b, dict): continue
            if b.get("type") == "tool_use" and b.get("name") in ("Write", "Edit", "Read"):
                uses[b["id"]] = (b["name"], b.get("input") or {}); events.append(b["id"])
            elif b.get("type") == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                results[b.get("tool_use_id")] = (bool(b.get("is_error")), c or "")

def parse_read(text):
    """cat -n output -> content; None if it looks truncated/partial."""
    if "Truncated" in text or "PARTIAL view" in text or "Omitted long" in text: return None
    lines = []
    for ln in text.split("\n"):
        m = re.match(r"^\s*(\d+)\t(.*)$", ln)
        if m: lines.append((int(m.group(1)), m.group(2)))
    if not lines or lines[0][0] != 1: return None
    nums = [n for n, _ in lines]
    if nums != list(range(1, len(nums) + 1)): return None
    return "\n".join(t for _, t in lines) + "\n"

state = {}                                # rel -> content
log = defaultdict(list)
for uid in events:
    name, inp = uses[uid]
    rel = norm(inp.get("file_path", "") or "")
    if not rel: continue
    err, text = results.get(uid, (True, ""))
    if err: continue
    if name == "Write":
        state[rel] = inp.get("content", ""); log[rel].append("W")
    elif name == "Edit":
        cur = state.get(rel)
        if cur is None: log[rel].append("E?"); continue
        old, new = inp.get("old_string", ""), inp.get("new_string", "")
        if old in cur:
            state[rel] = cur.replace(old, new) if inp.get("replace_all") else cur.replace(old, new, 1)
            log[rel].append("E")
        else: log[rel].append("E!")          # edit could not be replayed -> state suspect
    elif name == "Read":
        if inp.get("offset") or inp.get("limit"): continue
        c = parse_read(text)
        if c is not None and rel not in state:
            state[rel] = c; log[rel].append("R")
        elif c is not None:
            log[rel].append("R=" if c.rstrip() == state[rel].rstrip() else "R~")
            state[rel] = c                # a later full read is ground truth

os.makedirs(OUT_DIR, exist_ok=True)
ok, suspect = 0, 0
for rel in sorted(state):
    hist = "".join(x[0] for x in log[rel])
    bad = any(x in ("E?", "E!") for x in log[rel])
    flag = "SUSPECT" if bad else "ok"
    ok += not bad; suspect += bad
    dst = os.path.join(OUT_DIR, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8", newline="\n").write(state[rel])
    print(f"{flag:7s} {len(state[rel]):7d}B  {''.join(log[rel])[:40]:40s}  {rel}")
print(f"\nrecovered {len(state)} files: {ok} clean, {suspect} with an unreplayable edit")
