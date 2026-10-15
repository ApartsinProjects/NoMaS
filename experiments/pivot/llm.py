# -*- coding: utf-8 -*-
"""Budget-aware OpenRouter client for the ResearchPivot loop (project key OPENROUTER_API_KEY_ADRANK in the
gitignored project .env; the only allowed LLM route per global CLAUDE.md).

Cost discipline:
- Prompt caching: every call is [system, CONTEXT, ...turns]; CONTEXT is a byte-stable block (state brief +
  paper + prior answers) placed FIRST, so OpenAI/Gemini automatic prefix caching hits on repeat calls and
  Anthropic models get an explicit cache_control breakpoint on it.
- Concurrency: run_many() sends independent calls in parallel threads (e.g. strategist + second opinion).
- Model tiers: STRATEGIST for pivots, SECOND for a cross-family opinion, CHEAP for summaries/screening.
- Ledger: every call's real cost (usage.include) is appended to llm_spend.jsonl; a hard cap aborts calls
  once cumulative spend reaches BUDGET_USD.
"""
import json, os, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LEDGER = os.path.join(HERE, "llm_spend.jsonl")
BUDGET_USD = 40.0                      # hard stop below the key's $50 limit
STRATEGIST = "openai/gpt-6.1-sol-pro"
SECOND = "anthropic/claude-sonnet-5.5"
CHEAP = "openai/gpt-6-luna"


def _key():
    for ln in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        if ln.startswith("OPENROUTER_API_KEY_ADRANK="):
            return ln.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY_ADRANK missing from project .env")


def spent():
    if not os.path.exists(LEDGER): return 0.0
    return sum(json.loads(l).get("cost", 0.0) for l in open(LEDGER, encoding="utf-8"))


def call(model, context, turns, system="You are a rigorous senior ML research strategist.", tag="",
         max_tokens=16000, reasoning="high", timeout=900, retries=3):
    """context: big stable string (cached prefix). turns: list of {'role','content'} after it."""
    if spent() >= BUDGET_USD:
        raise RuntimeError(f"LLM budget cap reached (${spent():.2f} >= ${BUDGET_USD})")
    ctx_block = {"type": "text", "text": context}
    if model.startswith("anthropic/"):
        ctx_block["cache_control"] = {"type": "ephemeral"}
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [ctx_block]}] + turns
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "usage": {"include": True}}
    if reasoning:
        # Anthropic: an effort level alone can spend the whole output budget on hidden reasoning (seen:
        # 16000/16000 tokens, empty answer). Cap reasoning explicitly at half the output limit.
        body["reasoning"] = ({"max_tokens": max_tokens // 2} if model.startswith("anthropic/") else {"effort": reasoning})
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + _key(), "Content-Type": "application/json",
                                          "X-Title": "ADRank ResearchPivot"})
    last = None
    for a in range(retries):
        try:
            t0 = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            if "error" in r: raise RuntimeError(str(r["error"])[:300])
            u = r.get("usage", {}) or {}
            rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "model": r.get("model", model),
                   "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   "completion_tokens": u.get("completion_tokens"), "cost": float(u.get("cost") or 0.0),
                   "seconds": round(time.time() - t0, 1)}
            with open(LEDGER, "a", encoding="utf-8") as fo: fo.write(json.dumps(rec) + "\n")
            text = r["choices"][0]["message"].get("content")
            if not text:      # spent the budget without answering: do not retry blindly at the same cost
                raise ValueError(f"empty answer from {model} ({rec['completion_tokens']} tokens, likely all reasoning)")
            return text, rec
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as e:
            last = e
            code = getattr(e, "code", None)
            if code in (400, 401, 402, 403): break          # not retryable
            time.sleep(10 * (a + 1))
    raise RuntimeError(f"{model} failed: {last}")


def run_many(jobs, workers=4):
    """jobs: list of dicts of call() kwargs; returns list of (text, rec) or exceptions, in order."""
    def one(j):
        try: return call(**j)
        except Exception as e: return e
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))
