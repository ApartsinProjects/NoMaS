# -*- coding: utf-8 -*-
"""ResearchPivot strategist consult over OpenRouter (strategist + cross-family second opinion in parallel).
The CONTEXT block is byte-stable across cycles (cacheable prefix); each cycle appends its own turns.
Usage: python consult.py <cycle_number> <question_file>"""
import os, sys, json
import llm

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE); ROOT = os.path.dirname(EXP)


def read(p): return open(p, encoding="utf-8").read()


def context():
    parts = [("RESEARCHPIVOT_STATE.md (state brief, updated each cycle)", os.path.join(EXP, "RESEARCHPIVOT_STATE.md")),
             ("nomas.md (current paper source)", os.path.join(ROOT, "paper", "nomas.md")),
             ("PRIOR_ART_cycle1.md", os.path.join(HERE, "PRIOR_ART_cycle1.md"))]
    return "\n\n".join(f"===== FILE: {name} =====\n{read(p)}" for name, p in parts)


if __name__ == "__main__":
    cyc = int(sys.argv[1]); q = read(sys.argv[2])
    hist = []
    c1 = os.path.join(HERE, "GPT_cycle1_answer.md")
    if os.path.exists(c1):
        hist = [{"role": "user", "content": "Cycle 1 request: propose three ranked high-ceiling pivots and a triage experiment."},
                {"role": "assistant", "content": read(c1)}]
    turns = hist + [{"role": "user", "content": q}]
    ctx = context()
    jobs = [dict(model=llm.STRATEGIST, context=ctx, turns=turns, tag=f"cycle{cyc}-strategist"),
            dict(model=llm.SECOND, context=ctx, turns=turns, tag=f"cycle{cyc}-second", reasoning="high")]
    res = llm.run_many(jobs, workers=2)
    for j, r in zip(jobs, res):
        name = "strategist" if j["model"] == llm.STRATEGIST else "second"
        out = os.path.join(HERE, f"LLM_cycle{cyc}_{name}.md")
        if isinstance(r, Exception):
            print(f"{name}: FAILED {r}"); continue
        text, rec = r
        open(out, "w", encoding="utf-8").write(text)
        print(f"{name}: {rec['model']} {rec['prompt_tokens']} in ({rec['cached_tokens']} cached) / {rec['completion_tokens']} out, ${rec['cost']:.3f}, {rec['seconds']}s -> {os.path.basename(out)}")
    print(f"ledger total ${llm.spent():.3f}")
