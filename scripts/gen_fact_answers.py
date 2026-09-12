"""Let the base model write the recall answers for the memory drills in its own words, so the
scored tokens in fact streams are natural sentences rather than bare values.

  .venv/bin/python scripts/gen_fact_answers.py --base Qwen/Qwen3-1.7B
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rsma.data.factstreams import TEMPLATES
from rsma.data.figures import persona_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="data/fact_answers.json")
    ap.add_argument("--per-value", type=int, default=4)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    m = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("mps").eval()
    sysmsg = persona_prompt("Sankofa")
    out = {}
    for topic, question, values in TEMPLATES:
        for v in values:
            answers = []
            for k in range(args.per_value):
                msgs = [{"role": "system", "content": sysmsg},
                        {"role": "user", "content": f"Something about me: {topic} is {v}."},
                        {"role": "assistant", "content": "Understood."},
                        {"role": "user", "content": question}]
                try:
                    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
                except TypeError:
                    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                ids = tok(text, return_tensors="pt").input_ids.to("mps")
                torch.manual_seed(k)
                gen = m.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.8, top_p=0.9)
                ans = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
                core = v.replace("a ", "").replace("an ", "").replace("the ", "").lower()
                low = ans.lower()
                hedges = ("not sure", "i can help", "i don't", "i do not", "i am not", "i'm not", "symbolic", "you haven't", "you didn't",
                          "could you", "please", "?", "if you", "in many", "represent")
                if core in low and 3 < len(ans) < 120 and not any(h in low for h in hedges) and "*" not in ans:
                    answers.append(ans)
            if not answers:
                answers = [f"{v[0].upper() + v[1:]}."]
            out[f"{topic}|{v}"] = answers
        print(topic, "->", out[f"{topic}|{values[0]}"][0][:80])
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out, len(out), "entries")


if __name__ == "__main__":
    main()
