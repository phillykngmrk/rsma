"""
North star 1: does the memory carry meaning?

Session A: tell Sankofa a set of personal facts in conversation, let it consolidate, save state.
Session B: a fresh process with no transcript, only the persisted fast weights and checkpoint.
Ask about each fact and score the true answer against distractors by log-likelihood.

Control: the same questions with the fast state reset and the pre-session checkpoint restored,
so the only difference is what was learned in session A.

  python -m rsma.memtest --run mentor            # full A -> B protocol, prints recall accuracy
  python -m rsma.memtest --run mentor --facts 8 --sleep
"""
import argparse
import json
import os
import random
import shutil
import time

import torch

from .chat_graft import GraftChat
from .train import get_device

FACTS = [
    ("my dog's name", "What is my dog's name?", ["Biscuit", "Pepper", "Juno", "Maple", "Otis"]),
    ("the city I was born in", "Which city was I born in?", ["Philadelphia", "Atlanta", "Detroit", "Houston", "Oakland"]),
    ("my favorite composer", "Who is my favorite composer?", ["Coltrane", "Bach", "Nina Simone", "Ellington", "Debussy"]),
    ("the month of my birthday", "What month is my birthday?", ["March", "August", "November", "June", "January"]),
    ("the language I am learning", "Which language am I learning?", ["Portuguese", "Japanese", "Swahili", "Arabic", "Korean"]),
    ("my sister's profession", "What does my sister do for work?", ["a nurse", "an architect", "a pilot", "a chef", "a lawyer"]),
    ("the car I drive", "What car do I drive?", ["a Volvo", "a Honda", "a Tesla", "a Subaru", "a Jeep"]),
    ("my favorite book", "What is my favorite book?", ["Invisible Man", "Dune", "Beloved", "Meditations", "The Odyssey"]),
    ("the sport I played in college", "Which sport did I play in college?", ["rugby", "swimming", "basketball", "fencing", "track"]),
    ("the street I live on", "What street do I live on?", ["Cedar Street", "Lincoln Avenue", "Baker Road", "Vine Street", "Harbor Lane"]),
]


def sentence(topic, value):
    return f"Something about me: {topic} is {value}."


@torch.no_grad()
def answer_loglik(chat, question, answer):
    """Log-likelihood of `answer` as the assistant reply to `question`, under the current fast state."""
    tok, m, cfg = chat.tok, chat.model, chat.cfg
    msgs = [{"role": "system", "content": chat.system_prompt}, {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(answer + "<|im_end|>", add_special_tokens=False).input_ids
    ids = p_ids + a_ids
    C = cfg.chunk_size
    pad = (-len(ids)) % C
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    x = torch.tensor([pad_id] * pad + ids, device=chat.device)[None]
    attn = torch.tensor([0] * pad + [1] * len(ids), device=chat.device)[None]
    y = torch.full_like(x, -100)
    y[0, pad + len(p_ids) - 1: pad + len(ids) - 1] = x[0, pad + len(p_ids): pad + len(ids)]
    _, _, aux = m(x, chat.state.fast, targets=y, attention_mask=attn)
    per = aux["per_token_loss"][0]
    mask = (y[0] != -100)
    return -(per * mask).sum().item()


def score(chat, assignment, baseline=None):
    """Returns (correct, mean margin, per-fact log-likelihood table). With `baseline` (a previous
    table), correctness is judged on the SHIFT of each option's log-likelihood since baseline, which
    removes the model's prior preference among the options and isolates what was learned."""
    correct, margins, table = 0, [], {}
    for topic, question, options in FACTS[: len(assignment)]:
        truth = assignment[topic]
        lls = {o: answer_loglik(chat, question, o) for o in options}
        table[topic] = lls
        if baseline is not None:
            lls = {o: lls[o] - baseline[topic][o] for o in options}
        best = max(lls, key=lls.get)
        correct += int(best == truth)
        others = [v for k, v in lls.items() if k != truth]
        margins.append(lls[truth] - max(others))
    return correct, sum(margins) / len(margins), table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--facts", type=int, default=6)
    ap.add_argument("--sleep", action="store_true", help="also run a sleep pass in session A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt", default=None, help="evaluate this checkpoint file instead of the run's ckpt.pt (e.g. runs/x/ckpt_step800.pt)")
    ap.add_argument("--repeats", type=int, default=1, help="average over this many seeds")
    args = ap.parse_args()
    device = args.device or get_device()
    if args.ckpt:
        # evaluate an arbitrary checkpoint in a scratch run dir so the real run is untouched
        scratch = os.path.join("runs", f"_memtest_{os.path.basename(args.ckpt).replace('.pt', '')}")
        shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch)
        shutil.copy(args.ckpt, os.path.join(scratch, "ckpt.pt"))
        args.run = os.path.basename(scratch)
    if args.repeats > 1:
        totals = {"recall": 0, "reset": 0, "in_context": 0, "margin_recall": 0.0}
        for r in range(args.repeats):
            res = run_once(args, device, args.seed + r)
            for k in totals:
                totals[k] += res[k]
        n = args.repeats
        print(json.dumps({"ckpt": args.ckpt or args.run, "repeats": n, "facts": args.facts,
                          "recall_mean": totals["recall"] / n, "reset_mean": totals["reset"] / n,
                          "in_context_mean": totals["in_context"] / n, "margin_recall_mean": totals["margin_recall"] / n}))
        return
    run_once(args, device, args.seed)


def run_once(args, device, seed):
    rng = random.Random(seed)
    assignment = {topic: rng.choice(options) for topic, _, options in FACTS[: args.facts]}

    run_dir = os.path.join("runs", args.run)
    backup_ckpt = os.path.join(run_dir, "ckpt.memtest_backup.pt")
    shutil.copy(os.path.join(run_dir, "ckpt.pt"), backup_ckpt)
    self_dir = os.path.join(run_dir, "self")
    self_backup = os.path.join(run_dir, "self.memtest_backup")
    if os.path.exists(self_dir):
        shutil.rmtree(self_backup, ignore_errors=True)
        shutil.copytree(self_dir, self_backup)
    shutil.rmtree(self_dir, ignore_errors=True)

    try:
        # ---- session A: tell the facts
        a = GraftChat(args.run, device, tier=3, max_new=40, think=False)
        base_correct, base_margin, base_table = score(a, assignment)   # before learning anything
        print(f"[before] {base_correct}/{args.facts} correct by raw preference, margin {base_margin:+.2f}")
        for topic, question, options in FACTS[: args.facts]:
            text = sentence(topic, assignment[topic])
            a.learn(a.turn_ids("user", text))
            reply, _ = a.reply(text)
            a.learn(a.turn_ids("assistant", reply))
            a.state.turns += 1
        in_context_correct, in_context_margin, _ = score(a, assignment, base_table)
        print(f"[session A, after telling, fast weights only] {in_context_correct}/{args.facts} learned, shift margin {in_context_margin:+.3f}")
        merge = a.do_consolidate()
        print(f"[consolidate] accepted={merge.get('accepted')} eta={merge.get('eta')}")
        if args.sleep:
            nap = a.do_sleep(steps=20)
            print(f"[sleep] accepted={nap.get('accepted')} {nap.get('before', 0):.4f} -> {nap.get('after', 0):.4f}")
        a.state.save()
        del a

        # ---- session B: fresh process, no transcript
        b = GraftChat(args.run, device, tier=3, think=False)
        recall_correct, recall_margin, _ = score(b, assignment, base_table)
        print(f"[session B, fresh process, persisted weights] {recall_correct}/{args.facts} learned, shift margin {recall_margin:+.3f}")
        b.state.fast = None
        reset_correct, reset_margin, _ = score(b, assignment, base_table)
        print(f"[session B, fast state reset] {reset_correct}/{args.facts} learned, shift margin {reset_margin:+.3f}")
        result = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "facts": args.facts, "before": base_correct,
                  "in_context": in_context_correct, "recall": recall_correct, "reset": reset_correct,
                  "margin_before": base_margin, "margin_in_context": in_context_margin, "margin_recall": recall_margin,
                  "margin_reset": reset_margin, "consolidated": merge.get("accepted"), "sleep": args.sleep}
        with open(os.path.join(run_dir, "memtest.jsonl"), "a") as f:
            f.write(json.dumps(result) + "\n")
        print(json.dumps(result))
        return result
    finally:
        # restore the model and state exactly as they were
        shutil.move(backup_ckpt, os.path.join(run_dir, "ckpt.pt"))
        shutil.rmtree(self_dir, ignore_errors=True)
        if os.path.exists(self_backup):
            shutil.move(self_backup, self_dir)


if __name__ == "__main__":
    main()
