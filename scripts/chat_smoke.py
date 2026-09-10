"""Scripted end-to-end exercise of the chat loop: learn, reply, rollback path, consolidate, sleep, restart."""
import json
import sys

from rsma.chat import Chat

run = sys.argv[1] if len(sys.argv) > 1 else "malcolmx_test"
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

c = Chat(run, device, tier=3, max_new=80, consolidate_every=2)
turns = ["What is the ballot or the bullet?", "Tell me about Harlem.", "Who was Elijah Muhammad?", "What should black people do?"]
for t in turns:
    info = c.learn(f"\n{c.user_label}: {t}\n{c.model_label}:")
    ans = c.reply(c.transcript)
    info2 = c.learn(ans + "\n")
    c.state.turns += 1
    print(f"{c.user_label}: {t}\n{c.model_label}:{ans!r}\n  learn: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info2.items()})}")
    if c.state.turns % 2 == 0:
        res = c.do_consolidate()
        print("  consolidate:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
print("status:", json.dumps(c.status(), default=str)[:400])
print("sleep:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.do_sleep(steps=6).items()})
c.state.save()
c2 = Chat(run, device, tier=3)
assert c2.state.turns == c.state.turns, "self-state did not persist"
print("restart ok: turns", c2.state.turns, "events", len(c2.state.events))
