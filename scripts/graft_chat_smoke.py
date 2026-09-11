"""Scripted exercise of the grafted chat loop: learn, reply, rollback, consolidate, sleep, restart."""
import json
import sys

from rsma.chat_graft import GraftChat

run = sys.argv[1] if len(sys.argv) > 1 else "mentor"
device = sys.argv[2] if len(sys.argv) > 2 else "mps"

c = GraftChat(run, device, tier=3, max_new=60, consolidate_every=2)
turns = ["What does equal protection under the law mean?", "How should I think about risk when investing?",
         "What did Darwin actually claim?", "Give me one habit for a calmer mind."]
for t in turns:
    info = c.learn(c.turn_ids("user", t))
    ans, _ = c.reply(t)
    info2 = c.learn(c.turn_ids("assistant", ans))
    c.state.turns += 1
    print(f"You: {t}\n{c.persona_name}: {ans[:200]!r}\n  learn: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info2.items()})}")
    if c.state.turns % 2 == 0:
        res = c.do_consolidate()
        print("  consolidate:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "tried"})
print("status:", json.dumps(c.status(), default=str)[:300])
print("sleep:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.do_sleep(steps=4).items()})
c.state.save()
c2 = GraftChat(run, device, tier=3)
assert c2.state.turns == c.state.turns, "self-state did not persist"
print("restart ok: turns", c2.state.turns, "events", len(c2.state.events))
