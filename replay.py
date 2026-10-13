"""Replay hand recordings through the gesture engine, or run the synthetic self-test.

  python replay.py recordings\\2026-10-12_2301.jsonl   print the events a recording produces
  python replay.py --selftest                          check every gesture on synthetic hands

Record in AirTyper with Ctrl+Alt+F9 (start/stop). Each line is {"t": seconds, "control": bool,
"hands": [[[x, y, z] * 21], ...]}.
"""
import json
import sys
from types import SimpleNamespace

import numpy as np

from gesture_engine import Engine

FPS = 30


def to_lm(points):
    return [SimpleNamespace(x=p[0], y=p[1], z=p[2] if len(p) > 2 else 0.0) for p in points]


def replay(path):
    eng = Engine()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for e in eng.update([to_lm(h) for h in rec["hands"]], rec["t"], rec.get("control", True)):
                if e["type"] not in ("pointer", "carry"):
                    print(f"{rec['t']:8.2f}s  {e}")


# ---------- synthetic hands ----------
# offsets from the wrist in square units (x already scaled by 4/3), palm size 0.1, palm facing the camera
UP = {
    "thumb": [(-0.03, -0.02), (-0.055, -0.04), (-0.075, -0.06), (-0.095, -0.075)],
    "index": [(-0.03, -0.095), (-0.033, -0.135), (-0.035, -0.16), (-0.037, -0.185)],
    "middle": [(0, -0.1), (0, -0.145), (0, -0.172), (0, -0.198)],
    "ring": [(0.025, -0.095), (0.027, -0.135), (0.028, -0.16), (0.029, -0.183)],
    "pinky": [(0.048, -0.085), (0.052, -0.115), (0.054, -0.135), (0.056, -0.155)],
}
DOWN = {
    "thumb": [(-0.03, -0.02), (-0.055, -0.04), (-0.035, -0.06), (-0.01, -0.07)],
    "index": [(-0.03, -0.095), (-0.033, -0.12), (-0.03, -0.095), (-0.028, -0.08)],
    "middle": [(0, -0.1), (0, -0.125), (0, -0.1), (0, -0.085)],
    "ring": [(0.025, -0.095), (0.027, -0.12), (0.026, -0.097), (0.025, -0.082)],
    "pinky": [(0.048, -0.085), (0.05, -0.105), (0.049, -0.088), (0.048, -0.075)],
}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def hand(pattern, cx=0.65, cy=0.65, scale=1.0, ok=False):
    """Landmarks for a finger pattern (thumb..pinky, 1 = up), wrist at (cx, cy) in square units."""
    pts = [(0.0, 0.0)]
    for name, up in zip(FINGERS, pattern):
        pts += (UP if up else DOWN)[name]
    if ok:  # thumb and index tips touching, other fingers up
        pts[6], pts[7], pts[8] = (-0.045, -0.125), (-0.055, -0.11), (-0.06, -0.1)
        pts[3], pts[4] = (-0.07, -0.08), (-0.062, -0.098)
    return [SimpleNamespace(x=(cx + dx * scale) * 3 / 4, y=cy + dy * scale, z=0.0) for dx, dy in pts]


OPEN, FIST, TWO, ONE = (1, 1, 1, 1, 1), (0, 0, 0, 0, 0), (0, 1, 1, 0, 0), (0, 1, 0, 0, 0)
ROCK, HALF = (0, 1, 0, 0, 1), (0, 1, 1, 0, 0)


class Sim:
    def __init__(self, control=True):
        self.eng, self.t, self.control, self.events = Engine(), 0.0, control, []

    def run(self, seconds, *frames_fn):
        n = max(1, int(round(seconds * FPS)))
        for i in range(n):
            hands = [fn(i / max(n - 1, 1)) for fn in frames_fn]
            self.events += [(round(self.t, 2), e) for e in self.eng.update(hands, self.t, self.control)
                            if e["type"] != "pointer"]
            self.t += 1 / FPS
        return self

    def types(self):
        return [e["type"] + (f":{e.get('dir') or e.get('id') or e.get('n', '')}" if e["type"] in
                             ("swipe", "pose", "volume", "zoom") else "") for _, e in self.events
                if e["type"] != "carry"]


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got}" + ("" if ok else f"   (wanted {want})"))
    return ok


def selftest():
    results = []
    # swipe right, then the hand returns left: only one event
    s = Sim().run(0.6, lambda p: hand(OPEN)).run(0.2, lambda p: hand(OPEN, cx=0.65 + 0.25 * p)) \
             .run(0.3, lambda p: hand(OPEN, cx=0.9)).run(0.25, lambda p: hand(OPEN, cx=0.9 - 0.25 * p))
    results.append(check("swipe right + return stroke", s.types(), ["swipe:right"]))
    # two-finger swipe left
    s = Sim().run(0.6, lambda p: hand(TWO)).run(0.2, lambda p: hand(TWO, cx=0.65 - 0.25 * p))
    results.append(check("two-finger swipe", [(e["dir"], e["two"]) for _, e in s.events], [("left", True)]))
    # raising a hand into view fast must not swipe up
    s = Sim().run(0.25, lambda p: hand(OPEN, cy=0.95 - 0.35 * p)).run(0.3, lambda p: hand(OPEN, cy=0.6))
    results.append(check("hand entering frame", s.types(), []))
    # slow drift is not a swipe
    s = Sim().run(0.6, lambda p: hand(OPEN)).run(1.5, lambda p: hand(OPEN, cx=0.65 + 0.25 * p))
    results.append(check("slow drift", s.types(), []))
    # push toward the camera
    s = Sim().run(0.6, lambda p: hand(OPEN)).run(0.25, lambda p: hand(OPEN, scale=1 + 0.45 * p))
    results.append(check("push", s.types(), ["push"]))
    # catch, hold, flick open = splash (paste)
    s = Sim().run(0.6, lambda p: hand(OPEN)).run(1.0, lambda p: hand(FIST)).run(0.3, lambda p: hand(OPEN))
    results.append(check("catch + splash", s.types(), ["catch", "splash"]))
    # catch, then open slowly = drop
    s = Sim().run(0.6, lambda p: hand(OPEN)).run(0.8, lambda p: hand(FIST)) \
             .run(0.5, lambda p: hand(HALF)).run(0.3, lambda p: hand(OPEN))
    results.append(check("catch + slow open", s.types(), ["catch", "drop"]))
    # closing slowly is not a catch
    s = Sim().run(0.6, lambda p: hand(OPEN)).run(0.5, lambda p: hand(HALF)).run(0.6, lambda p: hand(FIST))
    results.append(check("slow close", s.types(), []))
    # hold with one hand, swipe with the other, splash with the first
    s = Sim().run(0.6, lambda p: hand(OPEN, cx=0.3), lambda p: hand(OPEN, cx=1.0)) \
             .run(0.4, lambda p: hand(FIST, cx=0.3), lambda p: hand(OPEN, cx=1.0)) \
             .run(0.2, lambda p: hand(FIST, cx=0.3), lambda p: hand(OPEN, cx=1.0 - 0.25 * p)) \
             .run(0.6, lambda p: hand(FIST, cx=0.3), lambda p: hand(OPEN, cx=0.75)) \
             .run(0.2, lambda p: hand(OPEN, cx=0.3), lambda p: hand(OPEN, cx=0.75))
    results.append(check("hold + other hand swipes + splash", s.types(), ["catch", "swipe:left", "splash"]))
    # holding hand vanishes for 2 s: dropped
    s = Sim().run(0.6, lambda p: hand(OPEN)).run(0.4, lambda p: hand(FIST)).run(2.0)
    results.append(check("held hand lost", s.types(), ["catch", "drop"]))
    # held pose launches once
    s = Sim().run(2.0, lambda p: hand(TWO))
    results.append(check("hold two fingers", s.types(), ["pose:two"]))
    # rock works in typing mode, other poses don't
    s = Sim(control=False).run(2.0, lambda p: hand(TWO)).run(1.6, lambda p: hand(ROCK))
    results.append(check("typing mode: only rock", s.types(), ["pose:rock"]))
    # pinch + drag down scrolls
    s = Sim().run(0.5, lambda p: hand(OPEN, ok=True)).run(0.5, lambda p: hand(OPEN, ok=True, cy=0.65 + 0.15 * p))
    s.run(0.2, lambda p: hand(OPEN, cy=0.8))
    sc = sum(e["amount"] for _, e in s.events if e["type"] == "scroll")
    kinds = {e["type"] for _, e in s.events}
    results.append(check("pinch scroll (palms moved, events)", (round(sc, 1) >= 1.0, sorted(kinds)),
                         (True, ["scroll", "scroll_end"])))
    # loose pinch (only middle finger up) also scrolls
    s = Sim().run(0.5, lambda p: hand((0, 0, 1, 0, 0), ok=True)) \
             .run(0.5, lambda p: hand((0, 0, 1, 0, 0), ok=True, cy=0.65 - 0.15 * p))
    sc = sum(e["amount"] for _, e in s.events if e["type"] == "scroll")
    results.append(check("loose pinch scrolls up", sc <= -1.0, True))
    # two-hand pinch pulled apart zooms in
    s = Sim().run(0.4, lambda p: hand(OPEN, cx=0.5, ok=True), lambda p: hand(OPEN, cx=0.8, ok=True)) \
             .run(0.5, lambda p: hand(OPEN, cx=0.5 - 0.15 * p, ok=True), lambda p: hand(OPEN, cx=0.8 + 0.15 * p, ok=True))
    results.append(check("two-hand zoom", sorted(set(s.types())), ["zoom:1"]))
    # one finger held: pointer on, pinch clicks, open hand leaves pointer mode
    s = Sim().run(0.8, lambda p: hand(ONE)).run(0.3, lambda p: hand(ONE, cx=0.65 + 0.1 * p)) \
             .run(0.2, lambda p: hand(OPEN, ok=True, cx=0.75)).run(0.2, lambda p: hand(ONE, cx=0.75)) \
             .run(0.5, lambda p: hand(OPEN, cx=0.75))
    results.append(check("air pointer", s.types(), ["pointer_on", "click_down", "click_up", "pointer_off"]))
    # double pinch while pointing: the pointer holds still between the two clicks
    s = Sim().run(0.8, lambda p: hand(ONE)).run(0.15, lambda p: hand(OPEN, ok=True)) \
             .run(0.12, lambda p: hand(ONE, cx=0.65 + 0.004 * p)).run(0.15, lambda p: hand(OPEN, ok=True, cx=0.654))
    results.append(check("double pinch", [t for t in s.types() if t.startswith("click")],
                         ["click_down", "click_up", "click_down"]))
    # carry: catch, move the fist, flick open -> carry moves sum to roughly the fist travel, then splash
    sim = Sim()
    sim.run(0.6, lambda p: hand(OPEN)).run(0.3, lambda p: hand(FIST)) \
       .run(0.5, lambda p: hand(FIST, cx=0.65 + 0.2 * p)).run(0.2, lambda p: hand(FIST, cx=0.85)) \
       .run(0.2, lambda p: hand(OPEN, cx=0.85))
    carried = sum(e["dx"] for _, e in sim.events if e["type"] == "carry")
    results.append(check("carry + drop", (sim.types(), 0.12 < carried < 0.22),
                         (["catch", "splash"], True)))
    # menu (shelf) open: open-hand swipes are ignored, two-finger swipes and catches still work
    s = Sim()
    s.eng.menu = True
    s.run(0.6, lambda p: hand(OPEN)).run(0.2, lambda p: hand(OPEN, cx=0.65 + 0.25 * p)) \
     .run(0.6, lambda p: hand(TWO, cx=0.9)).run(0.2, lambda p: hand(TWO, cx=0.9, cy=0.65 + 0.25 * p)) \
     .run(0.6, lambda p: hand(OPEN, cx=0.9)).run(0.3, lambda p: hand(FIST, cx=0.9))
    results.append(check("shelf mode", s.types(), ["swipe:down", "catch"]))
    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    replay(sys.argv[1])
