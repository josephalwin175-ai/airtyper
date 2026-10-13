"""Gesture engine: turns per-frame hand landmarks into gesture events.

Pure logic with no Windows calls, so recordings can be replayed offline (see replay.py).

Three kinds of gesture:
  poses        hold a hand shape still until it fires (app launchers, rock, pinky, air pointer)
  motions      instant: swipes, two-finger swipes, push toward the camera
  transitions  one shape changing into another fast: catch (open -> fist), splash (fist -> open)
plus continuous pinch controls (scroll, volume twist, two-hand zoom), an air pointer, and "carry":
while a hand holds a caught item, its fist movement is reported so the item can be dropped where it is moved.

Each hand gets a track with a stable id (matched by nearest palm position from frame to frame),
so one hand can hold a caught item while the other swipes.
"""
import math
from collections import Counter, deque

import numpy as np

# ---------- tuning ----------
POSE_STABLE = (10, 7)        # a pose counts once seen in 7 of the last 10 frames
POSE_HOLD = {"rock": 1.2, "pinky": 2.2, "one": 0.45, "thumbs_down": 0.9}
POSE_HOLD_DEFAULT = 1.2      # s
POSE_STILL = 1.4             # palm sizes / s; launcher poses need a nearly still hand
POSE_COOLDOWN = 1.5
TRACK_MATCH = 0.3            # max palm-centre jump between frames that keeps a hand's identity
TRACK_GRACE = 0.5            # s a lost hand keeps its track
HOLD_GRACE = 1.5             # s a hand holding a caught item may vanish before the item drops
WARMUP = 0.3                 # s a new hand must be tracked before motions or catches count
SWIPE_WINDOW = 0.4           # s
SWIPE_DIST = 1.4             # palm sizes, horizontal
SWIPE_DIST_V = 1.8           # palm sizes, vertical (raising or dropping a hand is common, so stricter)
SWIPE_AXIS = 1.8             # the main axis must dominate the other by this much
SWIPE_REFRACTORY = 0.45
SWIPE_RETURN_BLOCK = 0.8     # s the opposite swipe is ignored: the hand travelling back to centre
PUSH_GROW = 1.3              # palm grows this much (moving toward the camera) ...
PUSH_DRIFT = 0.7             # ... while drifting less than this many palm sizes sideways
CATCH_TIME = 0.35            # s from last open frame to first fist frame
SPLASH_TIME = 0.3            # s from last fist frame to first open frame
PINCH_ON, PINCH_OFF = 0.32, 0.55
LOCK_SCROLL = 0.12           # palm sizes of vertical travel before a pinch becomes scrolling
LOCK_TWIST = 18.0            # degrees of twist before a pinch becomes volume control
VOLUME_STEP = 9.0            # degrees per volume step
PINCH_RELEASE_FRAMES = 2     # frames the fingers must stay apart before a pinch ends (stops flicker)
ZOOM_STEP = 1.3              # two-hand pinch distance ratio per zoom step
POINTER_EXIT = 0.3           # s of open hand that leaves pointer mode
CLICK_LOOKBACK = 0.12        # s; a click lands where the pointer was just before the pinch closed
DRAG_DELAY = 0.25            # s after a click before the pointer moves again (dragging)
CLICK_SETTLE = 0.4           # s after a click the pointer stays put, so a second pinch double-clicks ...
SETTLE_DIST = 0.012          # ... unless the hand clearly moves away
CARRY_LOOKBACK = 0.08        # s; a drop lands where the fist was just before it started opening

OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}

# finger pattern: thumb, index, middle, ring, pinky
GESTURE_PATTERNS = {
    (1, 1, 1, 1, 1): "open_palm",
    (0, 1, 1, 0, 0): "two",
    (1, 1, 1, 0, 0): "three",
    (0, 1, 1, 1, 0): "three",
    (0, 1, 1, 1, 1): "four",
    (1, 0, 0, 0, 0): "thumbs_up",
    (1, 0, 0, 0, 1): "call_me",
    (0, 1, 0, 0, 0): "one",
    (1, 1, 0, 0, 0): "l_shape",
    (0, 0, 0, 0, 0): "fist",
    (0, 1, 0, 0, 1): "rock",
    (1, 1, 0, 0, 1): "rock",
    (0, 0, 0, 0, 1): "pinky",
}
NOT_POSES = {"open_palm", "fist"}  # used by motions and catch/splash, never held as launchers


class OneEuro:
    """One Euro filter: heavy smoothing when still, almost none when moving fast."""

    def __init__(self, mincutoff=1.2, beta=0.05, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx += self._a(self.dcutoff, dt) * (dx - self.dx)
        a = self._a(self.mincutoff + self.beta * abs(self.dx), dt)
        self.x += a * (x - self.x)
        self.t = t
        return self.x

    def reset(self):
        self.t = None


def finger_pattern(lm):
    def P(i):
        return np.array([lm[i].x * 4 / 3, lm[i].y])
    w = P(0)
    palm = np.linalg.norm(P(9) - w) + 1e-6
    up = [int(np.linalg.norm(P(t) - w) > np.linalg.norm(P(p) - w) * 1.12)
          for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]
    thumb = int(np.linalg.norm(P(4) - P(17)) > np.linalg.norm(P(2) - P(17)) * 1.2
                and np.linalg.norm(P(4) - P(5)) > 0.38 * palm)
    return (thumb, *up)


def classify(lm, pattern=None):
    g = GESTURE_PATTERNS.get(pattern or finger_pattern(lm))
    if g == "thumbs_up":
        if lm[4].y < lm[2].y:
            return "thumbs_up"
        if lm[4].y > max(lm[i].y for i in (5, 9, 13, 17)):  # thumb tip below every knuckle
            return "thumbs_down"
        return None
    return g


def wrap_deg(a):
    return (a + 180) % 360 - 180


class Feat:
    """Per-frame features of one hand. Positions use x scaled by 4/3 so units are square."""
    __slots__ = ("t", "fingers", "up4", "pose", "palm", "center", "roll", "pinch", "ok", "knuckle", "pinch_pt")


def features(lm, t):
    def P(i):
        return np.array([lm[i].x * 4 / 3, lm[i].y])

    def P3(i):
        return np.array([lm[i].x * 4 / 3, lm[i].y, lm[i].z * 4 / 3])

    f = Feat()
    f.t = t
    f.fingers = finger_pattern(lm)
    f.up4 = sum(f.fingers[1:])
    f.pose = classify(lm, f.fingers)
    f.palm = float(np.linalg.norm(P(9) - P(0))) + 1e-6
    f.center = (P(0) + P(5) + P(9) + P(13) + P(17)) / 5
    v = P(17) - P(5)
    f.roll = math.degrees(math.atan2(v[1], v[0]))
    f.pinch = float(np.linalg.norm(P3(4) - P3(8)) / (np.linalg.norm(P3(0) - P3(9)) + 1e-6))
    f.ok = sum(f.fingers[2:]) >= 1  # thumb + index touch with any other finger up (not a fist)
    f.knuckle = (P(5) + P(9)) / 2   # index + middle knuckles stay put while thumb and index pinch
    f.pinch_pt = (P(4) + P(8)) / 2
    return f


def motion_kind(f):
    if f.up4 >= 4:
        return "palm"
    if f.fingers[1:] == (1, 1, 0, 0):
        return "two"
    return None


class Track:
    _next = 1

    def __init__(self, t):
        self.id = Track._next
        Track._next += 1
        self.born = t
        self.lost_since = None
        self.updated = False
        self.lm = None
        self.frames = deque(maxlen=45)
        self.poses = deque(maxlen=POSE_STABLE[0])
        self.pose, self.pose_start, self.pose_fired, self.pose_cool = None, t, False, 0.0
        self.anchor = t + WARMUP   # motion frames before this are ignored
        self.motion_until = 0.0
        self.block = {}            # swipe direction -> blocked until
        self.open_run = self.closed_run = 0
        self.open_first = self.closed_first = 0.0
        self.last_open = self.last_closed = -1e9
        self.pinching, self.pinch_on, self.pinch_mode = False, 0, None
        self.pinch_ref, self.pinch_done = None, 0
        self.pointer, self.mouse_down, self.click_on, self.down_t = False, False, 0, 0.0
        self.p_last, self.p_open_since = None, None
        self.p_hist = deque(maxlen=12)
        self.settle_until, self.settle_at = 0.0, None
        self.fx, self.fy = OneEuro(2.0, 20.0), OneEuro(2.0, 20.0)  # light smoothing, little lag when moving
        self.c_last = None
        self.c_hist = deque(maxlen=15)
        self.cfx, self.cfy = OneEuro(2.0, 20.0), OneEuro(2.0, 20.0)
        self.sfy = OneEuro(2.0, 20.0)  # pinch-scroll height
        self.s_last, self.pinch_ema, self.pinch_off_n = None, None, 0

    @property
    def last(self):
        return self.frames[-1] if self.frames else None


class Engine:
    def __init__(self):
        self.tracks = []
        self.holder = None      # id of the track holding a caught item
        self.zoom_ref = None
        self.menu = False       # a menu (clipboard shelf) is open: hand height picks, only catch/rock/2-finger swipes

    def reset(self, now):
        """Mode switch: forget in-progress gestures. The current pose must be released and re-made."""
        out = []
        for tr in self.tracks:
            if tr.pointer:
                out += self._pointer_off(tr)
            tr.pose_fired, tr.pose_cool = True, now + 1.0
            tr.pinching, tr.pinch_on, tr.pinch_mode = False, 0, None
            tr.anchor = now
        self.holder = None
        self.zoom_ref = None
        return out

    def track(self, tid):
        return next((tr for tr in self.tracks if tr.id == tid), None)

    def progress(self, now):
        """(pose id, 0..1) of the pose currently being held, or None."""
        best = None
        for tr in self.tracks:
            if tr.pose and not tr.pose_fired and tr.updated and now >= tr.pose_cool:
                p = min((now - tr.pose_start) / POSE_HOLD.get(tr.pose, POSE_HOLD_DEFAULT), 1.0)
                if best is None or p > best[1]:
                    best = (tr.pose, p)
        return best

    # ---------- per frame ----------
    def update(self, hands, now, control):
        """hands: list of landmark lists. control: gesture mode (False = typing, only rock counts)."""
        ev = []
        dets = [(lm, features(lm, now)) for lm in hands]
        pairs = sorted((float(np.linalg.norm(tr.last.center - f.center)), ti, di)
                       for ti, tr in enumerate(self.tracks) if tr.frames
                       for di, (_, f) in enumerate(dets))
        used_t, assigned = set(), {}
        for d, ti, di in pairs:
            if ti in used_t or di in assigned or d > TRACK_MATCH:
                continue
            used_t.add(ti)
            assigned[di] = self.tracks[ti]
        for di in range(len(dets)):
            if di not in assigned:
                tr = Track(now)
                self.tracks.append(tr)
                assigned[di] = tr

        live = set(id(tr) for tr in assigned.values())
        for tr in list(self.tracks):
            tr.updated = id(tr) in live
            if tr.updated:
                tr.lost_since = None
                continue
            if tr.lost_since is None:
                tr.lost_since = now
            grace = HOLD_GRACE if tr.id == self.holder else TRACK_GRACE
            if now - tr.lost_since > grace:
                if tr.id == self.holder:
                    self.holder = None
                    ev.append({"type": "drop", "reason": "lost"})
                if tr.pointer:
                    ev += self._pointer_off(tr)
                self.tracks.remove(tr)
            elif tr.pointer and tr.mouse_down:  # never leave the mouse button stuck down
                tr.mouse_down = False
                ev.append({"type": "click_up"})

        for di, (lm, f) in enumerate(dets):
            tr = assigned[di]
            tr.lm = lm
            tr.frames.append(f)
            tr.poses.append(f.pose)
            if not control:
                self._poses(tr, f, now, ev, allowed={"rock"})
                continue
            self._open_close(tr, f, now, ev)
            if tr.id == self.holder and f.up4 == 0:
                self._carry(tr, f, now, ev)
            if self.menu and not tr.pointer:
                if tr.id != self.holder:
                    self._motions(tr, f, now, ev, kinds=("two",))
                    self._poses(tr, f, now, ev, allowed={"rock"})
                continue
            if tr.pointer:
                self._pointer(tr, f, now, ev)
                continue
            self._pinch_state(tr, f, now, ev)
            busy = tr.pinching or tr.id == self.holder
            if not busy:
                self._motions(tr, f, now, ev)
                self._poses(tr, f, now, ev)
            else:
                tr.pose_start = now
        if control:
            self._pinch_actions([assigned[di] for di in range(len(dets))], ev)
        return ev

    # ---------- catch / splash ----------
    def _open_close(self, tr, f, now, ev):
        if f.up4 >= 3:
            tr.open_run += 1
            if tr.open_run == 1:
                tr.open_first = now
            if tr.open_run >= 2:
                if tr.id == self.holder:
                    gap = tr.open_first - tr.last_closed
                    self.holder = None
                    if gap <= SPLASH_TIME:
                        past = next(((hx, hy) for ht, hx, hy in reversed(tr.c_hist)
                                     if ht <= tr.last_closed - CARRY_LOOKBACK), None)
                        if past and tr.c_last:
                            ev.append({"type": "carry", "dx": past[0] - tr.c_last[0], "dy": past[1] - tr.c_last[1],
                                       "dt": 1 / 30, "rewind": True})
                    tr.anchor, tr.motion_until, tr.pose_cool = now, now + 0.6, now + 0.8
                    ev.append({"type": "splash", "track": tr.id} if gap <= SPLASH_TIME
                              else {"type": "drop", "reason": "slow"})
                tr.last_open = now
        else:
            tr.open_run = 0
        if f.up4 == 0:
            tr.closed_run += 1
            if tr.closed_run == 1:
                tr.closed_first = now
            if (tr.closed_run == 2 and self.holder is None and not tr.pointer and not tr.pinching
                    and now - tr.born > WARMUP and tr.closed_first - tr.last_open <= CATCH_TIME):
                self.holder = tr.id
                tr.motion_until = now + 0.6
                tr.c_last = None
                tr.c_hist.clear()
                tr.cfx.reset(); tr.cfy.reset()
                ev.append({"type": "catch", "track": tr.id})
            tr.last_closed = now
        else:
            tr.closed_run = 0

    # ---------- swipes / push ----------
    def _motions(self, tr, f, now, ev, kinds=("palm", "two")):
        if now < tr.motion_until:
            return
        kind = motion_kind(f)
        if kind not in kinds:
            return
        start = None
        for g in reversed(tr.frames):
            if g.t < tr.anchor or now - g.t > SWIPE_WINDOW or motion_kind(g) != kind:
                break
            start = g
        if start is None or start is f:
            return
        palm = float(np.median([g.palm for g in tr.frames if g.t >= start.t]))
        dx, dy = (f.center - start.center) / palm
        dirn = None
        if abs(dx) >= SWIPE_DIST and abs(dx) >= SWIPE_AXIS * abs(dy):
            dirn = "right" if dx > 0 else "left"
        elif abs(dy) >= SWIPE_DIST_V and abs(dy) >= SWIPE_AXIS * abs(dx):
            dirn = "down" if dy > 0 else "up"
        if dirn:
            tr.anchor = now
            if now < tr.block.get(dirn, 0):
                return
            ev.append({"type": "swipe", "dir": dirn, "two": kind == "two", "track": tr.id})
            tr.motion_until = now + SWIPE_REFRACTORY
            tr.block[OPPOSITE[dirn]] = now + SWIPE_RETURN_BLOCK
            tr.pose_start = now
            return
        if kind == "palm" and now - start.t >= 0.12:
            drift = float(np.linalg.norm(f.center - start.center)) / start.palm
            if f.palm / start.palm >= PUSH_GROW and drift <= PUSH_DRIFT:
                ev.append({"type": "push", "track": tr.id})
                tr.anchor, tr.motion_until = now, now + 1.0

    # ---------- held poses ----------
    def _poses(self, tr, f, now, ev, allowed=None):
        stable, n = Counter(tr.poses).most_common(1)[0]
        if n < POSE_STABLE[1] or stable in NOT_POSES or (allowed is not None and stable not in allowed):
            stable = None
        if stable != tr.pose:
            tr.pose, tr.pose_start, tr.pose_fired = stable, now, False
        elif stable:
            recent = [g for g in tr.frames if now - g.t <= 0.3]
            if len(recent) >= 2:
                speed = (float(np.linalg.norm(recent[-1].center - recent[0].center)) / f.palm
                         / max(recent[-1].t - recent[0].t, 1e-3))
                if speed > POSE_STILL:
                    tr.pose_start = now
        if (stable and not tr.pose_fired and now >= tr.pose_cool
                and now - tr.pose_start >= POSE_HOLD.get(stable, POSE_HOLD_DEFAULT)):
            tr.pose_fired = True
            tr.pose_cool = now + POSE_COOLDOWN
            if stable == "one" and allowed is None:
                tr.pointer, tr.mouse_down, tr.click_on = True, False, 0
                tr.p_last, tr.p_open_since = None, None
                tr.fx.reset(); tr.fy.reset()
                tr.p_hist.clear()
                ev.append({"type": "pointer_on", "track": tr.id})
            else:
                ev.append({"type": "pose", "id": stable, "track": tr.id})

    # ---------- carry: the fist holding an item moves it ----------
    def _carry(self, tr, f, now, ev):
        x, y = tr.cfx(float(f.center[0]), now), tr.cfy(float(f.center[1]), now)
        tr.c_hist.append((now, x, y))
        if tr.c_last is not None:
            ev.append({"type": "carry", "dx": x - tr.c_last[0], "dy": y - tr.c_last[1], "dt": self._dt(tr)})
        tr.c_last = (x, y)

    @staticmethod
    def _dt(tr):
        return tr.frames[-1].t - tr.frames[-2].t if len(tr.frames) >= 2 else 1 / 30

    # ---------- air pointer ----------
    def _pointer_off(self, tr):
        out = []
        if tr.mouse_down:
            out.append({"type": "click_up"})
        tr.pointer = tr.mouse_down = False
        tr.p_last = None
        out.append({"type": "pointer_off", "track": tr.id})
        return out

    def _pointer(self, tr, f, now, ev):
        if f.up4 >= 4:
            if tr.p_open_since is None:
                tr.p_open_since = now
            if now - tr.p_open_since >= POINTER_EXIT:
                ev += self._pointer_off(tr)
                tr.anchor = now
                tr.pose_fired = True
                return
        else:
            tr.p_open_since = None
        x, y = tr.fx(float(f.knuckle[0]), now), tr.fy(float(f.knuckle[1]), now)
        tr.p_hist.append((now, x, y))
        if not tr.mouse_down:
            tr.click_on = tr.click_on + 1 if f.pinch < PINCH_ON else 0
            if tr.click_on >= 2:
                # land the click where the pointer was before the fingers started closing
                past = next(((hx, hy) for ht, hx, hy in reversed(tr.p_hist) if now - ht >= CLICK_LOOKBACK), None)
                if past and tr.p_last and not (tr.settle_at is not None and now < tr.settle_until):
                    ev.append({"type": "pointer", "dx": past[0] - tr.p_last[0], "dy": past[1] - tr.p_last[1],
                               "dt": 1 / 30, "rewind": True})
                    tr.p_last = past
                tr.mouse_down, tr.down_t = True, now
                ev.append({"type": "click_down", "track": tr.id})
                return
        elif f.pinch > PINCH_OFF:
            tr.mouse_down = False
            ev.append({"type": "click_up", "track": tr.id})
            tr.settle_until, tr.settle_at = now + CLICK_SETTLE, tr.p_last
        if tr.settle_at is not None and now < tr.settle_until:
            if math.hypot(x - tr.settle_at[0], y - tr.settle_at[1]) < SETTLE_DIST:
                return  # hold still for a possible double / triple pinch
            tr.settle_until = 0.0
        if tr.p_last is not None and (not tr.mouse_down or now - tr.down_t > DRAG_DELAY):
            ev.append({"type": "pointer", "dx": x - tr.p_last[0], "dy": y - tr.p_last[1], "dt": self._dt(tr)})
        tr.p_last = (x, y)

    # ---------- pinch: scroll, volume, zoom ----------
    def _pinch_state(self, tr, f, now, ev):
        tr.pinch_ema = f.pinch if tr.pinch_ema is None else tr.pinch_ema * 0.4 + f.pinch * 0.6
        if not tr.pinching:
            tr.pinch_on = tr.pinch_on + 1 if (tr.pinch_ema < PINCH_ON and f.ok) else 0
            if tr.pinch_on >= 2:
                tr.pinching, tr.pinch_mode, tr.pinch_done, tr.pinch_off_n = True, None, 0, 0
                tr.pinch_ref = (float(f.center[1]), f.roll, f.palm)
                tr.sfy.reset()
                tr.s_last = None
            return
        tr.pinch_off_n = tr.pinch_off_n + 1 if (tr.pinch_ema > PINCH_OFF or sum(f.fingers[2:]) == 0) else 0
        if tr.pinch_off_n >= PINCH_RELEASE_FRAMES:
            if tr.pinch_mode == "scroll":
                ev.append({"type": "scroll_end", "track": tr.id})
            tr.pinching, tr.pinch_on, tr.pinch_mode = False, 0, None
            tr.anchor, tr.pose_start = now, now

    def _pinch_actions(self, updated, ev):
        p = [tr for tr in updated if tr.pinching and not tr.pointer]
        if len(p) >= 2:
            a, b = p[:2]
            d = float(np.linalg.norm(a.last.pinch_pt - b.last.pinch_pt)) / ((a.last.palm + b.last.palm) / 2)
            if self.zoom_ref is None or a.pinch_mode != "zoom" or b.pinch_mode != "zoom":
                self.zoom_ref = d
            elif d / self.zoom_ref >= ZOOM_STEP:
                ev.append({"type": "zoom", "n": 1})
                self.zoom_ref = d
            elif d / self.zoom_ref <= 1 / ZOOM_STEP:
                ev.append({"type": "zoom", "n": -1})
                self.zoom_ref = d
            a.pinch_mode = b.pinch_mode = "zoom"
            return
        self.zoom_ref = None
        for tr in p:
            if tr.pinch_mode == "zoom":
                continue
            f = tr.last
            y = tr.sfy(float(f.center[1]), f.t)
            y0, r0, palm0 = tr.pinch_ref
            dr = wrap_deg(f.roll - r0)
            if tr.pinch_mode is None:
                if abs(y - y0) / palm0 >= LOCK_SCROLL:
                    tr.pinch_mode, tr.s_last = "scroll", y
                elif abs(dr) >= LOCK_TWIST:
                    tr.pinch_mode, tr.pinch_ref, tr.pinch_done = "volume", (y, f.roll, palm0), 0
                continue
            if tr.pinch_mode == "scroll":
                # continuous: palm sizes moved since the last frame (+ = hand moved down)
                d = (y - tr.s_last) / palm0
                tr.s_last = y
                if d:
                    ev.append({"type": "scroll", "amount": d, "dt": self._dt(tr), "track": tr.id})
                continue
            n = int(dr / VOLUME_STEP)
            if n != tr.pinch_done:
                ev.append({"type": "volume", "n": n - tr.pinch_done, "track": tr.id})
                tr.pinch_done = n
