"""AirTyper - type in the air with your webcam, and launch apps with hand gestures.

TYPING
  Raise your hand: the cursor starts at the centre of the keyboard. Small hand movements steer it.
  Pinch thumb + index to press. Keystrokes go to whatever window you last clicked.
  2-hand mode: left hand covers the left half of the keyboard, right hand the right half.
  Hybrid: your physical keyboard keeps working; hand typing pauses briefly after real keystrokes.

AI MODE (hold a gesture until the bar fills, ~1.6 s)  - edit gestures.json to remap
  Open palm -> Notepad     Two fingers -> Chrome     Three -> Claude     Four -> Antigravity
  Thumbs up -> Explorer    Call-me -> Spotify        One finger -> Calculator
  L-shape -> Screenshot    Pinky only -> close the active app (~2.4 s hold)
  Fist -> back to typing   Rock (index+pinky) -> toggle AI mode

WINDOW
  Drag the header to move it (position is remembered). Buttons: 1H/2H, AI, close.

HOTKEYS (Ctrl+Alt + ...)
  Up/Down sensitivity   R re-centre   M pinch/dwell   D one/two hands   A AI mode   H hide   Q quit
"""
import ctypes
import json
import math
import os
import subprocess
import threading
import time
from collections import Counter, deque
from ctypes import wintypes

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
from pynput.keyboard import Controller, GlobalHotKeys, Key, Listener

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "hand_landmarker.task")
SETTINGS = os.path.join(HERE, "settings.json")
GESTURES = os.path.join(HERE, "gestures.json")
WIN = "AirTyper"

# ---------- tuning ----------
# width of the hand-movement box (fraction of camera width) spanning the whole keyboard; smaller = more sensitive
SENS_LEVELS = [0.34, 0.27, 0.21, 0.16, 0.12]
SENS_DEFAULT = 3
PINCH_ON, PINCH_OFF = 0.32, 0.55  # thumb-index distance / palm size, wide hysteresis
PINCH_FRAMES = 2                  # frames a pinch must hold before it counts
REFRACTORY = 0.18                 # s; minimum gap between two presses from the same hand
LOST_GRACE = 0.30                 # s; brief tracking drop-outs do not release the pinch
LOOKBACK = 0.12                   # s; type the key hovered just before the pinch started
STICKY = 0.22                     # fraction of a key the cursor must cross before hover switches
DWELL_TIME = 0.7
REPEAT_DELAY, REPEAT_RATE = 0.5, 0.09
HYBRID_PAUSE = 1.2                # s; hand typing pauses after you touch the physical keyboard
GESTURE_HOLD = 1.6                # s; how long to hold a gesture in AI mode
GESTURE_HOLD_CLOSE = 2.4          # s; closing an app is destructive, so it takes longer
GESTURE_STABLE = (10, 8)          # a gesture counts once seen in 8 of the last 10 frames
GESTURE_COOLDOWN = 2.5
EASE = 24.0                       # display cursor easing rate (higher = snappier)
RENDER_FPS = 60

# ---------- palette (BGR) ----------
BG = (24, 13, 6)
PANEL = (44, 26, 11)
GRID = (44, 28, 13)
CYAN = (255, 212, 80)
CYAN_DIM = (130, 92, 36)
GOLD = (48, 178, 255)
WHITE = (255, 246, 232)
RED = (96, 90, 255)


def rgb(c):
    return (c[2], c[1], c[0])


def load_font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", n), size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


user32 = ctypes.windll.user32
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
user32.SendMessageW.argtypes = [wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.FindWindowW.restype = wintypes.HWND
user32.GetForegroundWindow.restype = wintypes.HWND


def dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        user32.SetProcessDPIAware()


def work_area():
    rc = wintypes.RECT()
    user32.SystemParametersInfoW(0x30, 0, ctypes.byref(rc), 0)
    return rc.left, rc.top, rc.right, rc.bottom


def window_title(hwnd):
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value


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


WORDS = """the be to of and a in that have i it for not on with he as you do at this but his by from they we
say her she or an will my one all would there their what so up out if about who get which go me when make can
like time no just him know take people into year your good some could them see other than then now look only
come its over think also back after use two how our work first well way even new want because any these give
day most us is are was were been has had did said am hello thanks thank please yes okay sorry great love
today tomorrow tonight morning night meeting email call message where why here very really more much many
again still never always maybe something nothing everything someone anyone keyboard typing type hand hands
air camera finger point pinch screen window open close send reply write read name need help home world
""".split()

# ---------- keyboard ----------
SHIFTED = dict(zip("1234567890,.", "!@#$%^&*()?!"))
ROWS = [
    [(c, 1) for c in "1234567890"],
    [(c, 1) for c in "qwertyuiop"],
    [(c, 1) for c in "asdfghjkl"],
    [("SHIFT", 1.5)] + [(c, 1) for c in "zxcvbnm"] + [("DEL", 1.5)],
    [("MODE", 1.5), (",", 1), ("SPACE", 5), (".", 1), ("ENTER", 1.5)],
]
NAMED = {"DEL": "⌫", "ENTER": "ENTER", "SPACE": "space", "SHIFT": "⇧", "MODE": "MODE"}


def key_label(kid, shift, mode):
    if len(kid) == 1:
        if shift and kid in SHIFTED:
            return SHIFTED[kid]
        return kid.upper() if (shift or kid.isalpha()) else kid
    if kid == "MODE":
        return "DWELL" if mode == "dwell" else "PINCH"
    return NAMED[kid]


# ---------- gestures ----------
# finger pattern: thumb, index, middle, ring, pinky
GESTURE_PATTERNS = {
    (1, 1, 1, 1, 1): "open_palm",
    (0, 1, 1, 0, 0): "two",
    (0, 1, 1, 1, 0): "three",
    (1, 1, 1, 0, 0): "three",
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
GESTURE_INFO = [  # (id, label, pattern shown on the card)
    ("open_palm", "Open palm", (1, 1, 1, 1, 1)),
    ("two", "Two fingers", (0, 1, 1, 0, 0)),
    ("three", "Three fingers", (0, 1, 1, 1, 0)),
    ("four", "Four fingers", (0, 1, 1, 1, 1)),
    ("thumbs_up", "Thumbs up", (1, 0, 0, 0, 0)),
    ("call_me", "Call me", (1, 0, 0, 0, 1)),
    ("one", "One finger", (0, 1, 0, 0, 0)),
    ("l_shape", "L shape", (1, 1, 0, 0, 0)),
    ("pinky", "Pinky only", (0, 0, 0, 0, 1)),
    ("fist", "Fist", (0, 0, 0, 0, 0)),
    ("rock", "Rock", (0, 1, 0, 0, 1)),
]
BUILTIN_ACTIONS = {"pinky": "Close active app", "fist": "Back to typing", "rock": "Toggle AI mode"}
DEFAULT_GESTURES = {
    "open_palm": {"name": "Notepad", "run": "shell:AppsFolder\\Microsoft.WindowsNotepad_8wekyb3d8bbwe!App"},
    "two": {"name": "Chrome", "run": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"},
    "three": {"name": "Claude", "run": "shell:AppsFolder\\Claude_pzs8sxrjxfjjc!Claude"},
    "four": {"name": "Antigravity", "run": "%LOCALAPPDATA%\\Programs\\Antigravity\\Antigravity.exe"},
    "thumbs_up": {"name": "File Explorer", "run": "explorer.exe"},
    "call_me": {"name": "Spotify", "run": "%APPDATA%\\Spotify\\Spotify.exe"},
    "one": {"name": "Calculator", "run": "shell:AppsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"},
    "l_shape": {"name": "Screenshot", "keys": "win+shift+s"},
}


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


def classify(lm):
    g = GESTURE_PATTERNS.get(finger_pattern(lm))
    if g == "thumbs_up" and not lm[4].y < lm[2].y:  # thumb must point up
        return None
    return g


# ---------- camera + detection thread ----------
class Tracker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.frame = None
        self.hands = []  # list of (landmarks, "Left"/"Right")
        self.seq = 0
        self.fps = 0.0
        self.error = None
        self.ready = False
        self.alive = True

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.error = ("Could not open the webcam.\n\nClose other apps that are using the camera "
                          "(Zoom, Teams, Camera) and start AirTyper again.")
            return
        if not os.path.exists(MODEL):
            self.error = f"The hand-tracking model is missing:\n{MODEL}"
            cap.release()
            return
        landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))
        t0 = last = time.perf_counter()
        last_ts = -1
        self.ready = True
        try:
            while self.alive:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)
                now = time.perf_counter()
                ts = max(int((now - t0) * 1000), last_ts + 1)
                last_ts = ts
                res = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
                hands = []
                for i, lm in enumerate(res.hand_landmarks):
                    label = res.handedness[i][0].category_name if i < len(res.handedness) else ""
                    hands.append((lm, label))
                dt = now - last
                last = now
                with self.lock:
                    self.frame, self.hands = frame, hands
                    self.seq += 1
                    if dt > 0:
                        self.fps = self.fps * 0.9 + (1 / dt) * 0.1
        finally:
            cap.release()
            landmarker.close()

    def latest(self):
        with self.lock:
            return self.seq, self.frame, list(self.hands), self.fps


# ---------- per-hand typing state ----------
class Hand:
    def __init__(self, name):
        self.name = name  # "S" single, "L", "R"
        self.fx, self.fy = OneEuro(), OneEuro()
        self.lost_since = None
        self.reset()

    def reset(self):
        self.fx.reset(); self.fy.reset()
        self.lm = None
        self.cursor = None
        self.disp = None
        self.ratio = None
        self.pinched = False
        self.on_frames = 0
        self.pinch_amt = 0.0
        self.hover = None
        self.hover_hist = deque(maxlen=30)
        self.held_key = None
        self.held_since = self.last_repeat = self.last_emit = 0.0
        self.dwell_key, self.dwell_start = None, 0.0
        self.recenter = True
        self.zone_c = (0.5, 0.5)


class AirTyper:
    def __init__(self):
        dpi_aware()
        self.settings = load_json(SETTINGS, {})
        self.gestures = load_json(GESTURES, None)
        if not isinstance(self.gestures, dict):
            self.gestures = DEFAULT_GESTURES
            save_json(GESTURES, self.gestures)

        l, t, r, b = work_area()
        self.W = W = int(min(max((r - l) * 0.21, 380), 470))
        self.pad = pad = 10
        self.gap = gap = 5

        self.f_title = load_font(["bahnschrift.ttf", "seguisb.ttf"], 20)
        self.f_key = load_font(["bahnschrift.ttf", "segoeui.ttf"], int(W / 22))
        self.f_small = load_font(["bahnschrift.ttf", "segoeui.ttf"], 11)
        self.f_btn = load_font(["bahnschrift.ttf", "segoeui.ttf"], 12)
        self.f_text = load_font(["consola.ttf", "segoeui.ttf"], 18)
        self.f_sug = load_font(["bahnschrift.ttf", "segoeui.ttf"], 14)
        self.f_pop = load_font(["bahnschrift.ttf", "seguisb.ttf"], int(W / 12))
        self.f_sym = load_font(["seguisym.ttf", "segoeui.ttf"], int(W / 20))

        # layout
        self.header_h = 40
        self.cam_y = self.header_h
        self.cam_h = int(W * 0.36)
        self.tgt_y = self.cam_y + self.cam_h + 8
        self.bar_y = self.tgt_y + 16
        self.bar_h = 36
        self.sug_y = self.bar_y + self.bar_h + 8
        self.sug_h = 30
        self.kb_y = self.sug_y + self.sug_h + gap
        self.unit = (W - 2 * pad - 9 * gap) / 10
        self.key_h = int(self.unit * 0.92)
        self.kb_h = 5 * self.key_h + 4 * gap
        self.footer_y = self.kb_y + self.kb_h + 8
        self.H = H = self.footer_y + 22
        pos = self.settings.get("pos")
        if pos and l - W // 2 < pos[0] < r - W // 2 and t - 10 < pos[1] < b - 60:
            self.pos = tuple(pos)
        else:
            self.pos = (r - W - 12, b - H - 12)

        # header buttons
        bx = W - pad
        self.buttons = {}
        for name, w in (("close", 26), ("ai", 34), ("hands", 40)):
            self.buttons[name] = (bx - w, 8, bx, 32)
            bx -= w + 6

        # keys
        self.keys = []
        for ri, row in enumerate(ROWS):
            units = sum(u for _, u in row)
            rw = units * self.unit + (units - 1) * gap
            kx = pad + (W - 2 * pad - rw) / 2
            ky = self.kb_y + ri * (self.key_h + gap)
            for kid, u in row:
                kw = u * self.unit + (u - 1) * gap
                self.keys.append({"id": kid, "r": (int(kx), int(ky), int(kx + kw), int(ky + self.key_h))})
                kx += kw + gap
        sw3 = (W - 2 * pad - 2 * gap) / 3
        self.sug_keys = [{"id": f"SUG{i}", "word": "", "r": (int(pad + i * (sw3 + gap)), self.sug_y,
                                                          int(pad + i * (sw3 + gap) + sw3), self.sug_y + self.sug_h)}
                         for i in range(3)]
        self.aim = (pad, self.sug_y, W - pad, self.kb_y + self.kb_h)
        self.kb_cache = {}

        self.out = Controller()
        self.mode = self.settings.get("mode", "pinch")
        self.dual = bool(self.settings.get("dual", False))
        self.sens = int(np.clip(self.settings.get("sens", SENS_DEFAULT), 0, len(SENS_LEVELS) - 1))
        self.ai_mode = False
        self.shift = False
        self.buffer = ""
        self.learned = {}
        self.running = True
        self.hidden = False
        self.toggle_hide = False

        self.hands = {"S": Hand("S"), "L": Hand("L"), "R": Hand("R")}
        self.vis = []  # landmarks drawn this frame
        self.flash = {}
        self.ripples = []
        self.target, self.target_t = "", 0.0
        self.hwnd = None
        self.toast, self.toast_t = "", 0.0
        self.last_real_key = 0.0
        self.sug_dirty = False
        self.booting = True

        self.g_hist = deque(maxlen=GESTURE_STABLE[0])
        self.g_cur, self.g_start, self.g_fired = None, 0.0, False
        self.g_last_fire = 0.0

        self.base = self.make_base()
        self.update_suggestions()

        self.hotkeys = GlobalHotKeys({
            "<ctrl>+<alt>+q": self.quit,
            "<ctrl>+<alt>+m": self.switch_mode,
            "<ctrl>+<alt>+h": self.request_hide,
            "<ctrl>+<alt>+r": self.request_recenter,
            "<ctrl>+<alt>+d": self.toggle_dual,
            "<ctrl>+<alt>+a": self.toggle_ai,
            "<ctrl>+<alt>+<up>": lambda: self.change_sens(1),
            "<ctrl>+<alt>+<down>": lambda: self.change_sens(-1),
        })
        self.hotkeys.start()
        self.real_keys = Listener(on_press=self.on_real_key)
        self.real_keys.start()

    # ---------- controls ----------
    def quit(self):
        self.running = False

    def switch_mode(self):
        self.mode = "dwell" if self.mode == "pinch" else "pinch"
        self.say(f"{self.mode.upper()} MODE")

    def request_hide(self):
        self.toggle_hide = True

    def request_recenter(self):
        for h in self.hands.values():
            h.recenter = True
        self.say("RE-CENTRED")

    def toggle_dual(self):
        self.dual = not self.dual
        for h in self.hands.values():
            h.reset()
        self.say("TWO-HAND TYPING" if self.dual else "ONE-HAND TYPING")

    def toggle_ai(self):
        self.ai_mode = not self.ai_mode
        for h in self.hands.values():
            h.reset()
        self.g_cur, self.g_fired = None, True  # require a fresh gesture
        self.say("AI MODE: HOLD A GESTURE" if self.ai_mode else "TYPING MODE")

    def change_sens(self, d):
        self.sens = int(np.clip(self.sens + d, 0, len(SENS_LEVELS) - 1))
        self.say(f"SENSITIVITY {self.sens + 1}/{len(SENS_LEVELS)}")

    def say(self, msg):
        self.toast, self.toast_t = msg, time.perf_counter()

    def save_settings(self):
        if self.hwnd:
            rc = wintypes.RECT()
            user32.GetWindowRect(self.hwnd, ctypes.byref(rc))
            self.pos = (rc.left, rc.top)
        save_json(SETTINGS, {"pos": list(self.pos), "sens": self.sens, "dual": self.dual, "mode": self.mode})

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x0, y0, x1, y1) in self.buttons.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                {"close": self.quit, "ai": self.toggle_ai, "hands": self.toggle_dual}[name]()
                return
        if y < self.header_h and self.hwnd:
            # native window drag from the header
            user32.ReleaseCapture()
            user32.SendMessageW(self.hwnd, 0x00A1, 2, 0)  # WM_NCLBUTTONDOWN, HTCAPTION
            self.save_settings()

    # ---------- hybrid: physical keyboard ----------
    def on_real_key(self, key, injected=False):
        if injected:
            return
        self.last_real_key = time.perf_counter()
        ch = getattr(key, "char", None)
        if ch and ch.isprintable():
            self.buffer = (self.buffer + ch)[-200:]
        elif key == Key.space:
            self.learn_last_word()
            self.buffer += " "
        elif key == Key.backspace:
            self.buffer = self.buffer[:-1]
        elif key == Key.enter:
            self.learn_last_word()
            self.buffer += " | "
        else:
            return
        self.sug_dirty = True

    def hand_paused(self, now):
        return now - self.last_real_key < HYBRID_PAUSE

    # ---------- static art ----------
    def make_base(self):
        W, H = self.W, self.H
        img = np.full((H, W, 3), BG, np.uint8)
        for x in range(0, W, 24):
            cv2.line(img, (x, 0), (x, H), GRID, 1)
        for y in range(0, H, 24):
            cv2.line(img, (0, y), (W, y), GRID, 1)
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), CYAN_DIM, 1)
        L = 16
        for (px, py, dx, dy) in ((0, 0, 1, 1), (W - 1, 0, -1, 1), (0, H - 1, 1, -1), (W - 1, H - 1, -1, -1)):
            cv2.line(img, (px, py), (px + L * dx, py), CYAN, 2)
            cv2.line(img, (px, py), (px, py + L * dy), CYAN, 2)
        cv2.line(img, (self.pad, self.header_h - 3), (W - self.pad, self.header_h - 3), CYAN_DIM, 1)
        for i in range(3):  # grip dots: the header is draggable
            for j in range(2):
                cv2.circle(img, (self.pad + 2 + j * 5, 14 + i * 6), 1, CYAN, -1)
        cv2.rectangle(img, (self.pad, self.bar_y), (W - self.pad, self.bar_y + self.bar_h), PANEL, -1)
        cv2.rectangle(img, (self.pad, self.bar_y), (W - self.pad, self.bar_y + self.bar_h), CYAN_DIM, 1)
        cv2.line(img, (self.pad, self.bar_y), (self.pad, self.bar_y + self.bar_h), CYAN, 2)
        pil = Image.fromarray(img[..., ::-1].copy())
        d = ImageDraw.Draw(pil)
        d.text((self.pad + 14, 9), "AIR", font=self.f_title, fill=rgb(WHITE))
        aw = d.textlength("AIR", font=self.f_title)
        d.text((self.pad + 14 + aw, 9), "TYPER", font=self.f_title, fill=rgb(CYAN))
        foot = "CTRL+ALT +  UP/DN sens · R centre · M mode · D 2H · A AI · Q quit"
        d.text((self.pad, self.footer_y + 2), foot, font=self.f_small, fill=rgb(CYAN_DIM))
        return np.array(pil)[..., ::-1].copy()

    def kb_layer(self):
        key = (self.shift, self.mode)
        if key in self.kb_cache:
            return self.kb_cache[key]
        layer = np.zeros((self.H, self.W, 3), np.uint8)
        for k in self.keys:
            x0, y0, x1, y1 = k["r"]
            cv2.rectangle(layer, (x0, y0), (x1, y1), PANEL, -1)
            cv2.rectangle(layer, (x0, y0), (x1, y1), CYAN_DIM, 1)
            cv2.line(layer, (x0 + 3, y1), (x1 - 3, y1), CYAN, 1)
        pil = Image.fromarray(layer[..., ::-1].copy())
        d = ImageDraw.Draw(pil)
        for k in self.keys:
            lab = key_label(k["id"], self.shift, self.mode)
            col = WHITE if len(k["id"]) == 1 else CYAN
            if k["id"] == "MODE" or (k["id"] == "SHIFT" and self.shift):
                col = GOLD
            x0, y0, x1, y1 = k["r"]
            d.text(((x0 + x1) / 2, (y0 + y1) / 2), lab, font=self.font_for(lab), fill=rgb(col), anchor="mm")
        out = np.array(pil)[..., ::-1].copy()
        self.kb_cache[key] = out
        return out

    def font_for(self, lab):
        if lab in ("⌫", "⇧"):
            return self.f_sym
        return self.f_key if len(lab) == 1 else self.f_small

    # ---------- window ----------
    def setup_window(self, frame):
        prev = user32.GetForegroundWindow()
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WIN, frame)
        cv2.waitKey(1)
        cv2.setMouseCallback(WIN, self.on_mouse)
        hwnd = user32.FindWindowW(None, WIN)
        if hwnd:
            style = user32.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF
            style &= ~(0x00C00000 | 0x00040000 | 0x00080000 | 0x00020000 | 0x00010000)
            user32.SetWindowLongW(hwnd, -16, ctypes.c_long(style).value)
            ex = (user32.GetWindowLongW(hwnd, -20) & 0xFFFFFFFF) | 0x08000000 | 0x00000008 | 0x00000080
            user32.SetWindowLongW(hwnd, -20, ctypes.c_long(ex).value)
            user32.SetWindowPos(hwnd, wintypes.HWND(-1), self.pos[0], self.pos[1], self.W, self.H,
                                0x0020 | 0x0010 | 0x0040)
        if prev and prev != hwnd:
            user32.SetForegroundWindow(prev)
        self.hwnd = hwnd

    # ---------- words ----------
    def current_word(self):
        w = ""
        for ch in reversed(self.buffer):
            if not ch.isalpha() and ch != "'":
                break
            w = ch + w
        return w

    def update_suggestions(self):
        prefix = self.current_word().lower()
        trusted = [w for w, n in sorted(self.learned.items(), key=lambda kv: -kv[1]) if n >= 3]
        pool = WORDS + trusted if not prefix else trusted + WORDS
        seen, picks = set(), []
        for w in pool:
            if w in seen or w == prefix or not w.startswith(prefix):
                continue
            seen.add(w)
            picks.append(w)
            if len(picks) == 3:
                break
        for i, sk in enumerate(self.sug_keys):
            sk["word"] = picks[i] if i < len(picks) else ""

    def learn_last_word(self):
        w = self.current_word().lower()
        if 3 <= len(w) <= 15 and any(v in w for v in "aeiouy"):
            self.learned[w] = self.learned.get(w, 0) + 1

    # ---------- output ----------
    def emit(self, k):
        kid = k["id"]
        now = time.perf_counter()
        self.flash[kid] = now
        x0, y0, x1, y1 = k["r"]
        self.ripples.append(((x0 + x1) // 2, (y0 + y1) // 2, now))
        if kid == "SHIFT":
            self.shift = not self.shift
            return
        if kid == "MODE":
            self.switch_mode()
            return
        if self.hwnd and user32.GetForegroundWindow() == self.hwnd:
            return
        if kid.startswith("SUG"):
            word = k["word"]
            if not word:
                return
            rest = word[len(self.current_word()):]
            self.out.type(rest + " ")
            self.buffer += rest
            self.learn_last_word()
            self.buffer += " "
        elif kid == "DEL":
            self.out.tap(Key.backspace)
            self.buffer = self.buffer[:-1]
        elif kid == "ENTER":
            self.learn_last_word()
            self.out.tap(Key.enter)
            self.buffer += " | "
        elif kid == "SPACE":
            self.learn_last_word()
            self.out.tap(Key.space)
            self.buffer += " "
        else:
            ch = SHIFTED.get(kid, kid.upper()) if self.shift else kid
            self.out.type(ch)
            self.buffer += ch
            self.shift = False
        self.buffer = self.buffer[-200:]
        self.update_suggestions()

    def send_combo(self, combo):
        names = {"win": Key.cmd, "shift": Key.shift, "ctrl": Key.ctrl, "alt": Key.alt}
        parts = combo.lower().split("+")
        mods = [names[p] for p in parts[:-1] if p in names]
        for m in mods:
            self.out.press(m)
        try:
            self.out.tap(parts[-1])
        finally:
            for m in reversed(mods):
                self.out.release(m)

    def run_gesture(self, g):
        a = self.gestures.get(g)
        if not a:
            return
        name = a.get("name", g)
        try:
            if "keys" in a:
                self.send_combo(a["keys"])
            else:
                run = os.path.expandvars(a["run"])
                if run.lower().startswith("shell:"):
                    subprocess.Popen(["explorer.exe", run])
                else:
                    os.startfile(run)
            self.say(f"OPENING {name.upper()}")
        except Exception:
            self.say(f"COULD NOT OPEN {name.upper()}")

    # ---------- hand logic ----------
    def aim_for(self, h):
        ax0, ay0, ax1, ay1 = self.aim
        if h.name == "L":
            return ax0, ay0, ax0 + (ax1 - ax0) * 0.56, ay1
        if h.name == "R":
            return ax0 + (ax1 - ax0) * 0.44, ay0, ax1, ay1
        return self.aim

    def zone_dims(self, h):
        ax0, ay0, ax1, ay1 = self.aim_for(h)
        zw = SENS_LEVELS[self.sens] * (ax1 - ax0) / (self.aim[2] - self.aim[0])
        zh = zw * (4 / 3) * ((ay1 - ay0) / (ax1 - ax0))
        return zw, zh

    def hit(self, px, py, prev=None):
        if prev is not None and not (prev["id"].startswith("SUG") and not prev["word"]):
            x0, y0, x1, y1 = prev["r"]
            m = self.unit * STICKY
            if x0 - m <= px <= x1 + m and y0 - m <= py <= y1 + m:
                return prev
        best, bd = None, 1e9
        for k in self.keys + self.sug_keys:
            if k["id"].startswith("SUG") and not k["word"]:
                continue
            x0, y0, x1, y1 = k["r"]
            d = math.hypot(max(x0 - px, 0, px - x1), max(y0 - py, 0, py - y1))
            if d < bd:
                best, bd = k, d
        return best if bd <= self.unit * 0.5 else None

    def update_hand(self, h, lm, now):
        if lm is None:
            if h.lost_since is None:
                h.lost_since = now
            if now - h.lost_since > LOST_GRACE:
                h.reset()
            return
        h.lost_since = None
        h.lm = lm

        def p(i):
            return np.array([lm[i].x * 4 / 3, lm[i].y, lm[i].z * 4 / 3])

        palm = np.linalg.norm(p(0) - p(9)) + 1e-6
        raw = np.linalg.norm(p(4) - p(8)) / palm
        h.ratio = raw if h.ratio is None else h.ratio * 0.5 + raw * 0.5
        h.pinch_amt = float(np.clip((0.75 - h.ratio) / (0.75 - PINCH_ON), 0, 1))
        was = h.pinched
        if not was:
            h.on_frames = h.on_frames + 1 if h.ratio < PINCH_ON else 0
            if h.on_frames >= PINCH_FRAMES:
                h.pinched = True
        elif h.ratio > PINCH_OFF:
            h.pinched = False
            h.on_frames = 0

        # aim from the index + middle knuckles: they stay put while thumb and index pinch
        tx, ty = (lm[5].x + lm[9].x) / 2, (lm[5].y + lm[9].y) / 2
        ax0, ay0, ax1, ay1 = self.aim_for(h)
        zw, zh = self.zone_dims(h)
        if h.recenter:
            h.zone_c = (tx, ty)
            h.fx.reset(); h.fy.reset()
            h.recenter = False
        if not h.pinched or h.cursor is None:
            cx, cy = h.zone_c
            nx = (tx - cx) / zw + 0.5
            ny = (ty - cy) / zh + 0.5
            # pushing past an edge drags the box along, like a trackpad that never runs out
            if nx < 0: cx += nx * zw; nx = 0.0
            if nx > 1: cx += (nx - 1) * zw; nx = 1.0
            if ny < 0: cy += ny * zh; ny = 0.0
            if ny > 1: cy += (ny - 1) * zh; ny = 1.0
            h.zone_c = (cx, cy)
            h.cursor = (h.fx(ax0 + nx * (ax1 - ax0), now), h.fy(ay0 + ny * (ay1 - ay0), now))

        h.hover = self.hit(*h.cursor, prev=h.hover)
        h.hover_hist.append((now, h.hover))

        if self.hand_paused(now):
            h.held_key = None
            h.dwell_key, h.dwell_start = h.hover, now
            if h.pinched and not was:
                h.last_emit = now
            return

        if self.mode == "pinch":
            if h.pinched and not was and now - h.last_emit > REFRACTORY:
                target = h.hover
                for t, k in h.hover_hist:
                    if now - t <= LOOKBACK:
                        target = k
                        break
                if target:
                    self.emit(target)
                    h.last_emit = now
                    h.held_key, h.held_since, h.last_repeat = target, now, now
            elif h.pinched and h.held_key and h.held_key["id"] == "DEL":
                if now - h.held_since > REPEAT_DELAY and now - h.last_repeat > REPEAT_RATE:
                    self.emit(h.held_key)
                    h.last_repeat = now
            if not h.pinched:
                h.held_key = None
        else:
            if h.hover is not h.dwell_key:
                h.dwell_key, h.dwell_start = h.hover, now
            elif h.hover and now - h.dwell_start >= DWELL_TIME:
                self.emit(h.hover)
                h.dwell_start = now

    def update_gesture(self, lm, now):
        g = classify(lm) if lm is not None else None
        self.g_hist.append(g)
        stable, n = Counter(self.g_hist).most_common(1)[0]
        stable = stable if n >= GESTURE_STABLE[1] else None
        if not self.ai_mode and stable != "rock":
            stable = None  # while typing, only the AI toggle gesture counts
        if stable != self.g_cur:
            self.g_cur, self.g_start, self.g_fired = stable, now, False
        if stable and not self.g_fired and now - self.g_start >= self.hold_time(stable) \
                and now - self.g_last_fire > GESTURE_COOLDOWN:
            self.g_fired = True
            self.g_last_fire = now
            if stable == "rock":
                self.toggle_ai()
            elif stable == "fist":
                if self.ai_mode:
                    self.toggle_ai()
            elif stable == "pinky":
                self.close_active_app()
            else:
                self.run_gesture(stable)

    @staticmethod
    def hold_time(g):
        return GESTURE_HOLD_CLOSE if g == "pinky" else GESTURE_HOLD

    def gesture_progress(self, now):
        if not self.g_cur or self.g_fired:
            return 0.0
        return min((now - self.g_start) / self.hold_time(self.g_cur), 1.0)

    def close_active_app(self):
        fg = user32.GetForegroundWindow()
        cls = ctypes.create_unicode_buffer(64)
        if fg:
            user32.GetClassNameW(fg, cls, 64)
        if not fg or fg == self.hwnd or cls.value in ("Progman", "WorkerW", "Shell_TrayWnd"):
            self.say("NO APP TO CLOSE")
            return
        title = window_title(fg) or "APP"
        user32.PostMessageW(fg, 0x0010, 0, 0)  # WM_CLOSE: same as clicking the window's X
        self.say(f"CLOSING {title[:28].upper()}")

    def on_detection(self, hands, now):
        self.vis = [lm for lm, _ in hands]
        primary = max(hands, key=lambda hl: hl[0][0].x)[0] if hands else None  # right-most wrist
        self.update_gesture(primary, now)

        if self.ai_mode:
            for h in self.hands.values():
                h.reset()
            return

        if self.dual:
            left = right = None
            if len(hands) >= 2:
                a, b = sorted(hands[:2], key=lambda hl: hl[0][0].x)
                left, right = a[0], b[0]
            elif len(hands) == 1:
                lm, label = hands[0]
                if label == "Left" or (label not in ("Left", "Right") and lm[0].x < 0.5):
                    left = lm
                else:
                    right = lm
            self.update_hand(self.hands["L"], left, now)
            self.update_hand(self.hands["R"], right, now)
        else:
            self.update_hand(self.hands["S"], primary, now)

    def active_hands(self):
        return [self.hands["L"], self.hands["R"]] if self.dual else [self.hands["S"]]

    # ---------- drawing ----------
    def cam_map(self, x, y):
        cw = self.W - 2 * self.pad
        full_h = int(cw * 0.75)
        off = (full_h - self.cam_h) // 2
        return int(self.pad + x * cw), int(self.cam_y + y * full_h - off)

    def draw(self, frame, now, fps):
        W, pad = self.W, self.pad
        img = self.base.copy()
        glow = np.zeros_like(img)
        texts = []

        # header buttons
        states = {"close": False, "ai": self.ai_mode, "hands": self.dual}
        labels = {"close": "×", "ai": "AI", "hands": "2H" if self.dual else "1H"}
        for name, (x0, y0, x1, y1) in self.buttons.items():
            on = states[name]
            col = GOLD if on else (RED if name == "close" else CYAN)
            cv2.rectangle(img, (x0, y0), (x1, y1), PANEL, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 1, cv2.LINE_AA)
            if on:
                cv2.rectangle(glow, (x0, y0), (x1, y1), col, 2)
            font = self.f_sug if name == "close" else self.f_btn
            texts.append(((x0 + x1) / 2, (y0 + y1) / 2, labels[name], col, font, "mm"))

        # camera strip
        cw = W - 2 * pad
        full_h = int(cw * 0.75)
        cam_rect = (pad, self.cam_y, pad + cw, self.cam_y + self.cam_h)
        if frame is not None:
            cam = cv2.resize(frame, (cw, full_h), interpolation=cv2.INTER_AREA)
            off = (full_h - self.cam_h) // 2
            g = cv2.cvtColor(cam[off:off + self.cam_h], cv2.COLOR_BGR2GRAY)
            holo = cv2.merge([g, (g * 0.72 + 18).astype(np.uint8), (g * 0.32 + 8).astype(np.uint8)])
            holo[::3] = (holo[::3] * 0.7).astype(np.uint8)
            img[self.cam_y:self.cam_y + self.cam_h, pad:pad + cw] = (holo * 0.8).astype(np.uint8)
        sv = img[self.cam_y:self.cam_y + self.cam_h, pad:pad + cw]
        gv = glow[self.cam_y:self.cam_y + self.cam_h, pad:pad + cw]

        def local(x, y):
            px, py = self.cam_map(x, y)
            return px - pad, py - self.cam_y

        if not self.ai_mode:
            for h in self.active_hands():
                if h.lm is None:
                    continue
                zw, zh = self.zone_dims(h)
                cx, cy = h.zone_c
                z0, z1 = local(cx - zw / 2, cy - zh / 2), local(cx + zw / 2, cy + zh / 2)
                cv2.rectangle(sv, z0, z1, CYAN_DIM, 1, cv2.LINE_AA)
                L = 8
                for (px, py, dx, dy) in ((z0[0], z0[1], 1, 1), (z1[0], z0[1], -1, 1),
                                         (z0[0], z1[1], 1, -1), (z1[0], z1[1], -1, -1)):
                    cv2.line(sv, (px, py), (px + L * dx, py), CYAN, 2)
                    cv2.line(sv, (px, py), (px, py + L * dy), CYAN, 2)
                if self.dual:
                    texts.append((pad + z0[0] + 4, self.cam_y + z0[1] + 10, h.name, CYAN, self.f_small, "lm"))
        for lm in self.vis:
            pts = [local(q.x, q.y) for q in lm]
            for a, b in HAND_BONES:
                cv2.line(gv, pts[a], pts[b], CYAN, 3, cv2.LINE_AA)
                cv2.line(sv, pts[a], pts[b], CYAN, 1, cv2.LINE_AA)
            for i, q in enumerate(pts):
                cv2.circle(sv, q, 2, GOLD if i in (4, 8) else WHITE, -1, cv2.LINE_AA)
            cv2.line(sv, pts[4], pts[8], CYAN_DIM, 1, cv2.LINE_AA)
            if not self.ai_mode:
                km = ((pts[5][0] + pts[9][0]) // 2, (pts[5][1] + pts[9][1]) // 2)
                cv2.circle(gv, km, 6, GOLD, 2, cv2.LINE_AA)
                cv2.circle(sv, km, 3, GOLD, -1, cv2.LINE_AA)
        if not self.vis:
            mid = ((cam_rect[0] + cam_rect[2]) / 2, (cam_rect[1] + cam_rect[3]) / 2)
            if self.booting:
                dots = "." * (int(now * 3) % 4)
                texts.append((mid[0], mid[1], f"STARTING CAMERA{dots}", GOLD, self.f_sug, "mm"))
                a = (now * 300) % 360
                cv2.ellipse(img, (int(mid[0]), int(mid[1]) - 30), (12, 12), 0, a, a + 250, CYAN, 2, cv2.LINE_AA)
            else:
                texts.append((mid[0], mid[1], "RAISE YOUR HAND", RED, self.f_sug, "mm"))
        cv2.rectangle(img, cam_rect[:2], cam_rect[2:], CYAN_DIM, 1)

        n = len(self.vis)
        scol = CYAN if n else RED
        cv2.circle(img, (pad + 10, self.cam_y + 12), 4, scol, -1, cv2.LINE_AA)
        if n:
            cv2.circle(glow, (pad + 10, self.cam_y + 12), 6, CYAN, -1)
        texts.append((pad + 20, self.cam_y + 12, f"{n} HAND{'S' if n != 1 else ''}" if n else "NO HAND",
                      scol, self.f_small, "lm"))
        if self.ai_mode:
            state, scol2 = "AI MODE", GOLD
        elif self.hand_paused(now):
            state, scol2 = "KEYBOARD", WHITE
        else:
            state, scol2 = ("2-HAND " if self.dual else "") + self.mode.upper(), GOLD
        texts.append((W - pad - 6, self.cam_y + 12, state, scol2, self.f_small, "rm"))
        texts.append((pad + 6, self.cam_y + self.cam_h - 6, f"{fps:3.0f} FPS", CYAN, self.f_small, "ls"))
        texts.append((W - pad - 6, self.cam_y + self.cam_h - 6, f"SENS {self.sens + 1}/{len(SENS_LEVELS)}",
                      CYAN, self.f_small, "rs"))

        # gesture readout with hold progress
        prog = self.gesture_progress(now)
        if self.g_cur and (self.ai_mode or self.g_cur == "rock"):
            label = {g: lab for g, lab, _ in GESTURE_INFO}.get(self.g_cur, self.g_cur)
            if self.g_cur == "rock":
                act = "Exit AI mode" if self.ai_mode else "AI mode"
            elif self.g_cur in BUILTIN_ACTIONS:
                act = BUILTIN_ACTIONS[self.g_cur]
            else:
                act = self.gestures.get(self.g_cur, {}).get("name", "-")
            msg = f"{label.upper()}  >  {act}"
            cxm, cym = W // 2, self.cam_y + self.cam_h - 30
            tw = int(self.f_sug.getlength(msg)) + 30
            cv2.rectangle(img, (cxm - tw // 2, cym - 13), (cxm + tw // 2, cym + 13), BG, -1)
            cv2.rectangle(img, (cxm - tw // 2, cym - 13), (cxm + tw // 2, cym + 13), GOLD, 1)
            bw = int((tw - 2) * prog)
            cv2.rectangle(img, (cxm - tw // 2 + 1, cym + 10), (cxm - tw // 2 + 1 + bw, cym + 12), GOLD, -1)
            cv2.rectangle(glow, (cxm - tw // 2 + 1, cym + 10), (cxm - tw // 2 + 1 + bw, cym + 12), GOLD, -1)
            texts.append((cxm, cym - 1, msg, GOLD, self.f_sug, "mm"))

        # target window + echo
        if now - self.target_t > 0.4:
            fg = user32.GetForegroundWindow()
            self.target = "" if fg == self.hwnd else window_title(fg)
            self.target_t = now
        if self.target:
            texts.append((pad, self.tgt_y + 6, "> " + self.target[:52], GOLD, self.f_small, "lm"))
        else:
            texts.append((pad, self.tgt_y + 6, "> CLICK THE WINDOW YOU WANT TO TYPE INTO", RED, self.f_small, "lm"))
        shown = self.buffer
        while shown and self.f_text.getlength(shown + "_") > W - 2 * pad - 30:
            shown = shown[1:]
        caret = "▌" if int(now * 2) % 2 == 0 else " "
        texts.append((pad + 10, self.bar_y + self.bar_h / 2, shown + caret, WHITE, self.f_text, "lm"))

        if self.ai_mode:
            self.draw_ai_panel(img, glow, texts, prog)
        else:
            self.draw_keyboard(img, glow, texts, now)

        # bloom
        small = cv2.resize(glow, (W // 2, self.H // 2), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 5)
        img = cv2.add(img, cv2.resize(small, (W, self.H)))

        if self.toast and now - self.toast_t < 1.6:
            tw = int(self.f_sug.getlength(self.toast)) + 24
            x0, y0 = (W - tw) // 2, self.cam_y + 26
            cv2.rectangle(img, (x0, y0), (x0 + tw, y0 + 26), BG, -1)
            cv2.rectangle(img, (x0, y0), (x0 + tw, y0 + 26), GOLD, 1)
            texts.append((W / 2, y0 + 13, self.toast, GOLD, self.f_sug, "mm"))

        pil = Image.fromarray(img[..., ::-1].copy())
        d = ImageDraw.Draw(pil)
        for x, y, s, col, font, anchor in texts:
            d.text((x, y), s, font=font, fill=rgb(col), anchor=anchor)
        return np.array(pil)[..., ::-1].copy()

    def draw_keyboard(self, img, glow, texts, now):
        W = self.W
        hands = self.active_hands()
        for sk in self.sug_keys:
            x0, y0, x1, y1 = sk["r"]
            if not sk["word"]:
                cv2.rectangle(img, (x0, y0), (x1, y1), GRID, 1)
                continue
            hot = any(h.hover is sk for h in hands)
            cv2.rectangle(img, (x0, y0), (x1, y1), PANEL, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), CYAN if hot else CYAN_DIM, 2 if hot else 1, cv2.LINE_AA)
            if hot:
                cv2.rectangle(glow, (x0, y0), (x1, y1), CYAN, 3)
            texts.append(((x0 + x1) / 2, (y0 + y1) / 2, sk["word"], WHITE if hot else CYAN, self.f_sug, "mm"))

        layer = self.kb_layer()
        sl = slice(self.kb_y, self.kb_y + self.kb_h + 1)
        mask = layer[sl].any(axis=2)
        img[sl][mask] = layer[sl][mask]
        hovered = {id(h.hover) for h in hands if h.hover is not None}
        for k in self.keys + self.sug_keys:
            x0, y0, x1, y1 = k["r"]
            t = self.flash.get(k["id"])
            is_sug = k["id"].startswith("SUG")
            lab = k.get("word") if is_sug else key_label(k["id"], self.shift, self.mode)
            font = self.f_sug if is_sug else self.font_for(lab)
            if t and now - t < 0.15:
                cv2.rectangle(img, (x0, y0), (x1, y1), GOLD, -1)
                cv2.rectangle(glow, (x0, y0), (x1, y1), GOLD, -1)
                texts.append(((x0 + x1) / 2, (y0 + y1) / 2, lab, BG, font, "mm"))
            elif id(k) in hovered and not is_sug:
                ov = img[y0:y1, x0:x1]
                ov[:] = cv2.addWeighted(ov, 0.55, np.full_like(ov, CYAN), 0.45, 0)
                cv2.rectangle(img, (x0, y0), (x1, y1), WHITE, 1, cv2.LINE_AA)
                cv2.rectangle(glow, (x0, y0), (x1, y1), CYAN, 3)
                texts.append(((x0 + x1) / 2, (y0 + y1) / 2, lab, BG, font, "mm"))
                if len(k["id"]) == 1:  # magnified callout
                    bw, bh = int(self.unit * 1.25), int(self.key_h * 1.15)
                    bx0 = int(np.clip((x0 + x1) / 2 - bw / 2, 2, W - bw - 2))
                    by0 = y0 - bh - 6
                    cv2.rectangle(img, (bx0, by0), (bx0 + bw, by0 + bh), BG, -1)
                    cv2.rectangle(img, (bx0, by0), (bx0 + bw, by0 + bh), CYAN, 1, cv2.LINE_AA)
                    cv2.rectangle(glow, (bx0, by0), (bx0 + bw, by0 + bh), CYAN, 2)
                    texts.append((bx0 + bw / 2, by0 + bh / 2, lab, WHITE, self.f_pop, "mm"))

        self.ripples = [rp for rp in self.ripples if now - rp[2] < 0.35]
        for x, y, t in self.ripples:
            p = (now - t) / 0.35
            col = tuple(int(c * (1 - p)) for c in GOLD)
            cv2.circle(glow, (x, y), int(8 + p * self.unit * 1.2), col, 2, cv2.LINE_AA)

        for h in hands:
            if h.disp is None:
                continue
            cx, cy = int(h.disp[0]), int(h.disp[1])
            r = int(self.unit * 0.42)
            ang = (now * 260) % 360 * (-1 if h.name == "L" else 1)
            for i in range(3):
                a0 = ang + i * 120
                cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 60, CYAN, 2, cv2.LINE_AA)
                cv2.ellipse(glow, (cx, cy), (r, r), 0, a0, a0 + 60, CYAN, 3, cv2.LINE_AA)
            inner = max(3, int(r * 0.8 * (1 - 0.75 * h.pinch_amt)))
            cv2.circle(img, (cx, cy), inner, GOLD, 2 if h.pinched else 1, cv2.LINE_AA)
            cv2.circle(glow, (cx, cy), inner, GOLD, 2)
            cv2.circle(img, (cx, cy), 2, GOLD, -1, cv2.LINE_AA)
            if self.dual:
                texts.append((cx + r + 4, cy - r, h.name, GOLD, self.f_small, "lm"))
            if self.mode == "dwell" and h.hover is not None and h.dwell_key is h.hover:
                prog = min((now - h.dwell_start) / DWELL_TIME, 1.0)
                cv2.ellipse(img, (cx, cy), (r + 6, r + 6), -90, 0, 360 * prog, GOLD, 2, cv2.LINE_AA)
                cv2.ellipse(glow, (cx, cy), (r + 6, r + 6), -90, 0, 360 * prog, GOLD, 4, cv2.LINE_AA)

    def draw_ai_panel(self, img, glow, texts, prog):
        W, pad, gap = self.W, self.pad, self.gap
        cols, rows = 2, 6
        panel_top = self.sug_y
        cw = (W - 2 * pad - gap) / cols
        ch = (self.kb_y + self.kb_h - panel_top - (rows - 1) * gap) / rows
        # last slot: how-to hint
        hx0 = int(pad + (cw + gap)); hy0 = int(panel_top + (rows - 1) * (ch + gap))
        cv2.rectangle(img, (hx0, hy0), (int(hx0 + cw), int(hy0 + ch)), GOLD, 1, cv2.LINE_AA)
        texts.append((hx0 + cw / 2, hy0 + ch / 2 - 7, "HOLD STEADY", GOLD, self.f_small, "mm"))
        texts.append((hx0 + cw / 2, hy0 + ch / 2 + 8, "until the bar fills", WHITE, self.f_small, "mm"))
        for i, (gid, label, pat) in enumerate(GESTURE_INFO):
            c, r = i % cols, i // cols
            x0 = int(pad + c * (cw + gap)); y0 = int(panel_top + r * (ch + gap))
            x1 = int(x0 + cw); y1 = int(y0 + ch)
            active = self.g_cur == gid
            cv2.rectangle(img, (x0, y0), (x1, y1), PANEL, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), CYAN if active else CYAN_DIM, 2 if active else 1, cv2.LINE_AA)
            if active:
                cv2.rectangle(glow, (x0, y0), (x1, y1), CYAN, 3)
                bw = int((x1 - x0 - 2) * prog)
                cv2.rectangle(img, (x0 + 1, y1 - 4), (x0 + 1 + bw, y1 - 1), GOLD, -1)
                cv2.rectangle(glow, (x0 + 1, y1 - 4), (x0 + 1 + bw, y1 - 1), GOLD, -1)
            # finger diagram: thumb, index, middle, ring, pinky
            heights = (0.45, 0.8, 0.9, 0.8, 0.6)
            fx, base, full = x0 + 10, y1 - 8, (y1 - y0) - 16
            for j, (upf, hf) in enumerate(zip(pat, heights)):
                top = int(base - full * (hf if upf else 0.22))
                col = GOLD if (upf and active) else (CYAN if upf else CYAN_DIM)
                cv2.rectangle(img, (fx + j * 7, top), (fx + j * 7 + 4, base), col, -1)
            act = BUILTIN_ACTIONS.get(gid) or self.gestures.get(gid, {}).get("name", "-")
            texts.append((fx + 42, (y0 + y1) / 2 - 7, label, CYAN if active else CYAN_DIM, self.f_small, "lm"))
            texts.append((fx + 42, (y0 + y1) / 2 + 8, act, WHITE if active else CYAN, self.f_sug, "lm"))

    # ---------- main loop ----------
    def run(self):
        tracker = Tracker()
        tracker.start()
        print("AirTyper running. Click the window you want to type into. Ctrl+Alt+Q to quit.", flush=True)
        last_seq, first = -1, True
        last = time.perf_counter()
        try:
            while self.running:
                if tracker.error:
                    raise SystemExit(tracker.error)
                now = time.perf_counter()
                dt = now - last
                last = now
                seq, frame, hands, fps = tracker.latest()
                self.booting = frame is None

                if self.toggle_hide:
                    self.toggle_hide = False
                    self.hidden = not self.hidden
                    if self.hwnd:
                        user32.ShowWindow(self.hwnd, 0 if self.hidden else 4)
                if self.hidden:
                    for h in self.hands.values():
                        h.reset()
                    cv2.waitKey(50)
                    continue

                if self.sug_dirty:
                    self.sug_dirty = False
                    self.update_suggestions()
                if seq != last_seq:
                    last_seq = seq
                    self.on_detection(hands, now)

                # ease the drawn cursor toward the filtered target every render frame
                k = 1 - math.exp(-dt * EASE)
                for h in self.hands.values():
                    if h.cursor is None:
                        h.disp = None
                    elif h.disp is None:
                        h.disp = h.cursor
                    else:
                        h.disp = (h.disp[0] + (h.cursor[0] - h.disp[0]) * k,
                                  h.disp[1] + (h.cursor[1] - h.disp[1]) * k)

                out = self.draw(frame, now, fps)
                if first:
                    self.setup_window(out)
                    first = False
                cv2.imshow(WIN, out)
                cv2.waitKey(1)
                spare = 1 / RENDER_FPS - (time.perf_counter() - now)
                if spare > 0:
                    time.sleep(spare)
        finally:
            self.save_settings()
            tracker.alive = False
            tracker.join(timeout=2)
            self.hotkeys.stop()
            self.real_keys.stop()
            cv2.destroyAllWindows()


HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]

def message_box(text, error=False):
    user32.MessageBoxW(None, text, "AirTyper", 0x10 if error else 0x40)


def main():
    import sys
    import traceback
    log_path = os.path.join(HERE, "airtyper.log")
    if sys.stdout is None or sys.stderr is None:  # launched with pythonw (no console)
        log = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    _mutex = kernel32.CreateMutexW(None, False, "Local\\AirTyperSingleInstance")  # noqa: F841 (held for app lifetime)
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        message_box("AirTyper is already running.\n\nLook for it on the screen, or press Ctrl+Alt+H to show it.")
        return

    try:
        AirTyper().run()
    except SystemExit as e:
        if isinstance(e.code, str):
            message_box(e.code, error=True)
    except Exception:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(time.strftime("\n[%Y-%m-%d %H:%M:%S] ") + traceback.format_exc())
        traceback.print_exc()
        message_box(f"AirTyper stopped because of an error.\n\nDetails were saved to:\n{log_path}", error=True)


if __name__ == "__main__":
    main()
