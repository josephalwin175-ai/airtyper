"""AirTyper - type in the air with your webcam, and control your PC with hand gestures.

TYPING
  Raise your hand: the cursor starts at the centre of the keyboard. Small hand movements steer it.
  Pinch thumb + index to press. Keystrokes go to whatever window you last clicked.
  2-hand mode: left hand covers the left half of the keyboard, right hand the right half.
  Hybrid: your physical keyboard keeps working; hand typing pauses briefly after real keystrokes.

GESTURE MODE (rock, Ctrl+Alt+A or the Gestures button)  - edit gestures.json to remap
  Swipe open hand left/right -> next/previous tab     Swipe up -> Task view
  Two-finger swipe -> switch app                      Push toward the camera -> play/pause
  Pinch (OK sign) + move up/down -> scroll            Pinch + twist -> volume
  Pinch with both hands, pull apart/together -> zoom
  Hold one finger -> air pointer: pinch = click, pinch twice/thrice = select word/line, pinch + drag = select
  Catch (open hand -> fist, fast) -> copy and hold. Move the fist to carry it (the cursor follows),
  flick it open to drop it there (click + paste). Open slowly -> cancel.
  Two-finger swipe down -> clipboard shelf: your last catches; hand height picks one, catch takes it
  Hold other shapes -> launchers, pinky = close app

LINK (two laptops on the same Wi-Fi, both running AirTyper)
  Two-finger swipe up (or Ctrl+Alt+L) -> Link panel: pair with push + matching codes (thumbs up)
  Carry an item to the screen edge where your friend sits and flick it open -> it flies to their laptop
  When something arrives, catch -> it's in your fist; drop it anywhere. Their hands show as ghosts.

WINDOW
  Drag the header to move it (position is remembered). Buttons: ?, 1H/2H, Gestures, close.

HOTKEYS (Ctrl+Alt + ...)
  Up/Down sensitivity  R re-centre  M pinch/dwell  D one/two hands  A gestures  V shelf  G guide  H hide
  L Link panel  K Link on/off
  F9 record hand data (for replay.py)  Q quit
"""
import ctypes
import io
import json
import math
import os
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from ctypes import wintypes

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
from pynput.keyboard import Controller, GlobalHotKeys, Key, Listener
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from gesture_engine import Engine, OneEuro

try:
    from link import SIDES, Link, local_ipv4s
except ImportError:  # cryptography missing: AirTyper still runs, without Link
    Link, SIDES, local_ipv4s = None, ("right", "left", "top"), lambda: []

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "hand_landmarker.task")
SETTINGS = os.path.join(HERE, "settings.json")
GESTURES = os.path.join(HERE, "gestures.json")
RECORDINGS = os.path.join(HERE, "recordings")
WIN = "AirTyper"
CHIP = "AirTyper carry"  # small floating label that follows the cursor while you carry an item
PORTAL = "AirTyper portal"  # glowing drop zones at the screen edges where linked friends sit
APPDATA_DIR = os.path.join(os.environ.get("APPDATA", HERE), "AirTyper")
LINK_IDENTITY = os.path.join(APPDATA_DIR, "link.json")
RECEIVED = os.path.join(os.environ.get("LOCALAPPDATA", HERE), "AirTyper", "Received")

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
POINTER_SPAN = 0.38               # camera width (square units) that spans the whole screen with the air pointer
POINTER_EASE = 30.0               # the OS cursor glides toward the hand at this rate (120 Hz), not in 30 fps steps
SCROLL_GAIN = 720                 # wheel units (120 = one notch) per palm-size of pinch-scroll travel
SCROLL_EASE = 22.0                # pending scroll is spread smoothly over the next frames at this rate
SCROLL_GLIDE = 0.32               # s; after a fast pinch-scroll the page keeps gliding and slows down
INPUT_HZ = 120                    # cursor and scroll updates per second
COPY_TIMEOUT = 0.6                # s to wait for the clipboard to change after a catch
WORD_GRAB_WAIT = 0.35             # s; with the air pointer on and nothing copied, grab the word under it
DRAG_PRECISION = 0.6              # pointer speed while the mouse button is held (selecting text)
CARRY_START = 0.05                # fist travel (square camera units) before a held item starts carrying the cursor
SHELF_SIZE = 8                    # clipboard shelf: last catches kept (memory only, cleared on quit)
EASE = 24.0                       # display cursor easing rate (higher = snappier)
PORTAL_ZONE = 70                  # px from a screen edge where a carried item drops into a friend's portal
INCOMING_FRESH = 25.0             # s an arrived item waits on the gesture card for a catch
GHOST_HZ = 15                     # hand skeletons sent to linked friends per second
RISKY_EXT = {".exe", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".msi", ".scr", ".com", ".lnk", ".jar", ".hta", ".reg"}
RENDER_FPS = 60

# ---------- palette (BGR) ----------
BG = (22, 19, 17)
SURF = (36, 32, 29)
SURF2 = (50, 45, 41)
LINE = (66, 60, 55)
TEXT = (245, 242, 238)
MUTED = (158, 148, 138)
ACCENT = (255, 170, 80)
ACCENT_DIM = (150, 110, 60)
AMBER = (64, 190, 255)
RED = (96, 96, 240)
GHOST = (230, 130, 205)  # a linked friend's hands


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def rrect(img, x0, y0, x1, y1, r, col, th=-1):
    """Rounded rectangle; th=-1 fills."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if th < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), col, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), col, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)
        return
    for p, q in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
                 ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
        cv2.line(img, p, q, col, th, cv2.LINE_AA)
    for (cx, cy), a in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, a, a + 90, col, th, cv2.LINE_AA)


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
kernel32 = ctypes.windll.kernel32
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
user32.SendMessageW.argtypes = [wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.FindWindowW.restype = wintypes.HWND
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.OpenClipboard.argtypes = [wintypes.HWND]
user32.GetClipboardData.argtypes = [wintypes.UINT]
user32.GetClipboardData.restype = wintypes.HANDLE
user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
user32.SetClipboardData.restype = wintypes.HANDLE
user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
CF_UNICODETEXT = 13
CF_DIB, CF_HDROP = 8, 15
shell32 = ctypes.windll.shell32
shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
shell32.DragQueryFileW.restype = wintypes.UINT
kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalSize.restype = ctypes.c_size_t
user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
user32.RegisterClipboardFormatW.restype = wintypes.UINT
# password managers mark what they copy with these; such items never leave this laptop
CF_EXCLUDE = user32.RegisterClipboardFormatW("ExcludeClipboardContentFromMonitorProcessing")
CF_HISTORY = user32.RegisterClipboardFormatW("CanIncludeInClipboardHistory")


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


def window_class(hwnd):
    buf = ctypes.create_unicode_buffer(64)
    user32.GetClassNameW(hwnd, buf, 64)
    return buf.value


def process_name(hwnd):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        n = wintypes.DWORD(260)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
            return os.path.basename(buf.value)
        return ""
    finally:
        kernel32.CloseHandle(h)


def open_clipboard():
    for _ in range(10):  # the app that just copied may still hold it open
        if user32.OpenClipboard(None):
            return True
        time.sleep(0.01)
    return False


def clip_get_text():
    if not open_clipboard():
        return None
    try:
        h = user32.GetClipboardData(CF_UNICODETEXT)
        if not h:
            return None
        p = kernel32.GlobalLock(h)
        if not p:
            return None
        try:
            return ctypes.wstring_at(p)
        finally:
            kernel32.GlobalUnlock(h)
    finally:
        user32.CloseClipboard()


def clip_put(fmt, data):
    h = kernel32.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
    if not h:
        return
    p = kernel32.GlobalLock(h)
    ctypes.memmove(p, data, len(data))
    kernel32.GlobalUnlock(h)
    if not open_clipboard():
        kernel32.GlobalFree(h)
        return
    try:
        user32.EmptyClipboard()
        if not user32.SetClipboardData(fmt, h):
            kernel32.GlobalFree(h)
    finally:
        user32.CloseClipboard()


def clip_set_text(s):
    clip_put(CF_UNICODETEXT, (s + "\0").encode("utf-16-le"))


def _global_bytes(h):
    p = kernel32.GlobalLock(h)
    if not p:
        return None
    try:
        return ctypes.string_at(p, kernel32.GlobalSize(h))
    finally:
        kernel32.GlobalUnlock(h)


def dib_to_png(dib):
    bi_size, = struct.unpack_from("<I", dib, 0)
    bits, = struct.unpack_from("<H", dib, 14)
    comp, clr_used = struct.unpack_from("<I", dib, 16)[0], struct.unpack_from("<I", dib, 32)[0]
    colors = clr_used or (1 << bits if bits <= 8 else 0)
    masks = 12 if comp == 3 and bi_size == 40 else 0
    bmp = b"BM" + struct.pack("<IHHI", 14 + len(dib), 0, 0, 14 + bi_size + masks + colors * 4) + dib
    im = Image.open(io.BytesIO(bmp)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "PNG", compress_level=3)
    return buf.getvalue(), im.size


def png_to_dib(png):
    buf = io.BytesIO()
    Image.open(io.BytesIO(png)).convert("RGB").save(buf, "BMP")
    return buf.getvalue()[14:]


def item_label(item):
    k = item["kind"]
    if item.get("private"):
        return "Private item (password manager)"
    if k == "text":
        t = " ".join(item["text"][:2000].split())[:300]
        if item.get("url"):
            return "Link · " + t.split("://", 1)[-1]
        return t
    if k == "image":
        w, h = item.get("size") or Image.open(io.BytesIO(item["png"])).size
        return f"Image {w}×{h}"
    names = [os.path.basename(os.path.normpath(p)) for p in item["paths"]]
    return names[0] if len(names) == 1 else f"{len(names)} items · {', '.join(names[:2])}…"


def finish_item(item):
    """Fill in label / url / risky flags for a clipboard or received item."""
    if item["kind"] == "text":
        t = item["text"].strip()
        item["url"] = t.lower().startswith(("http://", "https://")) and not any(c.isspace() for c in t)
    if item["kind"] == "files":
        risky = False
        for p in item["paths"]:
            for root, _, files in (os.walk(p) if os.path.isdir(p) else [("", None, [p])]):
                risky = risky or any(os.path.splitext(f)[1].lower() in RISKY_EXT for f in files)
        item["risky"] = risky
    item["label"] = item.get("label") or item_label(item)
    return item


def clip_read_item():
    """What the clipboard holds right now: files, text or an image (None if nothing usable)."""
    if not open_clipboard():
        return None
    dib, item = None, None
    try:
        private = bool(CF_EXCLUDE and user32.IsClipboardFormatAvailable(CF_EXCLUDE))
        if not private and CF_HISTORY and user32.IsClipboardFormatAvailable(CF_HISTORY):
            b = _global_bytes(user32.GetClipboardData(CF_HISTORY) or 0) or b""
            private = len(b) >= 4 and struct.unpack_from("<I", b)[0] == 0
        if user32.IsClipboardFormatAvailable(CF_HDROP):
            h = user32.GetClipboardData(CF_HDROP)
            paths = []
            for i in range(shell32.DragQueryFileW(h, 0xFFFFFFFF, None, 0) if h else 0):
                n = shell32.DragQueryFileW(h, i, None, 0)
                buf = ctypes.create_unicode_buffer(n + 1)
                shell32.DragQueryFileW(h, i, buf, n + 1)
                paths.append(buf.value)
            if paths:
                item = {"kind": "files", "paths": paths}
        if item is None and user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            h = user32.GetClipboardData(CF_UNICODETEXT)
            p = kernel32.GlobalLock(h) if h else None
            if p:
                try:
                    item = {"kind": "text", "text": ctypes.wstring_at(p)}
                finally:
                    kernel32.GlobalUnlock(h)
        if item is None and user32.IsClipboardFormatAvailable(CF_DIB):
            h = user32.GetClipboardData(CF_DIB)
            dib = _global_bytes(h) if h else None
    finally:
        user32.CloseClipboard()
    if dib:
        try:
            png, size = dib_to_png(dib)
            item = {"kind": "image", "png": png, "size": size}
        except Exception:
            item = None
    if item is None:
        return None
    item["private"] = private
    return finish_item(item)


def clip_set_item(item):
    k = item["kind"]
    if k == "text":
        clip_set_text(item["text"])
    elif k == "image":
        clip_put(CF_DIB, png_to_dib(item["png"]))
    else:  # DROPFILES header + double-null-terminated wide paths: Explorer pastes the real files
        clip_put(CF_HDROP, struct.pack("<IiiII", 20, 0, 0, 0, 1) + ("\0".join(item["paths"]) + "\0\0").encode("utf-16-le"))


TERMINAL_CLASSES = {"ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS", "mintty", "VirtualConsoleClass"}
TERMINAL_APPS = {"windowsterminal.exe", "wezterm-gui.exe", "alacritty.exe", "mintty.exe"}


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
NAMED = {"DEL": "⌫", "ENTER": "enter", "SPACE": "space", "SHIFT": "⇧", "MODE": "MODE"}


def key_label(kid, shift, mode):
    if len(kid) == 1:
        if shift and kid in SHIFTED:
            return SHIFTED[kid]
        return kid.upper() if (shift or kid.isalpha()) else kid
    if kid == "MODE":
        return "dwell" if mode == "dwell" else "pinch"
    return NAMED[kid]


# ---------- gesture config ----------
POSE_LABELS = {
    "one": "One finger", "two": "Two fingers", "three": "Three fingers", "four": "Four fingers",
    "thumbs_up": "Thumbs up", "thumbs_down": "Thumbs down", "call_me": "Call me", "l_shape": "L shape",
    "pinky": "Pinky", "rock": "Rock",
}
POSE_ORDER = ["two", "three", "four", "thumbs_up", "thumbs_down", "call_me", "l_shape"]
BUILTIN_POSES = {"one": "Air pointer", "pinky": "Close active app"}
DEFAULT_GESTURES = {
    "poses": {
        "two": {"name": "Chrome", "run": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"},
        "three": {"name": "Claude", "run": "shell:AppsFolder\\Claude_pzs8sxrjxfjjc!Claude"},
        "four": {"name": "Antigravity", "run": "%LOCALAPPDATA%\\Programs\\Antigravity\\Antigravity.exe"},
        "thumbs_up": {"name": "File Explorer", "run": "explorer.exe"},
        "thumbs_down": {"name": "Undo", "keys": "ctrl+z"},
        "call_me": {"name": "Spotify", "run": "%APPDATA%\\Spotify\\Spotify.exe"},
        "l_shape": {"name": "Screenshot", "keys": "win+shift+s"},
    },
    "motions": {
        "swipe_left": {"name": "Next tab", "keys": "ctrl+tab"},
        "swipe_right": {"name": "Previous tab", "keys": "ctrl+shift+tab"},
        "swipe_up": {"name": "Task view", "keys": "win+tab"},
        "two_swipe_left": {"name": "Switch app", "keys": "alt+tab"},
        "two_swipe_right": {"name": "Switch app back", "keys": "alt+shift+tab"},
        "two_swipe_down": {"name": "Clipboard shelf", "action": "shelf"},
        "two_swipe_up": {"name": "Link", "action": "link"},
        "push": {"name": "Play / pause", "keys": "media_play_pause"},
        "zoom_in": {"name": "Zoom in", "keys": "ctrl+="},
        "zoom_out": {"name": "Zoom out", "keys": "ctrl+-"},
    },
    "per_app": {
        "Code.exe": {
            "swipe_left": {"name": "Next tab", "keys": "ctrl+page_down"},
            "swipe_right": {"name": "Previous tab", "keys": "ctrl+page_up"},
        },
        "Antigravity.exe": {
            "swipe_left": {"name": "Next tab", "keys": "ctrl+page_down"},
            "swipe_right": {"name": "Previous tab", "keys": "ctrl+page_up"},
        },
    },
}


def load_gesture_config():
    raw = load_json(GESTURES, None)
    if not isinstance(raw, dict):
        save_json(GESTURES, DEFAULT_GESTURES)
        raw = DEFAULT_GESTURES
    elif "poses" not in raw and "motions" not in raw:  # old flat format: pose -> app
        old = {k: v for k, v in raw.items() if k not in ("open_palm", "one")}  # those now swipe / point
        raw = {"poses": old, "motions": DEFAULT_GESTURES["motions"], "per_app": DEFAULT_GESTURES["per_app"]}
        save_json(GESTURES, raw)
    return {
        "poses": raw.get("poses", DEFAULT_GESTURES["poses"]),
        "motions": raw.get("motions", DEFAULT_GESTURES["motions"]),
        "per_app": {k.lower(): v for k, v in raw.get("per_app", {}).items()},
    }


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
        self.new = threading.Event()

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
                self.new.set()
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
        self.gcfg = load_gesture_config()

        l, t, r, b = work_area()
        self.W = W = int(min(max((r - l) * 0.21, 380), 470))
        self.pad = pad = 12
        self.gap = gap = 6

        ui, semi = ["SegUIVar.ttf", "segoeui.ttf"], ["seguisb.ttf", "SegUIVar.ttf", "segoeui.ttf"]
        self.f_title = load_font(semi, 17)
        self.f_key = load_font(ui, int(W / 22))
        self.f_small = load_font(ui, 12)
        self.f_btn = load_font(semi, 12)
        self.f_text = load_font(["consola.ttf", "segoeui.ttf"], 18)
        self.f_sug = load_font(ui, 14)
        self.f_body = load_font(semi, 15)
        self.f_pop = load_font(semi, int(W / 12))
        self.f_sym = load_font(["seguisym.ttf", "segoeui.ttf"], int(W / 20))
        self.f_arrow = load_font(["seguisym.ttf", "segoeui.ttf"], 56)

        # typing layout
        self.header_h = 44
        self.cam_y = self.header_h
        self.cam_h_type = int(W * 0.34)
        self.tgt_y = self.cam_y + self.cam_h_type + 8
        self.bar_y = self.tgt_y + 18
        self.bar_h = 38
        self.sug_y = self.bar_y + self.bar_h + 8
        self.sug_h = 30
        self.kb_y = self.sug_y + self.sug_h + gap
        self.unit = (W - 2 * pad - 9 * gap) / 10
        self.key_h = int(self.unit * 0.92)
        self.kb_h = 5 * self.key_h + 4 * gap
        h_type = self.kb_y + self.kb_h + pad
        # gesture layout: bigger camera, an action card and a clipboard card
        self.cam_h_ctrl = int(W * 0.5)
        self.card_y = self.cam_y + self.cam_h_ctrl + 10
        self.card_h = 58
        self.clip_y = self.card_y + self.card_h + 8
        self.clip_h = 58
        h_ctrl = self.clip_y + self.clip_h + 30
        # clipboard shelf layout
        self.shelf_y = self.cam_y + self.cam_h_type + 10
        self.shelf_row = 36
        h_shelf = self.shelf_y + 30 + SHELF_SIZE * self.shelf_row + 34
        # Link panel
        self.link_row = 58
        h_link = self.shelf_y + 30 + 4 * self.link_row + 40
        # guide: two pages, rows shrink a little on short screens
        self.help_pages = self.build_help()
        self.help_page = 0
        need = []
        for rows in self.help_pages:
            n_sec = sum(1 for _, a in rows if a is None)
            n_row = len(rows) - n_sec
            need.append((n_sec, n_row))
        self.help_row_h = int(min(np.clip((b - t - 40 - self.header_h - 58 - ns * 28) / nr, 17, 21) for ns, nr in need))
        h_help = max(self.header_h + 6 + ns * 28 + nr * self.help_row_h + 52 for ns, nr in need)
        self.heights = {"type": h_type, "control": h_ctrl, "shelf": h_shelf, "link": h_link,
                        "help": max(h_type, h_help)}
        self.layout = "type"
        self.H = h_type

        lb = self.settings.get("pos_lb")  # left + bottom: the window grows and shrinks upward
        if lb and l - W // 2 < lb[0] < r - W // 2 and t + 100 < lb[1] <= b + 10:
            self.pos = (lb[0], max(t, lb[1] - h_type))
        else:
            self.pos = (r - W - 12, b - h_type - 12)

        self.buttons_by = {lay: self.make_buttons(lay) for lay in self.heights}

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
        self.base_cache = {}
        self.mask_cache = {}

        self.out = Controller()
        self.mouse = MouseController()
        self.mode = self.settings.get("mode", "pinch")
        self.dual = bool(self.settings.get("dual", False))
        self.sens = int(np.clip(self.settings.get("sens", SENS_DEFAULT), 0, len(SENS_LEVELS) - 1))
        self.hold_only = bool(self.settings.get("hold_only_clipboard", False))
        self.natural_scroll = bool(self.settings.get("natural_scroll", True))
        self.ai_mode = False
        self.show_help = False
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
        self.toast, self.toast_t, self.toast_dur = "", 0.0, 1.6
        self.last_real_key = 0.0
        self.sug_dirty = False
        self.booting = True

        # gestures
        self.engine = Engine()
        self.engine_reset = False
        self.last_action = ("", "", 0.0)
        self.held_text = None
        self.pending_copy = None  # (time, clipboard sequence number before the copy)
        self.clip_prev = None
        self.ptr = None           # pointer position (float screen px) while the air pointer or a carry moves it
        self.pointing = False     # air pointer mode is on
        self.mouse_is_down = False
        self.carrying = False     # a held item is moving the cursor with the fist
        self.carry_off = [0.0, 0.0]
        self.shelf = deque(maxlen=SHELF_SIZE)
        self.show_shelf = False
        self.shelf_sel = 0
        self.chip_hwnd, self.chip_shown, self.chip_key, self.chip_img = None, False, None, None
        self.ptr_target, self.ptr_set = None, None  # where the hand wants the cursor / where we last put it
        self.scroll_pending = self.scroll_frac = 0.0  # wheel units still to send / sub-unit remainder
        self.scroll_vel = self.scroll_glide = 0.0     # measured units/s while pinching / gliding speed after
        self.scrolling = False
        self.sprites = {}
        self.held_item = None         # what a catch holds: {"kind": text|image|files, "label", ...}
        # Link
        self.link = None
        self.link_enabled = bool(self.settings.get("link_enabled", True))
        self.link_name = str(self.settings.get("link_name") or socket.gethostname())[:40]
        self.link_toggle = False
        self.link_view, self.link_view_t, self.link_conn = [], 0.0, []
        self.link_ips, self.link_ips_t = "", 0.0
        self.show_link, self.link_sel = False, 0
        self.link_hits = []           # clickable rects drawn this frame: (x0, y0, x1, y1, fn)
        self.incoming = None          # (item, time) just arrived from a friend
        self.transfers = {}           # id -> {"dir", "name", "frac", "done", "t"}
        self.ghosts = {}              # peer id -> (hands, holding, time)
        self.ghost_t = 0.0
        self.portals = {}             # side -> {"hwnd", "key", "img", "shown"}
        self.proc = (None, "")    # (hwnd, process name) cache
        self.fx_swipe = self.fx_splash = self.fx_catch = None
        self.rec, self.rec_toggle = None, False

        self.update_suggestions()

        self.hotkeys = GlobalHotKeys({
            "<ctrl>+<alt>+q": self.quit,
            "<ctrl>+<alt>+m": self.switch_mode,
            "<ctrl>+<alt>+h": self.request_hide,
            "<ctrl>+<alt>+r": self.request_recenter,
            "<ctrl>+<alt>+d": self.toggle_dual,
            "<ctrl>+<alt>+a": self.toggle_ai,
            "<ctrl>+<alt>+g": self.toggle_help,
            "<ctrl>+<alt>+v": self.toggle_shelf,
            "<ctrl>+<alt>+l": self.toggle_link_panel,
            "<ctrl>+<alt>+k": self.request_link_toggle,
            "<ctrl>+<alt>+<f9>": self.request_rec,
            "<ctrl>+<alt>+<up>": lambda: self.change_sens(1),
            "<ctrl>+<alt>+<down>": lambda: self.change_sens(-1),
        })
        self.hotkeys.start()
        self.real_keys = Listener(on_press=self.on_real_key)
        self.real_keys.start()

    def make_buttons(self, layout):
        gesture = ("close", "gest", "clips", "link", "help")
        names = {"type": ("close", "gest", "hands", "help"), "control": gesture, "shelf": gesture, "link": gesture,
                 "help": ("close", "help")}[layout]
        widths = {"close": 28, "gest": 74, "hands": 38, "clips": 46, "link": 42, "help": 28}
        bx, out = self.W - self.pad, {}
        for n in names:
            out[n] = (bx - widths[n], 9, bx, 35)
            bx -= widths[n] + 6
        return out

    def build_help(self):
        m, p = self.gcfg["motions"], self.gcfg["poses"]

        def nm(d, k):
            return d.get(k, {}).get("name", "-")
        rows = [
            ("Motions  (open hand)", None),
            ("Swipe left / right", f"{nm(m, 'swipe_left')} / {nm(m, 'swipe_right')}"),
            ("Swipe up / down", f"{nm(m, 'swipe_up')} / {nm(m, 'swipe_down')}"),
            ("Two-finger swipe left / right", f"{nm(m, 'two_swipe_left')} / {nm(m, 'two_swipe_right')}"),
            ("Push toward the camera", nm(m, "push")),
            ("Pinch + move up / down", "Scroll (fling to glide)"),
            ("Pinch + twist like a knob", "Volume"),
            ("Pinch both hands, pull apart", "Zoom in / out"),
            ("Select text  (hold one finger = pointer)", None),
            ("Pinch", "Click, place the caret"),
            ("Pinch twice / three times", "Select word / line"),
            ("Pinch, hold and drag", "Select a range"),
            ("Point + catch with the other hand", "Grab the word pointed at"),
            ("Copy, carry, drop", None),
            ("Catch: open hand → fist, fast", "Copy and hold"),
            ("Move the fist", "Carry it, cursor follows"),
            ("Flick the fist open", "Drop there: click + paste"),
            ("Flick open without moving", "Paste at the caret"),
            ("Open the hand slowly", "Cancel"),
            ("Two-finger swipe down", nm(m, "two_swipe_down") + "  (Ctrl+Alt+V)"),
        ]
        page2 = [
            ("Link  (two laptops, same Wi-Fi)", None),
            ("Two-finger swipe up", "Link panel  (Ctrl+Alt+L)"),
            ("In the panel: push / thumbs up", "Pair / codes match"),
            ("Carry to a friend's edge, flick open", "Throw it to them"),
            ("Catch when something arrives", "Take it, then drop it"),
            ("Hold still", None),
            ("One finger", "Air pointer"),
        ]
        page2 += [(POSE_LABELS[g], p[g].get("name", g)) for g in POSE_ORDER if g in p]
        page2 += [("Pinky", "Close active app"), ("Rock (index + pinky)", "Gestures ↔ typing")]
        return [rows, page2]

    # ---------- controls ----------
    def quit(self):
        self.running = False

    def switch_mode(self):
        self.mode = "dwell" if self.mode == "pinch" else "pinch"
        self.say(f"{self.mode.capitalize()} mode")

    def request_hide(self):
        self.toggle_hide = True

    def request_recenter(self):
        for h in self.hands.values():
            h.recenter = True
        self.say("Re-centred")

    def request_rec(self):
        self.rec_toggle = True

    def toggle_dual(self):
        self.dual = not self.dual
        for h in self.hands.values():
            h.reset()
        self.say("Two-hand typing" if self.dual else "One-hand typing")

    def toggle_ai(self):
        self.ai_mode = not self.ai_mode
        for h in self.hands.values():
            h.reset()
        self.engine_reset = True  # applied on the detection thread's next frame, in the main loop
        self.held_text, self.pending_copy = None, None
        self.carrying, self.pointing = False, False
        self.held_item = None
        if not self.ai_mode:
            self.show_shelf = self.show_link = False
        self.say("Gesture mode" if self.ai_mode else "Typing mode")

    def toggle_help(self):
        """? / Ctrl+Alt+G: page 1, page 2, closed."""
        if not self.show_help:
            self.show_help, self.help_page = True, 0
        elif self.help_page == 0:
            self.help_page = 1
        else:
            self.show_help = False

    def toggle_link_panel(self):
        self.show_link = not self.show_link
        self.show_shelf = False
        self.link_sel = 0
        self.link_view_t = 0.0
        if self.show_link and not self.ai_mode:
            self.toggle_ai()

    def request_link_toggle(self):
        self.link_toggle = True

    def toggle_shelf(self):
        self.show_shelf = not self.show_shelf
        self.show_link = False
        self.shelf_sel = 0
        if self.show_shelf and not self.ai_mode:
            self.toggle_ai()

    def change_sens(self, d):
        self.sens = int(np.clip(self.sens + d, 0, len(SENS_LEVELS) - 1))
        self.say(f"Sensitivity {self.sens + 1}/{len(SENS_LEVELS)}")

    def say(self, msg, dur=1.6):
        self.toast, self.toast_t, self.toast_dur = msg, time.perf_counter(), dur

    def act(self, title, detail="", dur=1.6):
        self.last_action = (title, detail, time.perf_counter())
        if not self.ai_mode:
            self.say(title, dur)

    def save_settings(self):
        if self.hwnd:
            rc = wintypes.RECT()
            user32.GetWindowRect(self.hwnd, ctypes.byref(rc))
            self.pos = (rc.left, rc.top)
            bottom = rc.bottom
        else:
            bottom = self.pos[1] + self.H
        data = dict(self.settings)
        data.pop("pos", None)
        data.update({"pos_lb": [self.pos[0], bottom], "sens": self.sens, "dual": self.dual, "mode": self.mode,
                     "hold_only_clipboard": self.hold_only, "natural_scroll": self.natural_scroll,
                     "link_enabled": self.link_enabled})
        save_json(SETTINGS, data)

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x0, y0, x1, y1) in self.buttons_by[self.layout].items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                {"close": self.quit, "gest": self.toggle_ai, "hands": self.toggle_dual,
                 "clips": self.toggle_shelf, "link": self.toggle_link_panel, "help": self.toggle_help}[name]()
                return
        for x0, y0, x1, y1, fn in self.link_hits:
            if x0 <= x <= x1 and y0 <= y <= y1:
                fn()
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
            try:  # Windows 11 rounded corners
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(ctypes.c_int(2)), 4)
            except Exception:
                pass
        self.chip_setup()
        for side in SIDES:
            self.portals[side] = {"hwnd": self.make_overlay(f"{PORTAL} {side}"), "key": None, "shown": False}
        if prev and prev != hwnd:
            user32.SetForegroundWindow(prev)
        self.hwnd = hwnd

    def set_layout(self, layout):
        """Switch between typing, gesture and guide layouts; the window keeps its bottom edge."""
        self.layout = layout
        h = self.heights[layout]
        if self.hwnd:
            rc = wintypes.RECT()
            user32.GetWindowRect(self.hwnd, ctypes.byref(rc))
            _, top, _, bottom = work_area()
            y = max(top, min(rc.bottom - h, bottom - h))
            user32.SetWindowPos(self.hwnd, wintypes.HWND(-1), rc.left, y, self.W, h, 0x0010)
        self.H = h

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
        last = parts[-1]
        key = last if len(last) == 1 else getattr(Key, last)  # tab, page_down, media_play_pause, ...
        for m in mods:
            self.out.press(m)
        try:
            self.out.tap(key)
        finally:
            for m in reversed(mods):
                self.out.release(m)

    def fg_process(self):
        fg = user32.GetForegroundWindow()
        if fg != self.proc[0]:
            self.proc = (fg, process_name(fg) if fg else "")
        return self.proc[1]

    def is_terminal(self, hwnd):
        return window_class(hwnd) in TERMINAL_CLASSES or self.fg_process().lower() in TERMINAL_APPS

    def lookup(self, name):
        per = self.gcfg["per_app"].get(self.fg_process().lower(), {})
        return per.get(name) or self.gcfg["motions"].get(name) or self.gcfg["poses"].get(name)

    def run_action(self, a, detail):
        if not a:
            self.act("Not set", f"{detail}  ·  map it in gestures.json")
            return
        name = a.get("name", detail)
        try:
            if "action" in a:
                {"shelf": self.toggle_shelf, "link": self.toggle_link_panel,
                 "clipboard_history": lambda: self.send_combo("win+v")}[a["action"]]()
            elif "keys" in a:
                self.send_combo(a["keys"])
            else:
                run = os.path.expandvars(a["run"])
                if run.lower().startswith("shell:"):
                    subprocess.Popen(["explorer.exe", run])
                else:
                    os.startfile(run)
            self.act(name, detail)
        except Exception:
            self.act(f"Could not run {name}", detail)

    def close_active_app(self):
        fg = user32.GetForegroundWindow()
        if not fg or fg == self.hwnd or window_class(fg) in ("Progman", "WorkerW", "Shell_TrayWnd"):
            self.act("No app to close", "Pinky")
            return
        title = window_title(fg) or "app"
        user32.PostMessageW(fg, 0x0010, 0, 0)  # WM_CLOSE: same as clicking the window's X
        self.act(f"Closing {title[:28]}", "Pinky")

    # ---------- gesture events ----------
    def handle(self, e, now):
        t = e["type"]
        if t == "pose":
            g = e["id"]
            if self.show_link and g in ("thumbs_up", "thumbs_down"):
                self.link_pose(g)
            elif g == "rock":
                self.toggle_ai()
            elif not self.ai_mode:
                return
            elif g == "pinky":
                self.close_active_app()
            else:
                self.run_action(self.lookup(g), POSE_LABELS.get(g, g))
        elif t == "swipe":
            name = ("two_" if e["two"] else "") + "swipe_" + e["dir"]
            self.fx_swipe = (e["dir"], now)
            self.run_action(self.lookup(name), ("Two-finger swipe " if e["two"] else "Swipe ") + e["dir"])
        elif t == "push":
            if self.show_link:
                self.link_primary()
            else:
                self.run_action(self.lookup("push"), "Push")
        elif t == "zoom":
            self.run_action(self.lookup("zoom_in" if e["n"] > 0 else "zoom_out"), "Two-hand pinch")
        elif t == "scroll":
            if not self.scrolling:  # a new pinch stops any glide
                self.scrolling, self.scroll_glide, self.scroll_vel = True, 0.0, 0.0
            speed = abs(e["amount"]) / max(e["dt"], 1e-3)  # palm sizes per second
            accel = float(np.clip(0.8 + speed * 0.25, 0.8, 2.0))  # fast pulls scroll further
            units = e["amount"] * SCROLL_GAIN * accel * (1 if self.natural_scroll else -1)
            self.scroll_pending += units
            self.scroll_vel = self.scroll_vel * 0.5 + units / max(e["dt"], 1e-3) * 0.5
            self.act("Scrolling", "Pinch + move")
        elif t == "scroll_end":
            self.scrolling = False
            self.scroll_glide = self.scroll_vel * 0.4 if abs(self.scroll_vel) > 900 else 0.0
            self.scroll_vel = 0.0
        elif t == "volume":
            for _ in range(abs(e["n"])):
                self.out.tap(Key.media_volume_up if e["n"] > 0 else Key.media_volume_down)
            self.act("Volume up" if e["n"] > 0 else "Volume down", "Pinch + twist")
        elif t == "catch":
            self.copy_begin(now)
        elif t == "splash":
            self.paste(now, e.get("track"))
        elif t == "drop":
            self.drop(e.get("reason"))
        elif t == "pointer_on":
            self.pointing = True
            self.act("Air pointer", "Pinch: click · twice: word · hold + drag: select")
        elif t == "pointer" and self.pointing:
            self.move_ptr(e["dx"], e["dy"], e["dt"], DRAG_PRECISION if self.mouse_is_down else 1.0)
        elif t == "carry":
            if self.pointing:  # the pointing hand steers; the fist just holds
                return
            if not self.carrying:
                self.carry_off[0] += e["dx"]
                self.carry_off[1] += e["dy"]
                if e.get("rewind") or math.hypot(*self.carry_off) < CARRY_START:
                    return
                self.carrying = True
            self.move_ptr(e["dx"], e["dy"], e["dt"])
        elif t == "click_down":
            self.snap_ptr()
            self.mouse_is_down = True
            self.mouse.press(Button.left)
        elif t == "click_up":
            self.mouse_is_down = False
            self.mouse.release(Button.left)
        elif t == "pointer_off":
            self.pointing = False
            self.act("Pointer off", "Open hand")

    def move_ptr(self, dx, dy, dt, scale=1.0):
        if self.ptr_target is None:
            pt = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            self.ptr, self.ptr_target, self.ptr_set = [float(pt.x), float(pt.y)], [float(pt.x), float(pt.y)], (pt.x, pt.y)
        speed = math.hypot(dx, dy) / max(dt, 1e-3)
        accel = float(np.clip(0.6 + speed * 2.0, 0.6, 1.8))  # slow hand = fine control, fast hand = big jumps
        gain = user32.GetSystemMetrics(0) / POINTER_SPAN * accel * scale
        vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
        vw, vh = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
        self.ptr_target[0] = float(np.clip(self.ptr_target[0] + dx * gain, vx, vx + vw - 1))
        self.ptr_target[1] = float(np.clip(self.ptr_target[1] + dy * gain, vy, vy + vh - 1))

    def snap_ptr(self):
        """Jump the cursor to its target now (before a click, so it lands where the hand aimed)."""
        if self.ptr_target is not None:
            self.ptr = list(self.ptr_target)
            self.ptr_set = (int(round(self.ptr[0])), int(round(self.ptr[1])))
            user32.SetCursorPos(*self.ptr_set)

    def input_tick(self, now, dt):
        """120 Hz: glide the cursor toward the hand and spread scrolling smoothly between camera frames."""
        if self.ptr_target is not None:
            pt = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            if self.ptr_set and (abs(pt.x - self.ptr_set[0]) > 3 or abs(pt.y - self.ptr_set[1]) > 3):
                self.ptr = [float(pt.x), float(pt.y)]  # the real mouse moved it: follow from there
                self.ptr_target = list(self.ptr)
            k = 1 - math.exp(-dt * POINTER_EASE)
            self.ptr[0] += (self.ptr_target[0] - self.ptr[0]) * k
            self.ptr[1] += (self.ptr_target[1] - self.ptr[1]) * k
            pos = (int(round(self.ptr[0])), int(round(self.ptr[1])))
            if pos != self.ptr_set:
                user32.SetCursorPos(*pos)
                self.ptr_set = pos
            if not (self.pointing or self.carrying) and math.hypot(self.ptr_target[0] - self.ptr[0],
                                                                   self.ptr_target[1] - self.ptr[1]) < 0.5:
                self.ptr_target = None
        if not self.scrolling and self.scroll_glide:
            self.scroll_pending += self.scroll_glide * dt
            self.scroll_glide *= math.exp(-dt / SCROLL_GLIDE)
            if abs(self.scroll_glide) < 60:
                self.scroll_glide = 0.0
        if abs(self.scroll_pending) > 0.5:
            out = self.scroll_pending * (1 - math.exp(-dt * SCROLL_EASE))
            if abs(self.scroll_pending) < 4:
                out = self.scroll_pending
            self.scroll_pending -= out
            self.scroll_frac += out
            units = int(self.scroll_frac)
            if units:
                self.scroll_frac -= units
                self.mouse.scroll(0, units / 120)  # high-resolution wheel: smooth in browsers and editors

    def shelf_add(self, item):
        if item.get("private"):
            return
        for old in list(self.shelf):
            if old["kind"] == item["kind"] and old["label"] == item["label"] and old.get("from") == item.get("from"):
                self.shelf.remove(old)
        self.shelf.appendleft(item)

    def hold(self, item):
        self.held_item, self.held_text = item, item["label"]

    def copy_begin(self, now):
        self.carrying, self.carry_off = False, [0.0, 0.0]
        self.fx_catch = now
        if self.show_link:  # the Link panel uses push / thumbs; a fist there means nothing
            self.engine.holder = None
            return
        if self.incoming and not self.show_shelf and now - self.incoming[1] < INCOMING_FRESH:
            self.take_incoming(hold=True)
            return
        if self.show_shelf:
            if not self.shelf:
                self.engine.holder = None
                self.act("The shelf is empty", "Catch")
                return
            item = self.shelf[min(self.shelf_sel, len(self.shelf) - 1)]
            clip_set_item(item)
            self.hold(item)
            self.shelf_add(item)
            self.show_shelf = False
            self.act("Taken from the shelf", "Move the fist to carry it")
            return
        fg = user32.GetForegroundWindow()
        if not fg or fg == self.hwnd:
            self.engine.holder = None
            self.act("Click a window first", "Catch")
            return
        self.clip_prev = clip_get_text() if self.hold_only else None
        self.pending_copy = (now, user32.GetClipboardSequenceNumber(), False)
        self.held_text = self.held_item = None
        self.send_combo("ctrl+shift+c" if self.is_terminal(fg) else "ctrl+c")

    def poll_copy(self, now):
        if not self.pending_copy:
            return
        t0, seq0, grabbed = self.pending_copy
        if user32.GetClipboardSequenceNumber() != seq0:
            self.pending_copy = None
            item = clip_read_item() or finish_item({"kind": "text", "text": ""})
            self.hold(item)
            if item["kind"] != "text" or item["text"].strip():
                self.shelf_add(item)
            self.act("Grabbed the word" if grabbed else "Copied", "Move the fist to carry it")
        elif self.pointing and not grabbed and now - t0 > WORD_GRAB_WAIT:
            # nothing was selected: double-click selects the word under the air pointer, then copy it
            self.mouse.click(Button.left, 2)
            self.pending_copy = (now, user32.GetClipboardSequenceNumber(), True)
            self.send_combo("ctrl+c")
        elif now - t0 > COPY_TIMEOUT:
            self.pending_copy = None
            self.held_text = self.held_item = None
            self.engine.holder = None
            self.act("Nothing selected to copy", "Select text first, then catch")

    def paste(self, now, tid):
        dropped_here = self.carrying
        side = self.portal_hot() if dropped_here else None
        if side:  # dropped into a friend's portal: throw it to their laptop
            self.throw(now, *side)
            self.carrying, self.carry_off = False, [0.0, 0.0]
            self.held_text = self.held_item = self.clip_prev = self.pending_copy = None
            return
        if dropped_here:
            self.snap_ptr()
            self.mouse.click(Button.left)  # place the caret where the fist carried the item
            time.sleep(0.06)
        fg = user32.GetForegroundWindow()
        self.send_combo("ctrl+shift+v" if fg and self.is_terminal(fg) else "ctrl+v")
        self.carrying, self.carry_off = False, [0.0, 0.0]
        tr = self.engine.track(tid)
        if tr is not None and tr.lm is not None:
            self.fx_splash = (now, sum(tr.lm[i].x for i in (0, 5, 9, 13, 17)) / 5,
                              sum(tr.lm[i].y for i in (0, 5, 9, 13, 17)) / 5)
        self.held_text, self.held_item, self.clip_prev, self.pending_copy = None, None, None, None
        self.act("Dropped here" if dropped_here else "Pasted at the caret", "Splash")

    def drop(self, reason):
        self.carrying, self.carry_off = False, [0.0, 0.0]
        if self.pending_copy or self.held_text is not None:
            if self.hold_only and self.clip_prev is not None:
                clip_set_text(self.clip_prev)
                msg = "Dropped · clipboard restored"
            else:
                msg = "Dropped · still on the clipboard"
            self.act(msg, "Hand lost" if reason == "lost" else "Opened slowly")
        self.held_text, self.held_item, self.clip_prev, self.pending_copy = None, None, None, None

    # ---------- Link ----------
    def link_start(self):
        if Link is None:
            self.say("Link needs the cryptography package (run setup.bat)")
            return
        try:
            self.link = Link(self.link_name, LINK_IDENTITY, RECEIVED, manual=self.settings.get("link_manual", []))
            self.link.start()
        except OSError as e:
            self.link = None
            self.say(f"Link unavailable: {e.__class__.__name__}")

    def link_stop(self):
        if self.link:
            self.link.stop()
            self.link = None
        self.link_view, self.link_conn, self.ghosts = [], [], {}

    def peer_name(self, pid):
        return next((p["name"] for p in self.link_view if p["id"] == pid), "your friend")

    def pump_link(self, now):
        if self.link_toggle:
            self.link_toggle = False
            self.link_enabled = not self.link_enabled
            (self.link_start if self.link_enabled else self.link_stop)()
            self.say("Link on" if self.link_enabled else "Link off")
        if not self.link:
            return
        for e in self.link.poll():
            t = e["type"]
            if t not in ("ghost", "progress"):
                self.link_view_t = 0.0  # refresh the device list
            if t == "connected":
                self.act(f"Linked with {e['name']}", "Link", dur=4.0)
            elif t == "disconnected":
                self.ghosts.pop(e["peer"], None)
                self.act(f"{e['name']} went offline", "Link")
            elif t == "pair_request":
                self.act(f"{e['name']} wants to pair", "Open Link: two-finger swipe up", dur=4.0)
                if self.ai_mode and not self.show_link:
                    self.toggle_link_panel()
            elif t == "pair_code":
                if self.ai_mode and not self.show_link:
                    self.toggle_link_panel()
                self.act("Do both screens show the same code?", "Thumbs up or Confirm")
            elif t == "paired":
                self.act(f"Paired with {e['name']}", "Carry something to the screen edge to throw it")
            elif t == "pair_failed":
                self.act(e.get("reason") or "Pairing stopped", "Link")
            elif t == "error":
                self.act(e["message"], "Link")
            elif t == "progress":
                tr = self.transfers.setdefault(e["id"], {"dir": e["dir"], "name": self.peer_name(e["peer"])})
                tr.update(frac=e["frac"], done=e["done"], t=now)
            elif t == "transfer_failed":
                self.transfers.pop(e["id"], None)
                self.act("A transfer stopped (connection lost)", "Link")
            elif t == "item":
                item = finish_item(e["item"])
                self.shelf_add(item)
                self.incoming = (item, now)
                what = {"text": "a link" if item.get("url") else "some text", "image": "an image",
                        "files": "files"}[item["kind"]]
                self.act(f"{item['from']} threw you {what}", "Catch to take it" if self.ai_mode else "Ctrl+Alt+V shelf", dur=4.0)
            elif t == "taken":
                self.act(f"{e['name']} caught it", "Link")
            elif t == "ghost":
                self.ghosts[e["peer"]] = (e["hands"], e["holding"], now)
        if now - self.link_ips_t > 5:
            self.link_ips_t, self.link_ips = now, ", ".join(local_ipv4s()[:2])
        if now - self.link_view_t > 0.5:
            self.link_view_t = now
            self.link_view = self.link.snapshot()
            self.link_conn = self.link.connected()
            if self.link_sel >= max(1, len(self.link_view)):
                self.link_sel = max(0, len(self.link_view) - 1)

    def take_incoming(self, hold):
        item, _ = self.incoming
        self.incoming = None
        clip_set_item(item)
        if hold:
            self.hold(item)
            self.fx_catch = time.perf_counter()
        else:
            self.engine.holder = None
        if self.link:
            self.link.taken(item["peer"], item["tid"])
        self.act(f"Caught from {item['from']}", "Move the fist to carry it" if hold else "On your clipboard: Ctrl+V")

    def open_incoming_url(self):
        item, _ = self.incoming
        url = item["text"].strip()
        if item.get("url") and url.lower().startswith(("http://", "https://")):
            os.startfile(url)
            self.incoming = None
            self.act("Opened the link", item["from"])

    def screen_rect(self):
        vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
        return vx, vy, vx + user32.GetSystemMetrics(78), vy + user32.GetSystemMetrics(79)

    def portal_hot(self):
        """(peer id, name, side) whose screen-edge portal the cursor is in, or None."""
        if not self.link_conn:
            return None
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        x0, y0, x1, y1 = self.screen_rect()
        near = {"right": pt.x >= x1 - PORTAL_ZONE, "left": pt.x <= x0 + PORTAL_ZONE, "top": pt.y <= y0 + PORTAL_ZONE}
        return next(((pid, name, side) for pid, name, side in self.link_conn if near.get(side)), None)

    def throw(self, now, pid, name, side):
        item = self.held_item
        if item is None:
            self.act("Still copying, catch it again", "Throw")
            return
        if item.get("private"):
            self.act("Private items stay on this laptop", "Password manager")
            return
        send = {k: item[k] for k in ("kind", "label", "text", "png", "paths") if k in item}
        tid = self.link.send_item(pid, send) if self.link else None
        if tid:
            self.transfers[tid] = {"dir": "out", "name": name, "frac": 0.0, "done": False, "t": now}
            self.fx_swipe = ({"right": "right", "left": "left", "top": "up"}[side], now)
            self.act(f"Thrown to {name}", item["label"][:40])
        else:
            self.act(f"Could not send to {name}", "Link")

    def link_selected(self):
        return self.link_view[self.link_sel] if 0 <= self.link_sel < len(self.link_view) else None

    def link_primary(self, p=None):
        """Push or the row button: the obvious next step for that device."""
        p = p or self.link_selected()
        if not p or not self.link:
            return
        if p["pair"] == "incoming":
            self.link.accept_pair(p["id"])
        elif p["pair"] == "code":
            self.link.confirm_pair(p["id"])
        elif not p["paired"] and not p["pair"] and p["online"]:
            self.link.request_pair(p["id"])
            self.act(f"Asking {p['name']} to pair", "They push or click Accept")
        self.link_view_t = 0.0

    def link_pose(self, g):
        p = next((d for d in self.link_view if d["pair"] in ("code", "incoming")), None) or self.link_selected()
        if not p or not self.link:
            return
        if g == "thumbs_up" and p["pair"] == "code":
            self.link.confirm_pair(p["id"])
        elif g == "thumbs_down" and p["pair"]:
            self.link.cancel_pair(p["id"])
        self.link_view_t = 0.0

    def link_cycle_side(self, p):
        nxt = SIDES[(SIDES.index(p["side"]) + 1) % len(SIDES)] if p["side"] in SIDES else SIDES[0]
        self.link.set_side(p["id"], nxt)
        self.link_view_t = 0.0

    def send_ghosts(self, now):
        if not (self.link and self.link_conn and self.ai_mode) or now - self.ghost_t < 1 / GHOST_HZ:
            return
        hands = [[[round(q.x, 3), round(q.y, 3)] for q in tr.lm] for tr in self.engine.tracks if tr.updated and tr.lm]
        if hands:
            self.ghost_t = now
            self.link.send_ghost(hands, self.engine.holder is not None)

    # ---------- hand logic (typing) ----------
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

    def on_detection(self, hands, now):
        self.vis = [lm for lm, _ in hands]
        if self.rec:
            self.rec.write(json.dumps({"t": round(now, 4), "control": self.ai_mode, "hands": [
                [[round(q.x, 5), round(q.y, 5), round(q.z, 5)] for q in lm] for lm, _ in hands]}) + "\n")
        if self.engine_reset:
            self.engine_reset = False
            for e in self.engine.reset(now):
                self.handle(e, now)
        self.engine.menu = (self.show_shelf or self.show_link) and self.ai_mode
        self.engine.menu_poses = {"rock", "thumbs_up", "thumbs_down"} if self.show_link else {"rock"}
        self.engine.menu_push = self.show_link
        for e in self.engine.update([lm for lm, _ in hands], now, self.ai_mode):
            self.handle(e, now)
        rows = len(self.shelf) if self.show_shelf else len(self.link_view) if self.show_link else 0
        if self.engine.menu and rows:
            # hand height picks a row; frozen while the hand closes so a catch takes what you saw
            tr = next((tr for tr in self.engine.tracks if tr.updated and tr.id != self.engine.holder), None)
            if tr is not None and tr.last.up4 >= 3:
                sel = int(float(np.clip((tr.last.center[1] - 0.25) / 0.5, 0, 0.999)) * rows)
                if self.show_shelf:
                    self.shelf_sel = sel
                else:
                    self.link_sel = sel
        self.send_ghosts(now)

        if self.ai_mode:
            for h in self.hands.values():
                h.reset()
            return

        primary = max(hands, key=lambda hl: hl[0][0].x)[0] if hands else None  # right-most wrist
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
    def text_sprite(self, s, font, anchor):
        key = (s, id(font), anchor)
        sp = self.sprites.get(key)
        if sp is None:
            if len(self.sprites) > 800:
                self.sprites.clear()
            l, t, r, b = font.getbbox(s, anchor=anchor)
            im = Image.new("L", (max(1, r - l), max(1, b - t)), 0)
            ImageDraw.Draw(im).text((-l, -t), s, font=font, fill=255, anchor=anchor)
            sp = (np.asarray(im, np.float32)[..., None] / 255.0, l, t)
            self.sprites[key] = sp
        return sp

    def render_text(self, img, texts):
        """Draw text in place from cached anti-aliased glyph masks (no full-image PIL round trip)."""
        H, W = img.shape[:2]
        for x, y, s, col, font, anchor in texts:
            if not s:
                continue
            a, l, t = self.text_sprite(s, font, anchor)
            x0, y0 = int(round(x + l)), int(round(y + t))
            h, w = a.shape[:2]
            cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
            if cx0 >= cx1 or cy0 >= cy1:
                continue
            aa = a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            reg = img[cy0:cy1, cx0:cx1]
            reg[:] = (reg + (np.asarray(col, np.float32) - reg) * aa).astype(np.uint8)
        return img

    def font_for(self, lab):
        if lab in ("⌫", "⇧"):
            return self.f_sym
        return self.f_key if len(lab) == 1 else self.f_small

    def base_for(self, layout):
        key = (layout, self.shift, self.mode) if layout == "type" else (layout, self.help_page)
        if key in self.base_cache:
            return self.base_cache[key]
        W, H, pad = self.W, self.heights[layout], self.pad
        img = np.full((H, W, 3), BG, np.uint8)
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), LINE, 1)
        texts = [(pad + 2, 22, "AirTyper", TEXT, self.f_title, "lm")]
        if layout == "type":
            rrect(img, pad, self.bar_y, W - pad, self.bar_y + self.bar_h, 10, SURF)
            for k in self.keys:
                x0, y0, x1, y1 = k["r"]
                single = len(k["id"]) == 1
                rrect(img, x0, y0, x1, y1, 7, SURF if single else SURF2)
                lab = key_label(k["id"], self.shift, self.mode)
                col = TEXT if single else MUTED
                if k["id"] == "MODE" or (k["id"] == "SHIFT" and self.shift):
                    col = AMBER
                texts.append(((x0 + x1) / 2, (y0 + y1) / 2, lab, col, self.font_for(lab), "mm"))
        elif layout == "help":
            y = self.header_h + 6
            texts.append((W - pad - 2, 22, f"page {self.help_page + 1}/2 · ? next", MUTED, self.f_small, "rm"))
            for name, action in self.help_pages[self.help_page]:
                if action is None:
                    y += 8
                    texts.append((pad + 2, y + 9, name.upper(), ACCENT, self.f_btn, "lm"))
                    y += 20
                else:
                    texts.append((pad + 2, y + self.help_row_h / 2, name, TEXT, self.f_small, "lm"))
                    texts.append((W - pad - 2, y + self.help_row_h / 2, action, MUTED, self.f_small, "rm"))
                    y += self.help_row_h
            cv2.line(img, (pad, H - 46), (W - pad, H - 46), LINE, 1)
            texts.append((pad + 2, H - 32, "Ctrl+Alt +  A gestures · G guide · V shelf · L Link · K Link off",
                          MUTED, self.f_small, "lm"))
            texts.append((pad + 2, H - 14, "H hide · Q quit · Up/Down sensitivity · R re-centre · F9 record",
                          MUTED, self.f_small, "lm"))
        out = self.render_text(img, texts)
        self.base_cache[key] = out
        return out

    def cam_mask(self, w, h):
        if (w, h) not in self.mask_cache:
            m = np.zeros((h, w), np.uint8)
            rrect(m, 0, 0, w - 1, h - 1, 12, 255)
            self.mask_cache[(w, h)] = (m > 127)[..., None]
        return self.mask_cache[(w, h)]

    def draw_camera(self, img, frame, cam_h):
        W, pad = self.W, self.pad
        cw = W - 2 * pad
        full_h = int(cw * 0.75)
        off = (full_h - cam_h) // 2
        roi = img[self.cam_y:self.cam_y + cam_h, pad:pad + cw]
        if frame is not None:
            fh = frame.shape[0]
            band = fh * cam_h / full_h  # only the visible strip of the camera frame
            fy0 = int((fh - band) / 2)
            cam = cv2.resize(frame[fy0:fy0 + int(band)], (cw, cam_h), interpolation=cv2.INTER_LINEAR)
            g = cv2.cvtColor(cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            view = cv2.addWeighted(cam, 0.15, g, 0.35, 10)  # mostly grey, dimmed
        else:
            view = np.full_like(roi, SURF)
        np.copyto(roi, view, where=self.cam_mask(cw, cam_h))

        def local(x, y):
            return int(x * cw), int(y * full_h - off)
        return roi, local

    def draw_skeleton(self, roi, local, lm, col):
        pts = [local(q.x, q.y) for q in lm]
        for a, b in HAND_BONES:
            cv2.line(roi, pts[a], pts[b], col, 1, cv2.LINE_AA)
        for i, q in enumerate(pts):
            cv2.circle(roi, q, 2, TEXT if i in (4, 8) else col, -1, cv2.LINE_AA)
        return pts

    def cam_status(self, texts, cam_h, fps, now, mode_label, mode_col):
        pad, W, y = self.pad, self.W, self.cam_y
        n = len(self.vis)
        texts.append((pad + 12, y + 14, "●", ACCENT if n else RED, self.f_small, "lm"))
        texts.append((pad + 26, y + 14, f"{n} hand{'s' if n != 1 else ''}" if n else "No hand",
                      TEXT if n else RED, self.f_small, "lm"))
        if self.rec:
            mode_label, mode_col = "● REC", RED
        texts.append((W - pad - 12, y + 14, mode_label, mode_col, self.f_small, "rm"))
        texts.append((pad + 12, y + cam_h - 12, f"{fps:.0f} fps", MUTED, self.f_small, "lm"))
        if not n:
            mid = (W / 2, y + cam_h / 2)
            if self.booting:
                texts.append((mid[0], mid[1], "Starting camera" + "." * (int(now * 3) % 4), AMBER, self.f_sug, "mm"))
            else:
                texts.append((mid[0], mid[1], "Raise your hand", MUTED, self.f_sug, "mm"))

    def progress_pill(self, img, texts, msg, p, cy):
        tw = int(self.f_sug.getlength(msg)) + 36
        x0, x1 = (self.W - tw) // 2, (self.W + tw) // 2
        rrect(img, x0, cy - 15, x1, cy + 15, 15, SURF2)
        rrect(img, x0 + 12, cy + 9, x0 + 12 + max(2, (tw - 24) * p), cy + 11, 1, AMBER)
        texts.append((self.W / 2, cy - 2, msg, TEXT, self.f_sug, "mm"))

    def draw(self, frame, now, fps):
        img = self.base_for(self.layout).copy()
        texts = []
        self.link_hits = []
        states = {"gest": self.ai_mode, "hands": self.dual, "help": self.show_help, "clips": self.show_shelf,
                  "link": self.show_link}
        labels = {"close": "×", "gest": "Gestures", "hands": "2H" if self.dual else "1H", "help": "?",
                  "clips": "Clips", "link": "Link"}
        if self.link_conn:  # linked: a dot after the title
            cv2.circle(img, (int(self.pad + 13 + self.f_title.getlength("AirTyper")), 23), 4, ACCENT, -1, cv2.LINE_AA)
        for name, (x0, y0, x1, y1) in self.buttons_by[self.layout].items():
            on = states.get(name, False)
            rrect(img, x0, y0, x1, y1, 8, ACCENT if on else SURF2)
            col = BG if on else (MUTED if name == "close" else TEXT)
            texts.append(((x0 + x1) / 2, (y0 + y1) / 2, labels[name], col,
                          self.f_sug if name == "close" else self.f_btn, "mm"))
        if self.layout == "type":
            self.draw_type(img, frame, texts, now, fps)
        elif self.layout == "control":
            self.draw_control(img, frame, texts, now, fps)
        elif self.layout == "shelf":
            self.draw_shelf(img, frame, texts, now, fps)
        elif self.layout == "link":
            self.draw_link(img, frame, texts, now, fps)

        if self.toast and now - self.toast_t < self.toast_dur and self.layout != "help":
            tw = int(self.f_sug.getlength(self.toast)) + 28
            x0, y0 = (self.W - tw) // 2, self.cam_y + 30
            rrect(img, x0, y0, x0 + tw, y0 + 28, 14, SURF2)
            texts.append((self.W / 2, y0 + 14, self.toast, TEXT, self.f_sug, "mm"))
        return self.render_text(img, texts)

    def draw_type(self, img, frame, texts, now, fps):
        W, pad = self.W, self.pad
        cam_h = self.cam_h_type
        roi, local = self.draw_camera(img, frame, cam_h)
        for h in self.active_hands():
            if h.lm is None:
                continue
            zw, zh = self.zone_dims(h)
            cx, cy = h.zone_c
            z0, z1 = local(cx - zw / 2, cy - zh / 2), local(cx + zw / 2, cy + zh / 2)
            rrect(roi, z0[0], z0[1], z1[0], z1[1], 8, ACCENT_DIM, 1)
            if self.dual:
                texts.append((pad + z0[0] + 6, self.cam_y + z0[1] + 10, h.name, ACCENT, self.f_small, "lm"))
        for lm in self.vis:
            pts = self.draw_skeleton(roi, local, lm, ACCENT_DIM)
            km = ((pts[5][0] + pts[9][0]) // 2, (pts[5][1] + pts[9][1]) // 2)
            cv2.circle(roi, km, 4, AMBER, -1, cv2.LINE_AA)
        if self.hand_paused(now):
            state = "Keyboard"
        else:
            state = ("2-hand " if self.dual else "") + self.mode
        self.cam_status(texts, cam_h, fps, now, state.capitalize(), MUTED)
        pr = self.engine.progress(now)
        if pr and pr[0] == "rock":
            self.progress_pill(img, texts, "Rock  →  gesture mode", pr[1], self.cam_y + cam_h - 26)

        # target window + echo
        if now - self.target_t > 0.4:
            fg = user32.GetForegroundWindow()
            self.target = "" if fg == self.hwnd else window_title(fg)
            self.target_t = now
        if self.target:
            texts.append((pad + 2, self.tgt_y + 8, "Typing into  " + self.target[:46], MUTED, self.f_small, "lm"))
        else:
            texts.append((pad + 2, self.tgt_y + 8, "Click the window you want to type into", RED, self.f_small, "lm"))
        shown = self.buffer
        while shown and self.f_text.getlength(shown + "_") > W - 2 * pad - 30:
            shown = shown[1:]
        caret = "▌" if int(now * 2) % 2 == 0 else " "
        texts.append((pad + 12, self.bar_y + self.bar_h / 2, shown + caret, TEXT, self.f_text, "lm"))
        self.draw_keyboard(img, texts, now)

    def draw_keyboard(self, img, texts, now):
        W = self.W
        hands = self.active_hands()
        for sk in self.sug_keys:
            if not sk["word"]:
                continue
            x0, y0, x1, y1 = sk["r"]
            hot = any(h.hover is sk for h in hands)
            rrect(img, x0, y0, x1, y1, 9, ACCENT if hot else SURF)
            texts.append(((x0 + x1) / 2, (y0 + y1) / 2, sk["word"], BG if hot else TEXT, self.f_sug, "mm"))

        hovered = {id(h.hover) for h in hands if h.hover is not None}
        for k in self.keys:
            x0, y0, x1, y1 = k["r"]
            t = self.flash.get(k["id"])
            lab = key_label(k["id"], self.shift, self.mode)
            font = self.font_for(lab)
            if t and now - t < 0.15:
                rrect(img, x0, y0, x1, y1, 7, AMBER)
                texts.append(((x0 + x1) / 2, (y0 + y1) / 2, lab, BG, font, "mm"))
            elif id(k) in hovered:
                rrect(img, x0, y0, x1, y1, 7, ACCENT)
                texts.append(((x0 + x1) / 2, (y0 + y1) / 2, lab, BG, font, "mm"))
                if len(k["id"]) == 1:  # magnified callout
                    bw, bh = int(self.unit * 1.25), int(self.key_h * 1.15)
                    bx0 = int(np.clip((x0 + x1) / 2 - bw / 2, 2, W - bw - 2))
                    by0 = y0 - bh - 6
                    rrect(img, bx0, by0, bx0 + bw, by0 + bh, 9, SURF2)
                    rrect(img, bx0, by0, bx0 + bw, by0 + bh, 9, ACCENT, 1)
                    texts.append((bx0 + bw / 2, by0 + bh / 2, lab, TEXT, self.f_pop, "mm"))

        self.ripples = [rp for rp in self.ripples if now - rp[2] < 0.35]
        for x, y, t in self.ripples:
            p = (now - t) / 0.35
            cv2.circle(img, (x, y), int(8 + p * self.unit * 1.2), mix(AMBER, BG, p), 2, cv2.LINE_AA)

        for h in hands:
            if h.disp is None:
                continue
            cx, cy = int(h.disp[0]), int(h.disp[1])
            r = int(self.unit * 0.42)
            cv2.circle(img, (cx, cy), r, ACCENT, 2, cv2.LINE_AA)
            inner = max(3, int(r * 0.8 * (1 - 0.75 * h.pinch_amt)))
            cv2.circle(img, (cx, cy), inner, AMBER, 2 if h.pinched else 1, cv2.LINE_AA)
            if self.dual:
                texts.append((cx + r + 4, cy - r, h.name, ACCENT, self.f_small, "lm"))
            if self.mode == "dwell" and h.hover is not None and h.dwell_key is h.hover:
                prog = min((now - h.dwell_start) / DWELL_TIME, 1.0)
                cv2.ellipse(img, (cx, cy), (r + 5, r + 5), -90, 0, 360 * prog, AMBER, 2, cv2.LINE_AA)

    def pose_action_name(self, g):
        if g in BUILTIN_POSES:
            return BUILTIN_POSES[g]
        if g == "rock":
            return "Back to typing"
        return (self.lookup(g) or {}).get("name", "Not set")

    def draw_control(self, img, frame, texts, now, fps):
        W, pad = self.W, self.pad
        cam_h = self.cam_h_ctrl
        roi, local = self.draw_camera(img, frame, cam_h)
        cw = W - 2 * pad
        for tr in self.engine.tracks:
            if not tr.updated or tr.lm is None:
                continue
            holder = tr.id == self.engine.holder
            col = AMBER if holder else (ACCENT if tr.pointer or tr.pinching else ACCENT_DIM)
            pts = self.draw_skeleton(roi, local, tr.lm, col)
            c = local(sum(tr.lm[i].x for i in (0, 5, 9, 13, 17)) / 5, sum(tr.lm[i].y for i in (0, 5, 9, 13, 17)) / 5)
            tag = None
            if holder:
                tag = "Holding"
                cv2.circle(roi, c, int(26 + 3 * math.sin(now * 6)), AMBER, 2, cv2.LINE_AA)
            elif tr.pointer:
                tag = "Pointer"
                km = ((pts[5][0] + pts[9][0]) // 2, (pts[5][1] + pts[9][1]) // 2)
                cv2.circle(roi, km, 8, ACCENT, -1 if tr.mouse_down else 2, cv2.LINE_AA)
            elif tr.pinching:
                tag = {"scroll": "Scroll", "volume": "Volume", "zoom": "Zoom"}.get(tr.pinch_mode, "Pinch")
            if tag:
                texts.append((pad + c[0], self.cam_y + max(14, c[1] - 44), tag, col, self.f_btn, "mm"))

        self.draw_ghosts(roi, local, texts, now)
        if self.fx_catch and now - self.fx_catch < 0.35:
            p = (now - self.fx_catch) / 0.35
            tr = self.engine.track(self.engine.holder)
            if tr is not None and tr.lm is not None:
                c = local(tr.lm[9].x, tr.lm[9].y)
                cv2.circle(roi, c, int(70 * (1 - p)) + 26, mix(AMBER, BG, p), 2, cv2.LINE_AA)
        if self.fx_splash and now - self.fx_splash[0] < 0.5:
            p = (now - self.fx_splash[0]) / 0.5
            c = local(self.fx_splash[1], self.fx_splash[2])
            for k in range(3):
                rad = int(20 + (p * 90) + k * 14)
                cv2.circle(roi, c, rad, mix(AMBER, BG, min(1, p + k * 0.2)), 2, cv2.LINE_AA)
        if self.fx_swipe and now - self.fx_swipe[1] < 0.45:
            d, t0 = self.fx_swipe
            p = (now - t0) / 0.45
            vx, vy = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[d]
            ax = W / 2 + vx * (p - 0.5) * cw * 0.5
            ay = self.cam_y + cam_h / 2 + vy * (p - 0.5) * cam_h * 0.5
            arrow = {"left": "←", "right": "→", "up": "↑", "down": "↓"}[d]
            texts.append((ax, ay, arrow, mix(ACCENT, BG, p), self.f_arrow, "mm"))

        self.cam_status(texts, cam_h, fps, now, "Gestures", ACCENT)

        # action card: pose being held, last action, or idle hint
        x0, y0, x1, y1 = pad, self.card_y, W - pad, self.card_y + self.card_h
        rrect(img, x0, y0, x1, y1, 12, SURF)
        pr = self.engine.progress(now)
        title, detail, t_act = self.last_action
        if pr and pr[1] > 0.08:
            g, p = pr
            texts.append((x0 + 16, y0 + 18, POSE_LABELS.get(g, g) + "  ·  hold still", MUTED, self.f_small, "lm"))
            texts.append((x0 + 16, y0 + 37, self.pose_action_name(g), TEXT, self.f_body, "lm"))
            rrect(img, x0 + 16, y1 - 10, x1 - 16, y1 - 7, 1, SURF2)
            rrect(img, x0 + 16, y1 - 10, x0 + 16 + max(3, (x1 - x0 - 32) * p), y1 - 7, 1, AMBER)
        elif self.incoming and now - self.incoming[1] < INCOMING_FRESH and self.engine.holder is None:
            item = self.incoming[0]
            rrect(img, x0, y0, x1, y1, 12, AMBER, 1)
            hint = "catch to take · click to copy"
            if item.get("risky"):
                hint = "program file: only if you trust them"
            texts.append((x0 + 16, y0 + 18, f"From {item['from']}", AMBER, self.f_btn, "lm"))
            texts.append((x1 - 16, y0 + 18, hint, RED if item.get("risky") else MUTED, self.f_small, "rm"))
            right = x1 - 16
            if item.get("url"):
                bx0 = x1 - 66
                rrect(img, bx0, y0 + 28, x1 - 12, y0 + 50, 8, ACCENT)
                texts.append(((bx0 + x1 - 12) / 2, y0 + 39, "Open", BG, self.f_btn, "mm"))
                self.link_hits.append((bx0, y0 + 28, x1 - 12, y0 + 50, self.open_incoming_url))
                right = bx0 - 8
            texts.append((x0 + 16, y0 + 39, self.trim(item["label"], self.f_sug, right - x0 - 16), TEXT, self.f_sug, "lm"))
            self.link_hits.append((x0, y0, x1, y1, lambda: self.take_incoming(hold=False)))
        elif (tx := next((v for v in self.transfers.values() if not v.get("done") and now - v.get("t", 0) < 15), None)):
            verb = "Sending to" if tx["dir"] == "out" else "Receiving from"
            texts.append((x0 + 16, y0 + 18, "LINK", ACCENT, self.f_btn, "lm"))
            texts.append((x1 - 16, y0 + 18, f"{tx['frac'] * 100:.0f}%", MUTED, self.f_small, "rm"))
            texts.append((x0 + 16, y0 + 37, f"{verb} {tx['name']}", TEXT, self.f_body, "lm"))
            rrect(img, x0 + 16, y1 - 10, x1 - 16, y1 - 7, 1, SURF2)
            rrect(img, x0 + 16, y1 - 10, x0 + 16 + max(3, (x1 - x0 - 32) * tx["frac"]), y1 - 7, 1, ACCENT)
        elif title and now - t_act < 1.8:
            fade = max(0.0, (now - t_act - 1.2) / 0.6)
            cv2.circle(img, (x0 + 22, (y0 + y1) // 2), 5, mix(ACCENT, SURF, fade), -1, cv2.LINE_AA)
            texts.append((x0 + 38, y0 + 20, detail, mix(MUTED, SURF, fade), self.f_small, "lm"))
            texts.append((x0 + 38, y0 + 39, title, mix(TEXT, SURF, fade), self.f_body, "lm"))
        else:
            texts.append((W / 2, y0 + 21, "Ready", TEXT, self.f_body, "mm"))
            texts.append((W / 2, y0 + 40, "Swipe · catch · point · pinch · hold a shape", MUTED, self.f_small, "mm"))

        # clipboard card
        x0, y0, x1, y1 = pad, self.clip_y, W - pad, self.clip_y + self.clip_h
        holding = self.engine.holder is not None
        rrect(img, x0, y0, x1, y1, 12, SURF)
        if holding:
            rrect(img, x0, y0, x1, y1, 12, AMBER, 1)
            if self.held_text is None:
                texts.append((x0 + 16, y0 + 20, "Copying…", AMBER, self.f_btn, "lm"))
            else:
                texts.append((x0 + 16, y0 + 18, "CARRYING" if self.carrying else "HOLDING", AMBER, self.f_btn, "lm"))
                hot = self.portal_hot() if self.carrying else None
                hint = (f"flick open: throw to {hot[1]}" if hot else "flick open to drop here") if self.carrying \
                    else "move the fist · flick open"
                texts.append((x1 - 16, y0 + 18, hint, MUTED, self.f_small, "rm"))
                prev = self.trim(" ".join(self.held_text.split()), self.f_sug, x1 - x0 - 40)
                texts.append((x0 + 16, y0 + 39, f"“{prev}”" if prev else "(empty)", TEXT, self.f_sug, "lm"))
        else:
            texts.append((x0 + 16, y0 + 19, "Clipboard", MUTED, self.f_small, "lm"))
            if self.shelf:
                texts.append((x1 - 16, y0 + 19, f"{len(self.shelf)} on the shelf · 2-finger swipe ↓",
                              MUTED, self.f_small, "rm"))
            texts.append((x0 + 16, y0 + 38, "Select (point + pinch), then catch: open hand → fist",
                          TEXT, self.f_small, "lm"))
        texts.append((W / 2, self.clip_y + self.clip_h + 15, "Ctrl+Alt+G  gesture guide", MUTED, self.f_small, "mm"))

    @staticmethod
    def trim(s, font, maxw):
        s = s[:300]  # a huge copied text must not reach the font measurer (it rejects >1M chars)
        if font.getlength(s) <= maxw:
            return s
        while s and font.getlength(s + "…") > maxw:
            s = s[:-1]
        return s + "…"

    def draw_shelf(self, img, frame, texts, now, fps):
        W, pad = self.W, self.pad
        cam_h = self.cam_h_type
        roi, local = self.draw_camera(img, frame, cam_h)
        for tr in self.engine.tracks:
            if tr.updated and tr.lm is not None:
                self.draw_skeleton(roi, local, tr.lm, AMBER if tr.id == self.engine.holder else ACCENT_DIM)
        self.draw_ghosts(roi, local, texts, now)
        self.cam_status(texts, cam_h, fps, now, "Shelf", AMBER)
        y = self.shelf_y
        texts.append((pad + 2, y + 10, "Clipboard shelf", TEXT, self.f_body, "lm"))
        texts.append((W - pad - 2, y + 10, "hand up / down · catch to take", MUTED, self.f_small, "rm"))
        y += 28
        if not self.shelf:
            texts.append((W / 2, y + 40, "Nothing caught yet", MUTED, self.f_body, "mm"))
            texts.append((W / 2, y + 62, "Things you catch, or friends throw you, land here", MUTED, self.f_small, "mm"))
        for i, item in enumerate(self.shelf):
            y0 = y + i * self.shelf_row
            y1 = y0 + self.shelf_row - 6
            sel = i == self.shelf_sel
            rrect(img, pad, y0, W - pad, y1, 9, SURF2 if sel else SURF)
            if sel:
                rrect(img, pad, y0, W - pad, y1, 9, ACCENT, 1)
            texts.append((pad + 14, (y0 + y1) / 2, str(i + 1), ACCENT if sel else MUTED, self.f_btn, "lm"))
            tag_w = 0
            if item.get("from"):
                tag = f"from {item['from']}"
                tag_w = int(self.f_small.getlength(tag)) + 12
                texts.append((W - pad - 10, (y0 + y1) / 2, tag, GHOST, self.f_small, "rm"))
            prev = self.trim(item["label"], self.f_sug, W - 2 * pad - 50 - tag_w)
            texts.append((pad + 34, (y0 + y1) / 2, prev, TEXT if sel else MUTED, self.f_sug, "lm"))
        texts.append((W / 2, self.heights["shelf"] - 16, "Win+V  everything you copied (Windows clipboard history)",
                      MUTED, self.f_small, "mm"))

    def draw_ghosts(self, roi, local, texts, now):
        """A linked friend's hands, streamed as 21 points each (never their camera image)."""
        for pid, (hands, holding, t) in list(self.ghosts.items()):
            age = now - t
            if age > 0.6:
                continue
            col = mix(AMBER if holding else GHOST, SURF, min(1.0, age / 0.6) * 0.7)
            for pts in hands:
                if len(pts) != 21:
                    continue
                q = [local(x, y) for x, y in pts]
                for a, b in HAND_BONES:
                    cv2.line(roi, q[a], q[b], col, 1, cv2.LINE_AA)
                texts.append((self.pad + q[0][0], self.cam_y + min(q[0][1] + 14, roi.shape[0] - 8),
                              self.peer_name(pid), col, self.f_small, "mm"))

    def draw_link(self, img, frame, texts, now, fps):
        W, pad = self.W, self.pad
        cam_h = self.cam_h_type
        roi, local = self.draw_camera(img, frame, cam_h)
        for tr in self.engine.tracks:
            if tr.updated and tr.lm is not None:
                self.draw_skeleton(roi, local, tr.lm, ACCENT_DIM)
        self.draw_ghosts(roi, local, texts, now)
        self.cam_status(texts, cam_h, fps, now, "Link", ACCENT)
        y = self.shelf_y
        texts.append((pad + 2, y + 10, "Nearby devices", TEXT, self.f_body, "lm"))
        texts.append((W - pad - 2, y + 10, "hand up / down · push = pair", MUTED, self.f_small, "rm"))
        y += 28
        H = self.heights["link"]
        if not self.link:
            msg = "Link is off" if not self.link_enabled else "Link could not start"
            texts.append((W / 2, y + 40, msg, MUTED, self.f_body, "mm"))
            texts.append((W / 2, y + 62, "Ctrl+Alt+K to turn it on", MUTED, self.f_small, "mm"))
        elif not self.link_view:
            dots = "." * (int(now * 2) % 4)
            texts.append((W / 2, y + 40, "Looking for AirTyper on this network" + dots, TEXT, self.f_body, "mm"))
            texts.append((W / 2, y + 62, "Open AirTyper on your friend's laptop, same Wi-Fi", MUTED, self.f_small, "mm"))
            texts.append((W / 2, y + 80, "Allow it in the Windows Firewall prompt (Private networks)", MUTED,
                          self.f_small, "mm"))
        for i, p in enumerate(self.link_view[:4]):
            y0 = y + i * self.link_row
            y1 = y0 + self.link_row - 6
            sel = i == self.link_sel
            rrect(img, pad, y0, W - pad, y1, 10, SURF2 if sel else SURF)
            if sel:
                rrect(img, pad, y0, W - pad, y1, 10, ACCENT, 1)
            dot = ACCENT if p["connected"] else (MUTED if p["online"] else LINE)
            cv2.circle(img, (pad + 16, y0 + 18), 4, dot, -1, cv2.LINE_AA)
            texts.append((pad + 28, y0 + 18, p["name"], TEXT, self.f_body, "lm"))
            buttons = []  # (label, fn, primary)
            if p["pair"] == "code":
                status = f"Code  {p['sas'][:3]} {p['sas'][3:]}  ·  same on both screens?"
                buttons = [("Cancel", lambda p=p: self.link.cancel_pair(p["id"]), False)]
                if not p["confirmed"]:
                    buttons.append(("Confirm", lambda p=p: self.link_primary(p), True))
                else:
                    status = f"Code  {p['sas'][:3]} {p['sas'][3:]}  ·  waiting for them"
            elif p["pair"] == "incoming":
                status = "Wants to pair with you"
                buttons = [("Decline", lambda p=p: self.link.cancel_pair(p["id"]), False),
                           ("Accept", lambda p=p: self.link_primary(p), True)]
            elif p["pair"] == "waiting":
                status = "Waiting for them to accept…"
                buttons = [("Cancel", lambda p=p: self.link.cancel_pair(p["id"]), False)]
            elif p["paired"]:
                status = f"Connected · throw to the {p['side']} edge" if p["connected"] else "Paired · offline"
                buttons = [("Unpair", lambda p=p: self.link.unpair(p["id"]), False)]
                if p["connected"]:
                    buttons.append((f"{p['side'].capitalize()} edge", lambda p=p: self.link_cycle_side(p), False))
            else:
                status = "Nearby · push or click Pair"
                buttons = [("Pair", lambda p=p: self.link_primary(p), True)]
            code = p["pair"] == "code"
            texts.append((pad + 28, y0 + 38, status, AMBER if code else MUTED, self.f_btn if code else self.f_small, "lm"))
            bx = W - pad - 10
            for label, fn, primary in reversed(buttons):
                bw = int(self.f_btn.getlength(label)) + 20
                rrect(img, bx - bw, y0 + 8, bx, y0 + 30, 8, ACCENT if primary else SURF2 if not sel else SURF)
                texts.append((bx - bw / 2, y0 + 19, label, BG if primary else TEXT, self.f_btn, "mm"))
                self.link_hits.append((bx - bw, y0 + 8, bx, y0 + 30, fn))
                bx -= bw + 6
        if self.link:
            texts.append((W / 2, H - 16, f"This laptop: {self.link_name} · {self.link_ips}", MUTED, self.f_small, "mm"))

    def make_overlay(self, name):
        """A borderless, click-through, never-focused topmost window (hidden until used)."""
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, np.full((30, 120, 3), BG, np.uint8))
        cv2.waitKey(1)
        hwnd = user32.FindWindowW(None, name)
        if not hwnd:
            return None
        style = user32.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF
        style &= ~(0x00C00000 | 0x00040000 | 0x00080000 | 0x00020000 | 0x00010000)
        user32.SetWindowLongW(hwnd, -16, ctypes.c_long(style).value)
        ex = (user32.GetWindowLongW(hwnd, -20) & 0xFFFFFFFF) | 0x08000000 | 0x8 | 0x80 | 0x80000 | 0x20
        user32.SetWindowLongW(hwnd, -20, ctypes.c_long(ex).value)
        user32.SetLayeredWindowAttributes(hwnd, 0, 240, 2)
        try:
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(ctypes.c_int(2)), 4)
        except Exception:
            pass
        user32.ShowWindow(hwnd, 0)
        return hwnd

    def portal_update(self):
        """While carrying, glowing drop zones show at the edges where linked friends sit."""
        want = {}
        if self.carrying and self.engine.holder is not None and self.link_conn:
            hot = self.portal_hot()
            for pid, name, side in self.link_conn:
                want.setdefault(side, (name, bool(hot and hot[0] == pid)))
        x0, y0, x1, y1 = self.screen_rect()
        for side in SIDES:
            pw = self.portals.get(side)
            if side not in want:
                if pw and pw["shown"]:
                    user32.ShowWindow(pw["hwnd"], 0)
                    pw["shown"] = False
                continue
            if pw is None:
                pw = self.portals[side] = {"hwnd": self.make_overlay(f"{PORTAL} {side}"), "key": None, "shown": False}
            if not pw["hwnd"]:
                continue
            name, hot = want[side]
            key = (name, hot)
            if pw["key"] != key:
                arrow = {"right": "→", "left": "←", "top": "↑"}[side]
                w, h = 210, 64
                img = np.full((h, w, 3), BG, np.uint8)
                rrect(img, 0, 0, w - 1, h - 1, 14, ACCENT if hot else SURF2)
                rrect(img, 0, 0, w - 1, h - 1, 14, ACCENT, 2)
                col = BG if hot else TEXT
                pw["img"] = self.render_text(img, [
                    (w / 2, 24, f"{arrow}  {name}", col, self.f_body, "mm"),
                    (w / 2, 45, "let go to throw" if hot else "carry here to throw", BG if hot else MUTED,
                     self.f_small, "mm")])
                pw["key"] = key
                cv2.imshow(f"{PORTAL} {side}", pw["img"])
            h, w = pw["img"].shape[:2]
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            pos = {"right": (x1 - w - 6, cy - h // 2), "left": (x0 + 6, cy - h // 2), "top": (cx - w // 2, y0 + 6)}[side]
            user32.SetWindowPos(pw["hwnd"], wintypes.HWND(-1), pos[0], pos[1], w, h, 0x0010 | 0x0040)
            pw["shown"] = True

    def chip_setup(self):
        hwnd = self.make_overlay(CHIP)
        if not hwnd:
            return
        self.chip_hwnd = hwnd

    def chip_update(self):
        if not self.chip_hwnd:
            return
        if not (self.carrying and self.engine.holder is not None):
            if self.chip_shown:
                user32.ShowWindow(self.chip_hwnd, 0)
                self.chip_shown = False
            return
        text = self.trim(" ".join((self.held_text or "Copying…").split()), self.f_sug, 260)
        if text != self.chip_key:
            w, h = int(self.f_sug.getlength(text)) + 30, 30
            img = np.full((h, w, 3), BG, np.uint8)
            rrect(img, 0, 0, w - 1, h - 1, 10, SURF)
            rrect(img, 0, 0, w - 1, h - 1, 10, AMBER, 1)
            self.chip_img = self.render_text(img, [(15, h / 2, text, TEXT, self.f_sug, "lm")])
            self.chip_key = text
            cv2.imshow(CHIP, self.chip_img)
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        h, w = self.chip_img.shape[:2]
        user32.SetWindowPos(self.chip_hwnd, wintypes.HWND(-1), pt.x + 18, pt.y + 22, w, h, 0x0010 | 0x0040)
        self.chip_shown = True

    # ---------- main loop ----------
    def toggle_recording(self):
        if self.rec:
            self.rec.close()
            self.rec = None
            self.say("Recording saved to recordings/")
            return
        os.makedirs(RECORDINGS, exist_ok=True)
        self.rec = open(os.path.join(RECORDINGS, time.strftime("%Y-%m-%d_%H%M%S.jsonl")), "w", encoding="utf-8")
        self.say("Recording · Ctrl+Alt+F9 to stop")

    def run(self):
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)  # 1 ms timer resolution: waits and sleeps are precise
        except Exception:
            pass
        tracker = Tracker()
        tracker.start()
        if self.link_enabled:
            self.link_start()
        print("AirTyper running. Click the window you want to type into. Ctrl+Alt+Q to quit.", flush=True)
        last_seq, first = -1, True
        last = time.perf_counter()
        next_draw = 0.0
        try:
            while self.running:
                if tracker.error:
                    raise SystemExit(tracker.error)
                # sleep until a new hand frame arrives (handled at once) or the next input / render tick
                wait = min(next_draw - time.perf_counter(), 1 / INPUT_HZ)
                if wait > 0:
                    tracker.new.wait(wait)
                tracker.new.clear()
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
                if self.rec_toggle:
                    self.rec_toggle = False
                    self.toggle_recording()

                if self.sug_dirty:
                    self.sug_dirty = False
                    self.update_suggestions()
                if seq != last_seq:
                    last_seq = seq
                    self.on_detection(hands, now)
                self.poll_copy(now)
                self.pump_link(now)
                self.input_tick(now, dt)

                # ease the drawn typing cursor toward the filtered target
                k = 1 - math.exp(-dt * EASE)
                for h in self.hands.values():
                    if h.cursor is None:
                        h.disp = None
                    elif h.disp is None:
                        h.disp = h.cursor
                    else:
                        h.disp = (h.disp[0] + (h.cursor[0] - h.disp[0]) * k,
                                  h.disp[1] + (h.cursor[1] - h.disp[1]) * k)
                if now < next_draw:
                    continue
                next_draw = now + 1 / RENDER_FPS

                if self.show_help:
                    layout = "help"
                elif self.ai_mode:
                    layout = "link" if self.show_link else "shelf" if self.show_shelf else "control"
                else:
                    layout = "type"
                if layout != self.layout:
                    self.set_layout(layout)

                try:
                    out = self.draw(frame, now, fps)
                except Exception:  # a drawing glitch must never close the app: log it, keep the last frame
                    import traceback
                    msg = traceback.format_exc()
                    if msg != getattr(self, "_last_draw_error", None):
                        self._last_draw_error = msg
                        print(time.strftime("[%Y-%m-%d %H:%M:%S] draw error\n") + msg, flush=True)
                    if first:
                        raise
                    continue
                if first:
                    self.setup_window(out)
                    first = False
                cv2.imshow(WIN, out)
                self.chip_update()
                self.portal_update()
                cv2.waitKey(1)
        finally:
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
            for tr in self.engine.tracks:  # never leave the mouse button stuck down
                if tr.mouse_down:
                    self.mouse.release(Button.left)
            if self.rec:
                self.rec.close()
            self.link_stop()
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

    kernel32_le = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32_le.CreateMutexW.restype = wintypes.HANDLE
    _mutex = kernel32_le.CreateMutexW(None, False, "Local\\AirTyperSingleInstance")  # noqa: F841 (held for app lifetime)
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
