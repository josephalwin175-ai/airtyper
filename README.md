# AirTyper

Type in the air with your webcam, and control your PC with hand gestures: swipe tabs, catch and splash text to copy and paste, scroll, point and click. AirTyper runs as a small always-on-top HUD that never takes focus, so everything it types goes into the window you last clicked.

<p>
  <img src="assets/screenshot-typing.png" alt="AirTyper typing with two hands" width="380">
  <img src="assets/screenshot-ai-mode.png" alt="AirTyper AI gesture mode" width="380">
</p>

Built with [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), OpenCV and pynput. Windows only.

## Install

Requires Windows 10/11, a webcam, and [Python 3.10](https://www.python.org/downloads/) (with the `py` launcher).

```bat
git clone https://github.com/josephalwin175-ai/airtyper.git
cd airtyper
setup.bat
```

`setup.bat` creates a virtual environment, installs the dependencies, downloads the hand-tracking model, and adds **AirTyper** shortcuts to your Desktop and Start menu.

## Start it

- Double-click **AirTyper** on your Desktop or search for it in the Start menu.
- Problems? Run `run-debug.bat` to see messages, or check `airtyper.log`.

## Typing

1. Click the window you want to type into (Notepad, browser, chat...).
2. Raise your hand. The cursor starts in the middle of the keyboard, wherever your hand is.
3. Move your hand a little to steer. Pinch thumb + index to press the highlighted key.

- **1H / 2H** button: one-hand or two-hand typing. In 2H, your left hand covers the left half of the keyboard and your right hand the right half.
- **Suggestions**: pinch a word above the keyboard to finish it.
- **Hybrid**: your real keyboard works at the same time. Hand typing pauses for about a second after real keystrokes, so resting hands can't press keys.
- **MODE key**: switch between pinch and dwell (hover 0.7 s to press).

## Gesture mode

Turn it on with the **Gestures** button, `Ctrl+Alt+A`, or by holding the rock gesture (index + pinky). Press `Ctrl+Alt+G` (or **?**) for the built-in guide.

### Motions (instant)

| Gesture | Action |
|---|---|
| Swipe an open hand left / right | Next / previous tab (`Ctrl+Tab`; `Ctrl+PgDn/PgUp` in VS Code and Antigravity) |
| Swipe up | Task view |
| Two-finger swipe left / right | Switch app (`Alt+Tab`) |
| Push your open hand toward the camera | Play / pause |
| Pinch (OK sign) and move up / down | Scroll the window under the mouse |
| Pinch and twist like a knob | Volume up / down |
| Pinch with both hands, pull apart / together | Zoom in / out |

Moving your hand back to the centre after a swipe is ignored, so it doesn't undo the swipe.

### Select, catch, carry, drop

**Select** with the air pointer (hold one finger up until the bar fills):

| Gesture | Does |
|---|---|
| Pinch | click: place the caret |
| Pinch twice / three times | select the word / line (real double- and triple-clicks) |
| Pinch, hold and drag | select a range (the pointer slows down while dragging and when your hand moves slowly) |
| Point at a word, catch with the other hand | grabs that word if nothing is selected |

**Catch** it: open hand, then a fast fist. AirTyper copies the selection and shows it in the card.

**Carry** it: move the fist and the mouse cursor follows, with the text in a small chip next to it. Your other hand stays free to swipe tabs or switch apps.

**Drop** it: flick the fist open. If you carried it, AirTyper clicks there and pastes. If you didn't move it, it pastes at the current caret. Opening slowly cancels. (If your other hand is pointing, that hand steers and the fist only holds.)

**Clipboard shelf**: every catch also lands on a shelf of the last 8 items (kept in memory only, cleared when AirTyper quits). Open it with a two-finger swipe down, the **Clips** button or `Ctrl+Alt+V`. Move your hand up or down to highlight an item, then catch to take it and carry it anywhere. For everything else you have copied, use Windows clipboard history (`Win+V`).

Terminals get `Ctrl+Shift+C` / `Ctrl+Shift+V`, so a catch never sends Ctrl+C to a running program. Set `"hold_only_clipboard": true` in `settings.json` to restore your previous clipboard when you cancel.

### Hold still (until the bar fills)

| Gesture | Fingers up | Action |
|---|---|---|
| One finger | index | Air pointer: moves the mouse, pinch to click or drag, open hand to stop |
| Two fingers | index, middle | Chrome |
| Three fingers | index, middle, ring | Claude |
| Four fingers | all but thumb | Antigravity |
| Thumbs up / down | thumb | File Explorer / Undo |
| Call me | thumb, pinky | Spotify |
| L shape | thumb, index | Screenshot (Win+Shift+S) |
| Pinky only | pinky | Close the active app (longer hold) |
| Rock | index, pinky | Gesture mode on/off (works while typing) |

### gestures.json

Three sections: `poses` (held shapes), `motions` (`swipe_left`, `swipe_right`, `swipe_up`, `swipe_down`, `two_swipe_left`, `two_swipe_right`, `push`, `zoom_in`, `zoom_out`), and `per_app` (overrides keyed by process name, such as `"chrome.exe"`). Each entry has a `name` and either `run` (a program path, a URL, or a `shell:AppsFolder\...` ID) or `keys` (for example `"ctrl+shift+t"`, `"media_next"`). Restart AirTyper after editing. Old single-level files are converted automatically.

## Window and hotkeys

Drag the header to move the window; it keeps its bottom edge when it switches between the typing and gesture layouts. The **×** button closes AirTyper.

| Ctrl+Alt + | |
|---|---|
| Up / Down | more / less sensitive |
| R | re-centre the cursor on your hand |
| M | pinch / dwell |
| D | one / two hands |
| A | gesture mode |
| V | clipboard shelf |
| G | gesture guide |
| H | hide / show |
| F9 | start / stop recording hand data |
| Q | quit |

## Tuning

Typing feel: the constants at the top of `airtyper.py` (`SENS_LEVELS`, `PINCH_ON` / `PINCH_OFF`, `EASE`). Gestures: the constants at the top of `gesture_engine.py` (swipe distance, catch and splash speed, hold times, scroll and volume steps).

To tune with real data, record with `Ctrl+Alt+F9`, then replay it through the engine:

```bat
.venv\Scripts\python.exe replay.py recordings\<file>.jsonl
.venv\Scripts\python.exe replay.py --selftest
```

## Files

| File | Purpose |
|---|---|
| `airtyper.py` | the app |
| `gesture_engine.py` | gesture recognition (no Windows calls, replayable) |
| `replay.py` | replay recordings / self-test the engine |
| `gestures.json` | gesture → action mapping |
| `settings.json` | saved window position, sensitivity, mode |
| `assets/airtyper.ico` | app icon (regenerate with `make_icon.py`) |
| `models/hand_landmarker.task` | MediaPipe hand-tracking model |
| `setup.bat`, `create_shortcuts.ps1` | install and shortcut creation |
| `run-debug.bat` | run with a console for troubleshooting |
| `airtyper-concept.html` | original design concept |

## License

MIT. See [LICENSE](LICENSE).
