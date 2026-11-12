"""Press the keys a freshly launched TmForever waits for, so batches run unattended.

Measured on 2026-11-10 (screenshots of a launch with no input): the game opens
with the TMInterface console on top of "Select your Profile". It then plays a map
intro after the first map loads. A person pressed ``~`` (close the console),
Enter (take the highlighted profile), and Enter (skip the intro). Without them,
TMInterface never sends a step and the rollout dies with ``WinError 10060``.

Safety: every key goes through ``SendInput`` as a hardware scan code, and only
after checking that the TrackMania window is the foreground window. If focus
can't be taken, the key is skipped, so a press can never land in another app.
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable

TILDE = 0x29  # the grave/tilde key, which toggles the TMInterface console
ENTER = 0x1C
_KEYEVENTF_SCANCODE = 0x0008
_KEYEVENTF_KEYUP = 0x0002
_INPUT_KEYBOARD = 1

PROFILE_SCREEN_DELAY_S = 4.0  # the console and profile screen take a few seconds to appear
BETWEEN_KEYS_S = 1.0
INTRO_PRESS_INTERVAL_S = 2.0
INTRO_MAX_PRESSES = 20


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _InputUnion(ctypes.Union):
    # MOUSEINPUT is the largest member; padding keeps the struct the size Windows expects.
    _fields_ = [("ki", _KeyboardInput), ("padding", ctypes.c_byte * 32)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _InputUnion)]


def press_scancode(scancode: int) -> None:
    """Press and release one key as a hardware scan code (what DirectInput games read)."""

    for flags in (_KEYEVENTF_SCANCODE, _KEYEVENTF_SCANCODE | _KEYEVENTF_KEYUP):
        event = _Input(type=_INPUT_KEYBOARD)
        event.union.ki = _KeyboardInput(0, scancode, flags, 0, None)
        if ctypes.windll.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(event)) != 1:
            raise OSError(f"SendInput failed for scancode {scancode:#x}")
        time.sleep(0.05)


def is_foreground_window(hwnd: int) -> bool:
    return ctypes.windll.user32.GetForegroundWindow() == hwnd


class BootKeyPresser:
    """The boot key sequence, with all side effects injected for testing."""

    def __init__(
        self,
        *,
        press: Callable[[int], None],
        focus: Callable[[int], None],
        is_foreground: Callable[[int], bool],
        sleep: Callable[[float], None] = time.sleep,
        focus_attempts: int = 3,
    ):
        self._press = press
        self._focus = focus
        self._is_foreground = is_foreground
        self._sleep = sleep
        self._focus_attempts = focus_attempts

    def _press_into(self, hwnd: int, scancode: int) -> bool:
        for _ in range(self._focus_attempts):
            if self._is_foreground(hwnd):
                self._press(scancode)
                return True
            try:
                self._focus(hwnd)
            except Exception as error:  # noqa: BLE001 - pywin32/COM errors share no narrower base; logged, retried, then the key is skipped
                print(f"[boot-keys] focus failed: {error}")
            self._sleep(0.3)
        print(f"[boot-keys] skipped key {scancode:#x}: TrackMania is not the foreground window")
        return False

    def run_profile_phase(self, hwnd: int) -> None:
        """Close the TMInterface console, then take the highlighted profile."""

        self._sleep(PROFILE_SCREEN_DELAY_S)
        self._press_into(hwnd, TILDE)
        self._sleep(BETWEEN_KEYS_S)
        self._press_into(hwnd, ENTER)

    def run_intro_phase(self, hwnd: int, race_started: threading.Event, max_presses: int = INTRO_MAX_PRESSES) -> None:
        """Press Enter until the race runs, which skips the map intro."""

        for _ in range(max_presses):
            if race_started.is_set():
                return
            self._press_into(hwnd, ENTER)
            self._sleep(INTRO_PRESS_INTERVAL_S)


def start_in_background(target: Callable[..., None], *args) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread
