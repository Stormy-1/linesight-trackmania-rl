import threading

from trackmania_rl.tmi_interaction.boot_keys import ENTER, TILDE, BootKeyPresser


class FakeDesktop:
    """Records presses; ``foreground`` decides which window has focus."""

    def __init__(self, foreground=True):
        self.foreground = foreground
        self.pressed = []
        self.focus_requests = 0

    def press(self, scancode):
        self.pressed.append(scancode)

    def focus(self, hwnd):
        self.focus_requests += 1

    def is_foreground(self, hwnd):
        return self.foreground


def make(desktop, **kwargs):
    return BootKeyPresser(
        press=desktop.press,
        focus=desktop.focus,
        is_foreground=desktop.is_foreground,
        sleep=lambda s: None,
        **kwargs,
    )


def test_profile_phase_closes_console_then_picks_profile():
    desktop = FakeDesktop()
    make(desktop).run_profile_phase(hwnd=1)
    assert desktop.pressed == [TILDE, ENTER]


def test_never_presses_into_another_window():
    desktop = FakeDesktop(foreground=False)
    presser = make(desktop, focus_attempts=3)
    presser.run_profile_phase(hwnd=1)
    presser.run_intro_phase(hwnd=1, race_started=threading.Event(), max_presses=5)
    assert desktop.pressed == []
    assert desktop.focus_requests > 0


def test_intro_phase_stops_once_the_race_runs():
    desktop = FakeDesktop()
    started = threading.Event()

    def press(scancode):
        desktop.pressed.append(scancode)
        if len(desktop.pressed) == 2:
            started.set()

    presser = BootKeyPresser(press=press, focus=desktop.focus, is_foreground=desktop.is_foreground, sleep=lambda s: None)
    presser.run_intro_phase(hwnd=1, race_started=started, max_presses=10)
    assert desktop.pressed == [ENTER, ENTER]


def test_intro_phase_gives_up_after_max_presses():
    desktop = FakeDesktop()
    make(desktop).run_intro_phase(hwnd=1, race_started=threading.Event(), max_presses=4)
    assert desktop.pressed == [ENTER] * 4
