#!/usr/bin/env python3
"""
KAT - Kite Analysis Terminal
Interactive viewer.

Usage:
    python viewer.py

Controls
    Play / Pause       button, or the space bar
    time slider        drag to any moment of the flight
    Left / Right       step 1 s back / forward
    Shift+Left/Right   step 0.1 s back / forward
    [ and ]            slower / faster playback (0.1x ... 4x); real time is 1x
    airborne camera    radio buttons
    deform             show the deformation panels (extended layout)

    double-click       enlarge the panel under the cursor to the whole view;
                       double-click again or Esc to return to the grid
    1 .. 9             enlarge panel number N directly (row by row)
    F                  full-screen window (matplotlib's own shortcut)
    in an enlarged 3D panel:
        drag           rotate the view
        scroll wheel   zoom in / out
        R              reset the view

Graphs are always composed at the window's own pixel size, so they stay
sharp; only the videos come from the reduced (720p) proxies. The frame is
written straight into the window's pixel buffer instead of going through
imshow, which is what keeps full resolution fast.

viewer_tk.py is an alternative Tk-based window with plainer controls.
"""

import os
import sys
import time

# Windows: declare DPI awareness before any window exists; otherwise, with
# display scaling at 125-150 %, Windows stretches the window as a bitmap
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
for _key, _maps in (("left", ("keymap.back",)), ("right", ("keymap.forward",)),
                    ("r", ("keymap.home",))):          # free keys KAT uses
    for _map in _maps:
        if _key in plt.rcParams[_map]:
            plt.rcParams[_map].remove(_key)
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

import config as cfg
from kat import compose_frame, Slot, View3DMixin

PLAY_FPS = getattr(cfg, "VIEW_FPS", 30.0)   # upper bound on frames drawn per second
SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)    # playback speeds reachable with [ and ]
PROFILE = os.environ.get("KAT_PROFILE", "0") == "1"   # print timings while playing
MAX_W = 2560                                 # cap on the composed frame width
SLIDER_EVERY = 6                             # during playback, redraw the slider every Nth frame


class Viewer:
    def __init__(self):
        self.airborne = cfg.AIRBORNE
        self.with_def = cfg.WITH_DEFORMATION
        self.playing = False
        self._prof = []
        self._tick_n = 0
        self._last = None                    # last composed frame, RGB
        self.slots, self.sources = [], []
        self.focus = None                    # index of the enlarged panel, or None
        self._anchor = None                  # (wall clock, flight time) when playback (re)started
        self._drag = None                    # (x, y) of the last mouse position while rotating
        self._fast_paint = True              # write into the window buffer; see _paint
        self._im = None                      # only used on the fallback path
        self._load()

        w, h = getattr(cfg, "VIEW_SIZE", (1600, 900))
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        self.fig.canvas.manager.set_window_title("KAT - Kite Analysis Terminal")

        # the frame area: no artist is drawn here, the frame is painted into
        # the pixel buffer directly (see _paint)
        self.ax_img = self.fig.add_axes([0.0, 0.10, 1.0, 0.89])
        self.ax_img.axis("off")

        ax_sl = self.fig.add_axes([0.22, 0.045, 0.60, 0.025])
        self.slider = Slider(ax_sl, "t [s]", self.t0, self.t1,
                             valinit=self.t, valstep=0.1)
        self.slider.on_changed(self._on_slider)

        ax_btn = self.fig.add_axes([0.10, 0.035, 0.08, 0.045])
        self.btn = Button(ax_btn, "Play")
        self.btn.on_clicked(self._toggle)

        ax_radio = self.fig.add_axes([0.855, 0.015, 0.06, 0.075])
        ax_radio.set_title("airborne", fontsize=8)
        names = list(cfg.AIRBORNE_VIDEOS)
        self.radio = RadioButtons(ax_radio, names, active=names.index(self.airborne))
        self.radio.on_clicked(self._on_camera)

        ax_chk = self.fig.add_axes([0.925, 0.025, 0.07, 0.055])
        self.chk = CheckButtons(ax_chk, ["deform"], [self.with_def])
        self.chk.on_clicked(self._on_deform)

        # the timer only asks for the next frame; WHICH moment is shown comes from
        # the wall clock (see _tick), so playback speed does not depend on how
        # long a frame takes to draw
        self.speed = float(getattr(cfg, "VIEW_PLAY_SPEED", 1.0))
        self.timer = self.fig.canvas.new_timer(interval=max(1, int(1000 / PLAY_FPS) - 25))
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    # --------------------------- panel setup --------------------------
    def _load(self):
        for src in self.sources:
            src.release()
        for sl in self.slots:                        # close stale matplotlib figures
            if hasattr(sl.panel, "close"):
                sl.panel.close()
        self.slots, grid, self.tel, self.sources = cfg.build(
            with_deformation=self.with_def, airborne=self.airborne)
        self.n_rows, self.n_cols, _ = grid
        self.footer = cfg.make_footer(self.tel)
        self.t0, self.t1 = self.tel.t_start, self.tel.t_end
        if not hasattr(self, "t") or not (self.t0 <= self.t <= self.t1):
            self.t = self.t0
        self._last = None
        self.focus = None

    # ----------------------------- drawing ----------------------------
    def _area(self):
        """Pixel rectangle of the frame area: x0, y0 (from bottom), width, height."""
        bb = self.ax_img.get_window_extent()
        return int(bb.x0), int(bb.y0), int(bb.width), int(bb.height)

    def _target_size(self):
        _, _, w, h = self._area()
        if w > MAX_W:
            h, w = int(h * MAX_W / w), MAX_W
        return max(w, 64), max(h, 64)

    def _layout(self):
        """The grid to draw: all panels, or only the enlarged one."""
        if self.focus is None:
            return self.slots, self.n_rows, self.n_cols
        return [Slot(self.slots[self.focus].panel, 0, 0)], 1, 1

    def _focused_3d(self):
        if self.focus is None:
            return None
        panel = self.slots[self.focus].panel
        return panel if isinstance(panel, View3DMixin) else None

    def _compose(self):
        size = self._target_size()
        slots, n_rows, n_cols = self._layout()
        bgr = compose_frame(slots, n_rows, n_cols, self.t,
                            size, clock_label=False, footer=self.footer)
        if self.focus is not None:                     # usage hint, top-left
            hint = ("drag: rotate   scroll: zoom   R: reset   Esc: back to grid"
                    if self._focused_3d() else "double-click or Esc: back to grid")
            k = max(0.5, size[0] / 1600)
            (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, 1)
            cv2.rectangle(bgr, (8, 8), (8 + tw + 16, 8 + th + 16), (245, 245, 245), -1)
            cv2.putText(bgr, hint, (16, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k,
                        (60, 60, 60), max(1, int(k)), cv2.LINE_AA)
        self._last = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _panel_hit(self, x, y):
        """Which panel is under (x, y), and where inside it, in panel pixels."""
        if self._last is None:
            return None, 0, 0, 0, 0
        x0, y0, w, h = self._area()
        ih, iw = self._last.shape[:2]
        rx = x - (x0 + max(0, (w - iw) // 2))
        ry = (y0 + h - max(0, (h - ih) // 2)) - y
        bar = max(26, int(0.032 * ih))
        if not (0 <= rx < iw and 0 <= ry < ih - bar):
            return None, 0, 0, 0, 0
        if self.focus is not None:                       # one panel fills the view
            return self.focus, rx, ry, iw, ih - bar
        cw, ch = iw // self.n_cols, (ih - bar) // self.n_rows
        col, row = int(rx // cw), int(ry // ch)
        for i, sl in enumerate(self.slots):
            if sl.row <= row < sl.row + sl.rowspan and sl.col <= col < sl.col + sl.colspan:
                return (i, rx - sl.col * cw, ry - sl.row * ch,
                        cw * sl.colspan, ch * sl.rowspan)
        return None, 0, 0, 0, 0

    def _panel_at(self, x, y):
        """Index of the grid panel under display point (x, y), or None."""
        if self._last is None:
            return None
        x0, y0, w, h = self._area()
        ih, iw = self._last.shape[:2]
        rx = x - (x0 + max(0, (w - iw) // 2))
        ry = (y0 + h - max(0, (h - ih) // 2)) - y          # from the top of the frame
        bar = max(26, int(0.032 * ih))
        cw, ch = iw // self.n_cols, (ih - bar) // self.n_rows
        if not (0 <= rx < iw and 0 <= ry < ih - bar):
            return None
        col, row = int(rx // cw), int(ry // ch)
        for i, sl in enumerate(self.slots):
            if sl.row <= row < sl.row + sl.rowspan and sl.col <= col < sl.col + sl.colspan:
                return i
        return None

    def _set_focus(self, index):
        self.focus = index
        self._drag = None
        self._redraw()

    def _canvas_buffer(self):
        """The window's pixel buffer, if this backend exposes one.

        Painting into it is the fast path. Some backends (and some matplotlib
        builds) do not provide it, in which case the viewer falls back to an
        imshow artist: slower, but it works everywhere.
        """
        canvas = self.fig.canvas
        try:
            if hasattr(canvas, "buffer_rgba"):
                return np.asarray(canvas.buffer_rgba())
            renderer = canvas.get_renderer()          # Agg behind another front end
            return np.asarray(renderer.buffer_rgba())
        except Exception:
            return None

    def _paint_via_artist(self):
        """Fallback: hand the frame to matplotlib as an image."""
        if self._im is None:
            self._im = self.ax_img.imshow(self._last, interpolation="nearest")
        else:
            h, w = self._last.shape[:2]
            self._im.set_data(self._last)
            self._im.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
        self.fig.canvas.draw_idle()

    def _paint(self, blit=True):
        """Copy the frame into the canvas pixel buffer, pixel for pixel."""
        if self._last is None or not plt.fignum_exists(self.fig.number):
            return                                    # window already closed
        if not self._fast_paint:
            self._paint_via_artist()
            return
        buf = self._canvas_buffer()
        if buf is None:                               # first time we notice
            self._fast_paint = False
            print(f"[kat] this backend ({type(self.fig.canvas).__name__}, "
                  f"matplotlib {matplotlib.__version__}, python "
                  f"{sys.version.split()[0]}) exposes no pixel buffer;\n"
                  "      falling back to imshow - equivalent, slightly slower. "
                  "The tested\n      combination is Python 3.12 with the pinned "
                  "versions in requirements.txt.")
            self._paint_via_artist()
            return
        H, W = buf.shape[:2]
        x0, y0, w, h = self._area()
        ih, iw = self._last.shape[:2]
        left = x0 + max(0, (w - iw) // 2)
        top = H - (y0 + h) + max(0, (h - ih) // 2)     # buffer rows run top-down

        # The window can be resized between composing a frame and painting it,
        # which puts the target outside the buffer; copy only the overlap.
        # (A negative index would silently address the far edge instead.)
        sy, sx = max(0, -top), max(0, -left)
        dy, dx = max(0, top), max(0, left)
        nh = min(ih - sy, H - dy)
        nw = min(iw - sx, W - dx)
        if nh <= 0 or nw <= 0:
            return
        buf[dy:dy + nh, dx:dx + nw, :3] = self._last[sy:sy + nh, sx:sx + nw]
        buf[dy:dy + nh, dx:dx + nw, 3] = 255
        if blit:
            self.fig.canvas.blit(self.ax_img.bbox)

    def _redraw(self):
        a = time.perf_counter()
        self._compose()
        b = time.perf_counter()
        self._paint()
        if PROFILE:
            now = time.perf_counter()
            self._prof.append(((b - a) * 1000, (now - b) * 1000, now))
            if len(self._prof) >= 60:
                p = self._prof
                span = p[-1][2] - p[0][2]
                fps = (len(p) - 1) / span if span > 0 else 0.0
                w, h = self._last.shape[1], self._last.shape[0]
                print(f"[profile] {w}x{h}  compose {np.mean([x[0] for x in p]):5.1f} ms"
                      f"  paint {np.mean([x[1] for x in p]):4.1f} ms  achieved {fps:5.1f} fps")
                self._prof = []

    def _on_draw(self, _event):
        """After any full redraw (widgets, resize), put the frame back."""
        if not self._fast_paint:                      # the artist is drawn normally
            return
        if self._last is None or self._last.shape[1::-1] != self._target_size():
            self._compose()                           # first draw, or the window changed size
        self._paint(blit=False)                       # the backend blits the whole canvas next

    # ----------------------------- events -----------------------------
    def _set_slider(self, t, draw=True):
        self.slider.eventson = False
        self.slider.drawon = draw
        self.slider.set_val(t)
        self.slider.drawon = True
        self.slider.eventson = True

    def _on_slider(self, val):
        self.t = float(val)
        if self.playing:
            self._restart_clock()                     # continue from where the slider was put
        self._redraw()

    def _play_label(self):
        spd = "" if self.speed == 1.0 else f" {self.speed:g}x"
        return ("Pause" if self.playing else "Play") + spd

    def _restart_clock(self):
        """Tie flight time to the wall clock from this moment on."""
        self._anchor = (time.perf_counter(), self.t)

    def _toggle(self, _):
        self.playing = not self.playing
        self.btn.label.set_text(self._play_label())
        if self.playing:
            self._restart_clock()
            self.timer.start()
        else:
            self.timer.stop()
            self._set_slider(self.t)                  # make sure the thumb is exact
        self.fig.canvas.draw_idle()

    def _tick(self):
        if not self.playing:
            return
        if not plt.fignum_exists(self.fig.number):     # window closed under us
            self._on_close()
            return
        wall0, t0 = self._anchor
        t = t0 + (time.perf_counter() - wall0) * self.speed
        if t > self.t1:                               # loop back to the start
            t = self.t0
            self.t = t
            self._restart_clock()
        self.t = t
        self._redraw()
        self._tick_n += 1
        # moving the slider forces a full redraw; doing it every frame would
        # halve the frame rate, so the thumb is updated a few times a second
        self._set_slider(self.t, draw=(self._tick_n % SLIDER_EVERY == 0))

    def _change_speed(self, direction):
        i = min(range(len(SPEEDS)), key=lambda k: abs(SPEEDS[k] - self.speed))
        self.speed = SPEEDS[min(max(i + direction, 0), len(SPEEDS) - 1)]
        if self.playing:
            self._restart_clock()                     # keep the current moment
        self.btn.label.set_text(self._play_label())
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        steps = {"left": -1.0, "right": 1.0, "shift+left": -0.1, "shift+right": 0.1}
        if event.key == " ":
            self._toggle(None)
        elif event.key == "]":
            self._change_speed(+1)
        elif event.key == "[":
            self._change_speed(-1)
        elif event.key == "escape" and self.focus is not None:
            self._set_focus(None)
        elif event.key in [str(i) for i in range(1, 10)]:
            i = int(event.key) - 1
            if i < len(self.slots):
                self._set_focus(None if self.focus == i else i)
        elif event.key == "r" and self._focused_3d():
            self._focused_3d().set_view(reset=True)
            self._redraw()
        elif event.key in steps:
            self.t = min(max(self.t + steps[event.key], self.t0), self.t1)
            if self.playing:
                self._restart_clock()
            self._redraw()
            self._set_slider(self.t)

    def _on_press(self, event):
        if event.inaxes is not self.ax_img or event.button != 1:
            return
        if event.dblclick:
            if self.focus is not None:
                self._set_focus(None)
            else:
                i = self._panel_at(event.x, event.y)
                if i is not None:
                    self._set_focus(i)
        else:
            i, px, py, pw, ph = self._panel_hit(event.x, event.y)
            panel = self.slots[i].panel if i is not None else None
            if panel is not None and hasattr(panel, "click") and panel.click(px, py, pw, ph):
                self._redraw()                            # a drop-down took the click
            elif self._focused_3d():
                self._drag = (event.x, event.y)

    def _on_release(self, _event):
        self._drag = None

    def _on_motion(self, event):
        panel = self._focused_3d()
        if self._drag is None or panel is None or event.x is None:
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        self._drag = (event.x, event.y)
        panel.set_view(d_elev=-0.4 * dy, d_azim=-0.4 * dx)
        self._redraw()

    def _on_scroll(self, event):
        panel = self._focused_3d()
        if panel is None or event.inaxes is not self.ax_img:
            return
        panel.set_view(zoom_factor=0.85 if event.button == "up" else 1 / 0.85)
        self._redraw()

    def _on_close(self, _event=None):
        """Stop the clock and let the videos go when the window closes.

        Without this the timer fires once more after the window is gone and
        tries to paint into a canvas that no longer exists.
        """
        self.playing = False
        self._last = None
        try:
            self.timer.stop()
        except Exception:
            pass
        for src in getattr(self, "sources", []):
            src.release()

    def _on_camera(self, label):
        if label == self.airborne:
            return
        old = self.airborne
        self.airborne = label
        try:
            self._load()
        except Exception as e:
            print(f"could not switch camera ({label}): {e}")
            self.airborne = old
            self._load()
        if self.playing:
            self._restart_clock()
        self.fig.canvas.draw_idle()

    def _on_deform(self, _):
        if getattr(self, "_silent_check", False):
            return
        want = not self.with_def
        if want:
            missing = [m for m in cfg.check_inputs(with_deformation=True)
                       if "photogrammetry" in m[1]]
            if missing:
                print(f"[kat] the deformation panels need {missing[0][0]}, which is "
                      "not there.\n      Build a stand-in from the static point cloud:\n"
                      "          python kat_dataset.py --flight-csv <flight log> "
                      "--flight-start-s <second>\n"
                      "      or point DATASET_CSV in the flight file at the real dataset.")
                self._silent_check = True             # put the tick back without recursing
                self.chk.set_active(0)
                self._silent_check = False
                return
        self.with_def = want
        self._load()
        if self.playing:
            self._restart_clock()        # the time range may have changed
        self.slider.valmin, self.slider.valmax = self.t0, self.t1
        self.slider.ax.set_xlim(self.t0, self.t1)
        self._set_slider(self.t)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    cfg.require_inputs()
    matplotlib.use("TkAgg", force=True)
    Viewer().show()
