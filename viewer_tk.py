#!/usr/bin/env python3
"""
KAT - Kite Analysis Terminal
Interactive viewer - Tk version (alternative).

Sharp and fast at the same time, with plainer controls. The default viewer
is viewer.py (matplotlib).

Usage:
    python viewer_tk.py

Controls
    Play / Pause       button, or the space bar
    time slider        drag to any moment of the flight
    Left / Right       step 1 s back / forward
    Shift+Left/Right   step 0.1 s back / forward
    airborne camera    radio buttons
    deform             show the deformation panels (extended layout)

Every frame is composed at the window's own pixel size and copied to the
screen through Pillow, so graphs stay sharp at any window size. Panels are
built through config.build(), exactly as for render.py.

viewer.py is the matplotlib-based default viewer.
"""

import os
import sys
import time

# Windows: declare DPI awareness BEFORE any window exists, otherwise Windows
# scales the whole window as a bitmap on high-DPI screens and everything blurs
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

import config as cfg
from kat import compose_frame

PLAY_FPS = getattr(cfg, "VIEW_FPS", 24.0)
STEP_S = 1.0 / PLAY_FPS
PROFILE = os.environ.get("KAT_PROFILE", "0") == "1"
MAX_W = 2560                 # upper bound on the composed frame width


# =====================================================================
# Core: everything that is not GUI. Testable without a display.
# =====================================================================

class ViewerCore:
    """Holds the panels, the time range and the current moment."""

    def __init__(self, airborne=None, with_deformation=None):
        self.airborne = airborne or cfg.AIRBORNE
        self.with_def = cfg.WITH_DEFORMATION if with_deformation is None else with_deformation
        self.slots = []
        self.sources = []
        self.t = None
        self.load()

    def release(self):
        for src in self.sources:
            src.release()
        for sl in self.slots:
            if hasattr(sl.panel, "close"):
                sl.panel.close()
        self.slots, self.sources = [], []

    def load(self):
        """(Re)build every panel, e.g. after switching camera or layout."""
        self.release()
        self.slots, grid, self.tel, self.sources = cfg.build(
            with_deformation=self.with_def, airborne=self.airborne)
        self.n_rows, self.n_cols, _ = grid
        self.footer = cfg.make_footer(self.tel)
        self.t0, self.t1 = self.tel.t_start, self.tel.t_end
        if self.t is None or not (self.t0 <= self.t <= self.t1):
            self.t = self.t0

    def clamp(self, t):
        return min(max(t, self.t0), self.t1)

    def frame_rgb(self, size):
        """The current moment composed at `size`, as an RGB array."""
        bgr = compose_frame(self.slots, self.n_rows, self.n_cols, self.t,
                            size, clock_label=False, footer=self.footer)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# =====================================================================
# Tk front end
# =====================================================================

def run():
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    core = ViewerCore()

    root = tk.Tk()
    root.title("KAT - Kite Analysis Terminal")
    vw, vh = getattr(cfg, "VIEW_SIZE", (1280, 720))
    root.geometry(f"{vw}x{vh + 60}")
    root.configure(bg="#202020")

    canvas = tk.Canvas(root, bg="#202020", highlightthickness=0)
    canvas.pack(side="top", fill="both", expand=True)
    bar = tk.Frame(root, bg="#2b2b2b")
    bar.pack(side="bottom", fill="x")

    state = {"playing": False, "photo": None, "item": None, "size": None,
             "setting_scale": False, "after": None, "resize_after": None,
             "prof": []}

    # ------------------------------------------------------------ drawing
    def target_size():
        w = max(canvas.winfo_width(), 64)
        h = max(canvas.winfo_height(), 64)
        if w > MAX_W:                                  # cap cost, keep aspect
            h = int(h * MAX_W / w)
            w = MAX_W
        return (w // 2 * 2, h // 2 * 2)

    def redraw():
        t_start = time.perf_counter()
        size = target_size()
        rgb = core.frame_rgb(size)
        t_mid = time.perf_counter()
        img = Image.fromarray(rgb)
        if state["photo"] is None or state["size"] != size:
            state["photo"] = ImageTk.PhotoImage(img)
            state["size"] = size
            if state["item"] is None:
                state["item"] = canvas.create_image(0, 0, anchor="nw", image=state["photo"])
            else:
                canvas.itemconfigure(state["item"], image=state["photo"])
        else:
            state["photo"].paste(img)                  # fast in-place copy
        cx = (canvas.winfo_width() - size[0]) // 2
        cy = (canvas.winfo_height() - size[1]) // 2
        canvas.coords(state["item"], max(cx, 0), max(cy, 0))
        if PROFILE:
            now = time.perf_counter()
            state["prof"].append(((t_mid - t_start) * 1000, (now - t_mid) * 1000, now))
            if len(state["prof"]) >= 60:
                p = state["prof"]
                span = p[-1][2] - p[0][2]
                fps = (len(p) - 1) / span if span > 0 else 0.0
                print(f"[profile] {size[0]}x{size[1]}  compose {np.mean([x[0] for x in p]):5.1f} ms"
                      f"  display {np.mean([x[1] for x in p]):4.1f} ms  achieved {fps:5.1f} fps")
                state["prof"] = []

    def set_time(t, from_scale=False):
        core.t = core.clamp(t)
        if not from_scale:
            state["setting_scale"] = True
            scale.set(core.t)
            state["setting_scale"] = False
        time_lbl.configure(text=f"t = {core.t:7.1f} s")
        redraw()

    # ----------------------------------------------------------- playback
    def tick():
        if not state["playing"]:
            return
        t0 = time.perf_counter()
        # real-time playback: advance by the elapsed wall-clock time
        dt = min(t0 - state.get("last_tick", t0 - STEP_S), 0.25)
        state["last_tick"] = t0
        t = core.t + dt
        set_time(core.t0 if t > core.t1 else t)
        spent = (time.perf_counter() - t0) * 1000
        state["after"] = root.after(max(1, int(1000 / PLAY_FPS - spent)), tick)

    def toggle(_event=None):
        state["playing"] = not state["playing"]
        play_btn.configure(text="Pause" if state["playing"] else "Play")
        if state["playing"]:
            state["last_tick"] = time.perf_counter() - STEP_S
            tick()
        elif state["after"] is not None:
            root.after_cancel(state["after"])
            state["after"] = None

    def step(ds):
        def handler(_event=None):
            set_time(core.t + ds)
        return handler

    # ------------------------------------------------------------ widgets
    # controls never take keyboard focus: otherwise space would press a focused
    # button AND trigger the shortcut, and arrows would move a focused slider twice
    play_btn = tk.Button(bar, text="Play", width=7, command=toggle, takefocus=0)
    play_btn.pack(side="left", padx=8, pady=8)

    def on_scale(value):
        if not state["setting_scale"]:
            set_time(float(value), from_scale=True)

    scale = tk.Scale(bar, from_=core.t0, to=core.t1, orient="horizontal",
                     resolution=0.1, showvalue=False, command=on_scale,
                     bg="#2b2b2b", troughcolor="#444", highlightthickness=0,
                     sliderlength=18, takefocus=0)
    scale.bind("<ButtonRelease-1>", lambda _e: canvas.focus_set())
    scale.pack(side="left", fill="x", expand=True, padx=8)

    time_lbl = tk.Label(bar, text="", fg="#ddd", bg="#2b2b2b", width=12,
                        font=("Consolas", 10))
    time_lbl.pack(side="left", padx=4)

    cam_var = tk.StringVar(value=core.airborne)

    def on_camera():
        new = cam_var.get()
        if new == core.airborne:
            return
        old = core.airborne
        core.airborne = new
        try:
            core.load()
        except Exception as e:
            print(f"could not switch camera ({new}): {e}")
            core.airborne = old
            cam_var.set(old)
            core.load()
        state["size"] = None
        set_time(core.t)

    for name in cfg.AIRBORNE_VIDEOS:
        tk.Radiobutton(bar, text=name, value=name, variable=cam_var, command=on_camera,
                       fg="#ddd", bg="#2b2b2b", selectcolor="#444", takefocus=0,
                       activebackground="#2b2b2b").pack(side="left")

    def_var = tk.BooleanVar(value=core.with_def)

    def on_deform():
        if def_var.get():
            missing = [m for m in cfg.check_inputs(with_deformation=True)
                       if "photogrammetry" in m[1]]
            if missing:
                print(f"[kat] the deformation panels need {missing[0][0]}, which is "
                      "not there.\n      Build a stand-in from the static point cloud:\n"
                      "          python kat_dataset.py --flight-csv <flight log> "
                      "--flight-start-s <second>")
                def_var.set(False)
                return
        core.with_def = def_var.get()
        core.load()
        scale.configure(from_=core.t0, to=core.t1)
        state["size"] = None
        set_time(core.t)

    tk.Checkbutton(bar, text="deform", variable=def_var, command=on_deform,
                   fg="#ddd", bg="#2b2b2b", selectcolor="#444", takefocus=0,
                   activebackground="#2b2b2b").pack(side="left", padx=8)

    # --------------------------------------------------------- bindings
    def on_resize(_event):
        if state["resize_after"] is not None:
            root.after_cancel(state["resize_after"])
        # panels rebuild on a size change, so wait until the drag settles
        state["resize_after"] = root.after(200, lambda: set_time(core.t))

    canvas.bind("<Configure>", on_resize)
    root.bind("<space>", toggle)
    root.bind("<Left>", step(-1.0))
    root.bind("<Right>", step(1.0))
    root.bind("<Shift-Left>", step(-0.1))
    root.bind("<Shift-Right>", step(0.1))

    def on_close():
        state["playing"] = False
        core.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    canvas.focus_set()
    root.after(100, lambda: set_time(core.t))
    root.mainloop()


if __name__ == "__main__":
    cfg.require_inputs()
    run()
