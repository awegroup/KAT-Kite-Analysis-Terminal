#!/usr/bin/env python3
"""
KAT - Kite Analysis Terminal
Core engine: sources, panels, grid.

Architecture (three layers):
  1. SOURCES (Source) : answer "what is your data at time t?".
                        Video, telemetry, point cloud - each knows its own
                        time mapping and nothing about the others.
  2. PANELS (Panel)   : answer "what do you look like at time t?" -> BGR image.
  3. GRID             : places the panels and writes the mp4.

Adding a new panel = adding one line to the slot list in config.py.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional

import cv2
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg


# ===================================================================
# 1. SOURCES
# ===================================================================

class VideoSource:
    """A video file plus its offset relative to the master clock.

    master_t = master clock [s]. Its position in this video: t0_s + master_t
    Every camera carries its own t0_s and its own fps, which is why several
    cameras can run side by side without interfering.
    """

    def __init__(self, path: str, t0_s: float = 0.0, rotate_180: bool = False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        self.path = path
        self.t0_s = t0_s
        self.rotate_180 = rotate_180
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cur = -1          # index of the currently decoded frame
        self._last = None       # the image of that frame
        self._cache = {}        # frame index -> image, for backward scrubbing
        self._cache_max = 240

    def frame_at(self, master_t: float):
        """Forward playback skips frames instead of seeking - much faster."""
        target = int(round((self.t0_s + master_t) * self.fps))
        if target < 0 or target >= self.n_frames:
            return None
        if target == self._cur:
            return self._last
        hit = self._cache.get(target)
        if hit is not None:
            return hit

        # stepping forward frame by frame beats a seek only for short jumps;
        # with the recommended proxies (-g 10 -bf 0) a seek costs ~20 ms, about
        # as much as decoding a dozen frames in sequence
        if target < self._cur or target > self._cur + 12:
            # going backwards or jumping far ahead: a seek is unavoidable
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self._cur = target - 1
        while self._cur < target - 1:
            if not self.cap.grab():
                return self._last
            self._cur += 1

        ok, frame = self.cap.read()
        if not ok:
            return self._last
        self._cur = target
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        self._last = frame
        if len(self._cache) >= self._cache_max:
            self._cache.clear()
        self._cache[target] = frame
        return frame

    def release(self):
        self.cap.release()


def repair_time_axis(df: pd.DataFrame, col: str = "time") -> np.ndarray:
    """Spread repeated timestamps evenly inside their second.

    In the flight log 'time' holds whole seconds with ~10 rows per second.
    Without this repair interpolation fails, and dropping the duplicates
    would throw away 90% of the data.
    """
    t = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    if len(t) < 2 or np.median(np.diff(t)) > 0:
        return t
    g = pd.Series(t)
    idx = g.groupby(g).cumcount().to_numpy(float)
    n = g.groupby(g).transform("size").to_numpy(float)
    return t + idx / np.maximum(n, 1.0)


class TelemetrySource:
    """Time-series telemetry. Intermediate values come from linear interpolation."""

    def __init__(self, df: pd.DataFrame, time_col: str = "timestamp"):
        df = df.copy()
        t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)

        # Rows sharing a timestamp can mean two different things:
        #  (a) flight log: genuine successive samples, values differ -> spread in time
        #  (b) complete dataset: points of one frame, telemetry identical -> deduplicate
        if len(t_raw) > 1 and np.median(np.diff(t_raw)) == 0:
            probe = next((c for c in df.columns
                          if c not in (time_col, "x", "y", "z", "marker_id",
                                       "frame_idx", "utc")
                          and pd.api.types.is_numeric_dtype(df[c])), None)
            duplicated_rows = probe is not None and \
                df.groupby(time_col)[probe].transform("nunique").max() <= 1
            if duplicated_rows:
                df = df.drop_duplicates(subset=time_col)
                t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
                df["_t"] = t_raw
            else:
                df["_t"] = repair_time_axis(df, time_col)
        else:
            df["_t"] = t_raw

        self.df = df.sort_values("_t")
        self.t = self.df["_t"].to_numpy(float)
        self._unwrapped = {}          # angle columns, with the +-180 jumps removed

    def series(self, col: str):
        """The full history - used to draw the static background once."""
        if col not in self.df.columns:
            return None, None
        return self.t, pd.to_numeric(self.df[col], errors="coerce").to_numpy(float)

    def angle_at(self, col: str, master_t: float) -> float:
        """Interpolate an angle column [deg] the short way round.

        Straight interpolation between +179 and -179 would sweep almost a
        full turn and flip the kite for a frame; unwrapping the series first
        keeps the step at two degrees.
        """
        if col not in self._unwrapped:
            t, v = self.series(col)
            if t is None:
                self._unwrapped[col] = None
            else:
                v = pd.Series(v).ffill().bfill().to_numpy(float)
                self._unwrapped[col] = np.degrees(np.unwrap(np.radians(v)))
        u = self._unwrapped[col]
        if u is None:
            return np.nan
        return float(np.interp(master_t, self.t, u))

    def value_at(self, col: str, master_t: float) -> float:
        t, v = self.series(col)
        if t is None:
            return np.nan
        return float(np.interp(master_t, t, v))

    @property
    def t_start(self):
        return float(self.t[0])

    @property
    def t_end(self):
        return float(self.t[-1])


class PointCloudSource:
    """Photogrammetry point cloud: 3D points, frame by frame."""

    def __init__(self, df: pd.DataFrame, time_col: str = "timestamp",
                 max_gap: float = 0.5):
        self.df = df
        self.times = np.sort(df[time_col].unique().astype(float))
        self.time_col = time_col
        # outside the covered interval there is nothing to show, rather than a
        # frozen shape: photogrammetry usually covers only part of a flight
        self.max_gap = float(max_gap)
        self._groups = {t: g for t, g in df.groupby(time_col)}

    def points_at(self, master_t: float):
        """Returns the nearest frame; points are never interpolated."""
        if len(self.times) == 0:
            return None
        i = int(np.argmin(np.abs(self.times - master_t)))
        if abs(self.times[i] - master_t) > self.max_gap:
            return None
        return self._groups[self.times[i]]


# ===================================================================
# 2. PANELS
# ===================================================================

def _new_fig(w: int, h: int, dpi: int = 100):
    """A figure that is never registered with the window manager.

    With plt.figure() an interactive backend such as TkAgg would open these
    figures as separate windows. Figure + FigureCanvasAgg draws purely in
    memory, so no window is created.
    """
    # the small epsilon stops e.g. 464/100*100 = 463.999... being truncated to 463 px
    fig = Figure(figsize=((w + 1e-3) / dpi, (h + 1e-3) / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


def utc_string(epoch_unix: Optional[float], t: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Absolute UTC time for master clock t. epoch_unix is the Unix time at t = 0."""
    if epoch_unix is None or not np.isfinite(t):
        return ""
    return datetime.fromtimestamp(epoch_unix + t, tz=timezone.utc).strftime(fmt)


def _nice_step(window_s: float) -> float:
    """A readable tick spacing, roughly six ticks across the visible window."""
    for step in (1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600):
        if window_s / step <= 7:
            return float(step)
    return 1200.0


REF_PANEL_W = 480.0     # panel width the font sizes below were tuned for


def _fs(w: float, base: float) -> float:
    """Font size scaled to the panel width.

    A fixed point size makes labels unreadable on a large display, so every
    panel grows its text in proportion to its own width.
    """
    return base * max(0.75, min(2.4, w / REF_PANEL_W))


def _fig_to_bgr(fig) -> np.ndarray:
    """Matplotlib figure -> OpenCV BGR array (works with recent matplotlib)."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)


def _label(img, text, org=None, scale=0.6):
    if not text:
        return img
    h = img.shape[0]
    org = org or (10, h - 12)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, th), base = cv2.getTextSize(text, font, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + w + 6, y + base + 2), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _tag(img, text: str, corner: str = "bl"):
    """Small, semi-transparent caption in a corner of a video panel."""
    if not text:
        return img
    h, w = img.shape[:2]
    k = max(0.45, min(1.6, h / 420.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5 * k, max(1, int(round(1.1 * k)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad, m = int(6 * k), int(10 * k)
    x0 = m if corner.endswith("l") else w - m - tw - 2 * pad
    y1 = h - m if corner.startswith("b") else m + th + base + 2 * pad
    y0 = y1 - th - base - 2 * pad
    x1 = x0 + tw + 2 * pad
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = (roi * 0.45).astype(np.uint8)          # darken, 55 % opaque
    cv2.putText(img, text, (x0 + pad, y1 - pad - base), font, scale,
                (235, 235, 235), thick, cv2.LINE_AA)
    return img


class VideoPanel:
    """A single camera stream, fitted to its grid cell.

    fit = "cover"   : fill the whole cell, cropping the overflow evenly
                      (every camera looks the same, whatever its aspect ratio)
    fit = "contain" : show the whole frame, with dark bars where it does not fill
    """

    def __init__(self, source: VideoSource, title: str = "", fit: str = "contain"):
        self.src = source
        self.title = title
        self.fit = fit

    def render(self, t: float, w: int, h: int) -> np.ndarray:
        frame = self.src.frame_at(t)
        if frame is None:
            img = np.full((h, w, 3), 28, np.uint8)
            k = max(0.45, min(1.6, h / 420.0))
            txt = "no video at this time"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, 1)
            cv2.putText(img, txt, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 * k, (120, 120, 120), max(1, int(k)), cv2.LINE_AA)
            return _tag(img, self.title)
        ih, iw = frame.shape[:2]
        if self.fit == "cover":
            s = max(w / iw, h / ih)
            nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            x, y = (nw - w) // 2, (nh - h) // 2
            img = np.ascontiguousarray(resized[y:y + h, x:x + w])
        else:
            s = min(w / iw, h / ih)
            nw, nh = int(iw * s), int(ih * s)
            img = np.full((h, w, 3), 28, np.uint8)
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            y, x = (h - nh) // 2, (w - nw) // 2
            img[y:y + nh, x:x + nw] = resized
            _tag(img[y:y + nh, x:x + nw], self.title)     # caption on the picture, not the bar
            return img
        return _tag(img, self.title)


class ScrollingTimeSeriesPanel:
    """Time plot with a fixed cursor and an axis that scrolls to the left.

    The whole series is drawn once as a very wide "strip"; each frame simply
    crops a window out of that strip. Matplotlib therefore runs once in
    total rather than once per frame.

    cursor_frac: horizontal position of the cursor inside the panel
                 (0.75 -> near the right, so the past is wide and the
                 future narrow)
    """

    def __init__(self, telemetry: TelemetrySource, channels, window_s: float = 60.0,
                 cursor_frac: float = 0.75, colors=None, axis_px: int = 0,
                 epoch_unix: Optional[float] = None, index: int = 0,
                 strip_windows: float = 3.0, whisker: float = 1.5,
                 pad_frac: float = 0.18):
        """channels: list of dicts {"label", "cols", "unit"} the panel can show.

        The label of the current one is drawn as a drop-down in the corner;
        clicking it opens the list, clicking an entry switches the plot.
        """
        self.tel = telemetry
        self.epoch = epoch_unix     # when given, the x axis shows UTC clock time
        self.whisker = float(whisker)               # how far past the middle half to keep
        self.pad_frac = float(pad_frac)             # blank margin above and below the curves
        self.strip_windows = float(strip_windows)   # how many windows are drawn either side
        self._strip_span = 0.0
        self.channels = list(channels)
        self.index = int(index)
        self.menu_open = False
        self.window_s = float(window_s)
        self.cursor_frac = float(cursor_frac)
        self.colors = colors or ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
        self.axis_px = axis_px
        self._strip = None
        self._size = None

    @property
    def columns(self):
        return self.channels[self.index]["cols"]

    @property
    def title(self):
        return self.channels[self.index]["label"]

    @property
    def ylabel(self):
        return self.channels[self.index].get("unit", "")

    def select(self, i):
        """Show another channel; the strip is rebuilt on the next frame."""
        if 0 <= i < len(self.channels) and i != self.index:
            self.index = i
            self._strip = None
        self.menu_open = False

    # ------------------------- drop-down -------------------------------
    def _menu_geometry(self, w, h):
        """Button rectangle and the row height of the open list, in panel pixels."""
        bh = max(18, int(0.085 * h))
        bw = max(140, int(0.36 * w))
        x1, y0 = w - max(6, int(0.01 * w)), max(4, int(0.02 * h))
        return (x1 - bw, y0, x1, y0 + bh), bh

    @staticmethod
    def _short(col: str) -> str:
        """A readable series name when the channel does not give one."""
        for prefix in ("airspeed_", "ground_tether_", "ground_", "kite_actual_",
                       "kite_measured_", "kite_0_", "kite_"):
            if col.startswith(prefix):
                col = col[len(prefix):]
                break
        return col.replace("_", " ")

    def _draw_legend(self, img, w, h):
        """Colour key for the current channel, drawn where the strip cannot scroll it away."""
        ch = self.channels[self.index]
        cols = ch["cols"]
        if len(cols) < 2:
            return img
        names = ch.get("names") or [self._short(c) for c in cols]
        k = max(0.4, min(1.4, h / 420.0))
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.42 * k
        th = max(1, int(round(k)))
        line, gap, pad = int(16 * k), int(6 * k), int(5 * k)
        rows = list(zip(names, self.colors))
        tw = max(cv2.getTextSize(n, font, scale, th)[0][0] for n, _ in rows)
        rh = int(15 * k)
        x0, y0 = self.axis_px + int(10 * k), int(0.13 * h)
        x1, y1 = x0 + line + gap + tw + 2 * pad, y0 + rh * len(rows) + pad
        roi = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = (roi * 0.15 + 255 * 0.85).astype(np.uint8)   # soft white box
        for i, (name, colour) in enumerate(rows):
            cy = y0 + pad + int(rh * (i + 0.5))
            bgr = tuple(int(colour.lstrip("#")[j:j + 2], 16) for j in (4, 2, 0))
            cv2.line(img, (x0 + pad, cy), (x0 + pad + line, cy), bgr, max(2, int(2 * k)),
                     cv2.LINE_AA)
            cv2.putText(img, name, (x0 + pad + line + gap, cy + int(4 * k)),
                        font, scale, (40, 40, 40), th, cv2.LINE_AA)
        return img

    def _draw_menu(self, img, w, h):
        (bx0, by0, bx1, by1), bh = self._menu_geometry(w, h)
        k = max(0.4, min(1.4, h / 420.0))
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k
        th = max(1, int(round(1.1 * k)))

        def row(y0, y1, text, active=False):
            cv2.rectangle(img, (bx0, y0), (bx1, y1), (252, 252, 252), -1)
            cv2.rectangle(img, (bx0, y0), (bx1, y1), (150, 150, 150), 1)
            colour = (20, 20, 20) if active else (70, 70, 70)
            cv2.putText(img, text, (bx0 + int(8 * k), y1 - int(0.32 * (y1 - y0))),
                        font, scale, colour, th, cv2.LINE_AA)

        row(by0, by1, self.title + "   v", True)
        if self.menu_open:
            for i, ch in enumerate(self.channels):
                y0 = by1 + i * bh
                row(y0, y0 + bh, ch["label"], i == self.index)
        return img

    def click(self, px, py, w, h):
        """Handle a click at panel pixel (px, py). True if the panel used it."""
        (bx0, by0, bx1, by1), bh = self._menu_geometry(w, h)
        if self.menu_open:
            if bx0 <= px <= bx1 and py > by1:
                i = int((py - by1) // bh)
                if 0 <= i < len(self.channels):
                    self.select(i)
                    return True
            self.menu_open = False
            return True
        if bx0 <= px <= bx1 and by0 <= py <= by1:
            self.menu_open = True
            return True
        return False

    def _build(self, w, h, centre):
        if not self.axis_px:
            self.axis_px = int(round(0.12 * w))   # proportional to panel width
        plot_w = max(64, w - self.axis_px)
        self.px_per_s = plot_w / self.window_s
        # Draw only a few windows' worth around the current moment instead of the
        # whole flight: a large panel needs ~25 px per second, which for a 30 min
        # flight would be a 45 000 px wide figure - beyond what the renderer can
        # produce, and it silently came out short, leaving the panel blank later on.
        half = self.strip_windows * self.window_s
        self.t0 = max(self.tel.t_start, centre - half)
        span = min(self.tel.t_end, centre + half) - self.t0
        span = max(span, self.window_s)
        strip_w = int(span * self.px_per_s) + 2

        dpi = 100
        total_w = self.axis_px + strip_w
        fig = _new_fig(total_w, h, dpi)
        ax = fig.add_axes([self.axis_px / total_w, 0.16,
                           strip_w / total_w, 0.74])
        for col, c in zip(self.columns, self.colors):
            t, v = self.tel.series(col)
            if t is None or not np.isfinite(v).any():
                continue
            ax.plot(t, v, color=c, linewidth=1.2, label=col)
        ax.set_xlim(self.t0, self.t0 + span)

        # The y range comes from the body of the distribution over the WHOLE
        # flight, not from the extremes: one spike at take-off would otherwise
        # squash the rest into a thin band. Outliers simply run off the chart.
        # A channel can override this with "ylim": (lo, hi).
        ylim = self.channels[self.index].get("ylim")
        if ylim is None:
            lo_p, hi_p = [], []
            for c in self.columns:
                _, v = self.tel.series(c)
                if v is None or not np.isfinite(v).any():
                    continue
                q1, q3 = np.nanpercentile(v, [25, 75])
                iqr = q3 - q1
                if iqr > 0:
                    # the usual outlier fence: keep the middle of the distribution
                    # and let brief extremes (a take-off spike) run off the chart
                    lo_p.append(max(np.nanmin(v), q1 - self.whisker * iqr))
                    hi_p.append(min(np.nanmax(v), q3 + self.whisker * iqr))
                else:
                    lo_p.append(np.nanmin(v))
                    hi_p.append(np.nanmax(v))
            if lo_p:
                lo, hi = min(lo_p), max(hi_p)
                pad = self.pad_frac * max(hi - lo, 1e-9)
                ylim = (lo - pad, hi + pad)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=_fs(w, 8))
        if self.ylabel:
            ax.set_ylabel(self.ylabel, fontsize=_fs(w, 9))
        step = _nice_step(self.window_s)
        if self.epoch is not None:
            # align ticks to round clock times (…:10, …:20) rather than to t = 0
            ax.xaxis.set_major_locator(MultipleLocator(step, offset=(-self.epoch) % step))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(step))
        if self.epoch is not None:
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: utc_string(self.epoch, x, "%H:%M:%S")))
            ax.set_xlabel("time [UTC]", fontsize=_fs(w, 9))
        else:
            ax.set_xlabel("t [s]", fontsize=_fs(w, 9))
        # a tick label sitting at the very start of the strip would spill into
        # the fixed y-axis area on the left; hide any that do
        fig.canvas.draw()
        left = ax.get_window_extent().x0
        for lab in ax.get_xticklabels():
            if lab.get_window_extent().x0 < left:
                lab.set_visible(False)
        strip = _fig_to_bgr(fig)

        # blank padding on both sides so cropping still works at the edges
        pad = np.full((h, plot_w, 3), 255, np.uint8)
        self._axis = strip[:, :self.axis_px].copy()
        self._axis[int(0.86 * h):, :] = 255   # erase the clipped x tick labels
        self._strip = np.concatenate([pad, strip[:, self.axis_px:], pad], axis=1)
        self._pad = plot_w
        self._plot_w = plot_w
        self._size = (w, h)
        self._strip_span = span

    def render(self, t: float, w: int, h: int) -> np.ndarray:
        left_t = t - self.cursor_frac * self.window_s
        need = (self._strip is None or self._size != (w, h)
                or left_t < self.t0 - 1e-6
                or left_t + self.window_s > self.t0 + self._strip_span + 1e-6)
        if need:
            self._build(w, h, t)
        x0 = self._pad + int(round((left_t - self.t0) * self.px_per_s))
        x0 = max(0, min(x0, self._strip.shape[1] - self._plot_w))
        img = np.concatenate([self._axis, self._strip[:, x0:x0 + self._plot_w]], axis=1)

        cx = self.axis_px + int(self.cursor_frac * self._plot_w)
        cv2.line(img, (cx, int(0.10 * h)), (cx, int(0.84 * h)), (0, 0, 220), 1, cv2.LINE_AA)
        self._draw_legend(img, w, h)
        return self._draw_menu(img, w, h)


class WireframePanel:
    """2D projection of the point cloud. view: 'front' (x,z) | 'side' (y,z) | 'bottom' (x,y)"""

    AXES = {"front": (0, 2, "x [m]", "z [m]"),
            "side":  (1, 2, "y [m]", "z [m]"),
            "bottom": (0, 1, "x [m]", "y [m]")}

    def __init__(self, points: PointCloudSource, view: str = "front",
                 title: str = "", lims=None, invert_x: bool = False):
        self.pts = points
        self.view = view
        self.title = title or view
        self.lims = lims
        self.invert_x = invert_x

    def _build(self, w, h):
        xi, yi, xlab, ylab = self.AXES[self.view]
        self._fig = _new_fig(w, h)
        ax = self._fig.add_subplot(1, 1, 1)
        self._pts_line, = ax.plot([], [], "o", color="red", ms=3.2, ls="none",
                                  animated=True)
        if self.lims:
            ax.set_xlim(*self.lims[xi])
            ax.set_ylim(*self.lims[yi])
        ax.set_xlabel(xlab, fontsize=_fs(w, 9))
        ax.set_ylabel(ylab, fontsize=_fs(w, 9))
        ax.set_title(self.title, fontsize=_fs(w, 11))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=_fs(w, 8))
        if self.invert_x:
            ax.invert_xaxis()
        ax.set_autoscale_on(False)
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._bg = self._fig.canvas.copy_from_bbox(self._fig.bbox)
        self._ax = ax
        self._idx = (xi, yi)
        self._size = (w, h)

    def render(self, t: float, w: int, h: int) -> np.ndarray:
        if getattr(self, "_fig", None) is None or self._size != (w, h):
            self._build(w, h)
        self._fig.canvas.restore_region(self._bg)
        fr = self.pts.points_at(t)
        xi, yi = self._idx
        if fr is not None and len(fr):
            P = fr[["x", "y", "z"]].to_numpy(float)
            self._pts_line.set_data(P[:, xi], P[:, yi])
        else:
            self._pts_line.set_data([], [])
        self._ax.draw_artist(self._pts_line)
        self._fig.canvas.blit(self._fig.bbox)
        rgba = np.asarray(self._fig.canvas.buffer_rgba())
        return cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)

    def close(self):
        self._fig = None


def mount_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Constant correction between the attitude sensor's frame and the kite's.

    Applied as B_true = M @ B_sensor, where the rows of B are the body axes.
    Roll turns about the nose, pitch about the right wing, yaw about the down
    axis. Estimated from flight data by kat_validate.py, check 5.
    """
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    Rz = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotate_about_body_x(B: np.ndarray, deg: float) -> np.ndarray:
    """Turn a body frame about its own x (nose) axis by `deg` degrees.

    B has the body axes as rows. The nose stays where it is; only the right
    and down axes turn. Used to correct an IMU mounted at an angle about the
    nose axis (see kat_validate.py, check 5).
    """
    if not deg:
        return B
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    y, z = B[..., 1, :], B[..., 2, :]
    return np.stack([B[..., 0, :], c * y + s * z, -s * y + c * z], axis=-2)


class View3DMixin:
    """Interactive camera for the 3D panels: rotate, zoom, reset.

    The panel keeps its default view and the base axis limits; set_view()
    changes the camera and re-captures the blitting background, so the
    per-frame cost stays the same after the change.
    """

    def _init_view(self, elev, azim):
        self.elev, self.azim = elev, azim
        self._view0 = (elev, azim)
        self.zoom = 1.0                  # < 1 zooms in, > 1 zooms out
        self._lims0 = None               # base limits, set when the axes are built

    def _apply_view(self, ax):
        ax.view_init(elev=self.elev, azim=self.azim)
        if self._lims0:
            setters = (ax.set_xlim, ax.set_ylim, ax.set_zlim)
            for setter, (lo, hi) in zip(setters, self._lims0):
                c, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * self.zoom
                setter(c - half, c + half)

    def set_view(self, d_elev=0.0, d_azim=0.0, zoom_factor=1.0, reset=False):
        if reset:
            self.elev, self.azim = self._view0
            self.zoom = 1.0
        else:
            self.elev = float(np.clip(self.elev + d_elev, -89.0, 89.0))
            self.azim = (self.azim + d_azim + 180.0) % 360.0 - 180.0
            self.zoom = float(np.clip(self.zoom * zoom_factor, 0.05, 3.0))
        if getattr(self, "_fig", None) is not None:
            self._apply_view(self._ax)
            self._fig.canvas.draw()
            self._bg = self._fig.canvas.copy_from_bbox(self._fig.bbox)


def default_kite_mesh(n_span: int = 28, arc_deg: float = 74.0, taper: float = 0.35,
                      aspect: float = 3.16):
    """A simple LEI-style canopy in body axes (x forward, y right, z down).

    Used when no .obj model is supplied. Proportions follow the published
    TU Delft V3 figures: projected span 8.32 m, projected height 3.13 m,
    flat maximum chord 2.63 m — so height/span = 0.376 (arc_deg = 74) and
    span/chord = 3.16. The span is normalised to 1 and the tips curve
    towards +z, i.e. towards the tether, as on a real LEI kite.
    Returns (vertices, triangles).
    """
    phi = np.radians(np.linspace(-arc_deg, arc_deg, n_span))
    R = 0.5 / np.sin(np.radians(arc_deg))          # projected span = 1
    y = R * np.sin(phi)
    z = R * (1.0 - np.cos(phi))
    s = np.linspace(-1.0, 1.0, n_span)
    chord = (1.0 / aspect) * (1.0 - taper * s ** 2)
    le = np.column_stack([0.30 * chord, y, z])     # leading edge, forward
    mid = np.column_stack([0.0 * chord, y, z - 0.02 * chord])
    te = np.column_stack([-0.70 * chord, y, z])    # trailing edge
    V = np.vstack([le, mid, te])
    F = []
    for row in (0, 1):
        a, b = row * n_span, (row + 1) * n_span
        for i in range(n_span - 1):
            F += [(a + i, b + i, b + i + 1), (a + i, b + i + 1, a + i + 1)]
    return V, np.asarray(F, int)


def simplify_mesh(V: np.ndarray, F: np.ndarray, max_faces: int = 300):
    """Reduce a mesh to roughly max_faces triangles by vertex clustering.

    Vertices falling in the same cell of a regular grid are merged into one;
    triangles that collapse are dropped. The grid is coarsened until the
    triangle count is low enough. Shape is kept, fine detail is not.
    """
    if len(F) <= max_faces:
        return V, F
    lo, hi = V.min(axis=0), V.max(axis=0)
    n = 96                                   # cells along the longest side
    for _ in range(12):
        cell = max((hi - lo).max(), 1e-9) / n
        keys = np.floor((V - lo) / cell).astype(np.int64)
        # np.unique returns (values, first occurrences, inverse) in that order
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)     # numpy 2 keeps a trailing axis
        Vs = V[first]                        # one representative vertex per cell
        Fs = inverse[F]
        Fs = Fs[(Fs[:, 0] != Fs[:, 1]) & (Fs[:, 1] != Fs[:, 2]) & (Fs[:, 0] != Fs[:, 2])]
        Fs = np.unique(np.sort(Fs, axis=1), axis=0)
        if len(Fs) <= max_faces:
            return Vs, Fs
        n = max(4, int(n * 0.75))
    return Vs, Fs


_OBJ_CACHE = {}


def load_obj(path: str, axes_map: str = "+x +y +z", max_faces: int = 300):
    """Minimal Wavefront .obj reader: vertices and faces only.

    axes_map says which .obj axis becomes each body axis (forward, right,
    down). Example: "-z +x -y" means body_x = -obj_z, body_y = +obj_x,
    body_z = -obj_y. The model is centred and scaled to a span of 1.

    A CAD export can carry tens of thousands of triangles; anything above
    max_faces is simplified on load, so the file needs no preparation. The
    kite is a small object on screen, and matplotlib draws 3D faces slowly:
    200 triangles cost about 14 ms per frame, 1400 about 64 ms, with no
    visible gain.
    """
    key = (path, axes_map, max_faces)
    if key in _OBJ_CACHE:
        return _OBJ_CACHE[key]
    V, F = [], []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) for p in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(V) + i for i in idx]
                for k in range(1, len(idx) - 1):          # fan triangulation
                    F.append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(V, float)
    cols = []
    for tok in axes_map.split():
        sign = -1.0 if tok.startswith("-") else 1.0
        cols.append(sign * V[:, "xyz".index(tok[-1])])
    V = np.column_stack(cols)
    V -= 0.5 * (V.min(axis=0) + V.max(axis=0))
    V /= max(np.ptp(V[:, 1]), 1e-9)                     # span (body y) -> 1
    F = np.asarray(F, int)
    n_in = len(F)
    V, F = simplify_mesh(V, F, max_faces)
    if len(F) < n_in:
        print(f"[kat] {path}: {n_in} triangles simplified to {len(F)} for display")
    _OBJ_CACHE[key] = (V, F)
    return V, F


class Path3DPanel(View3DMixin):
    """The kite's trajectory in the sky plus a body-axis triad.

    Trajectory: ENU [m], centred on the ground station
    Arrows    : body axes built from the Euler angles (3-2-1, NED->FRD)
                red = nose (forward), green = right wing, blue = down
    """

    POS = ("kite_pos_east", "kite_pos_north", "kite_height")
    ATT = ("kite_0_roll", "kite_0_pitch", "kite_0_yaw")

    def __init__(self, telemetry: TelemetrySource, title: str = "3D flight path",
                 trail_s: float = 12.0, triad_len: float = 30.0,
                 elev: float = 22.0, azim: float = -60.0,
                 yaw_offset_deg: float = 0.0, model=None,
                 model_size: Optional[float] = None, show_triad: bool = True,
                 mount_rpy=(0.0, 0.0, 0.0), show_tether: bool = True):
        self.tel = telemetry
        # draw the tether from the ground station to the kite; the axes then
        # have to include the origin, so the box grows and the kite looks smaller
        self.show_tether = show_tether
        # sensor-to-kite mounting correction (roll, pitch, yaw in degrees),
        # estimated by kat_validate.py check 5
        self.mount = mount_matrix(*mount_rpy) if any(mount_rpy) else None
        # kite model: (vertices, faces) in body axes with span 1, or None for no model
        self.model = model
        self.model_size = model_size if model_size is not None else 1.8 * triad_len
        self.show_triad = show_triad
        self.title = title
        self.trail_s = trail_s
        self.L = triad_len
        self._init_view(elev, azim)
        self.yaw_offset = float(yaw_offset_deg)   # mounting / convention correction
        self.t_all = telemetry.t
        self.path = np.column_stack([telemetry.series(c)[1] for c in self.POS]) \
            if all(telemetry.series(c)[0] is not None for c in self.POS) else None

    @staticmethod
    def _body_axes_enu(roll_deg, pitch_deg, yaw_deg):
        r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
        cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                                  np.sin(p), np.cos(y), np.sin(y))
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R = Rz @ Ry @ Rx                      # body -> NED
        ned = R.T                             # rows: x_b, y_b, z_b; columns: N, E, D
        # NED -> ENU swaps the COLUMNS (components), not the rows (axes):
        #   e = v_E, n = v_N, u = -v_D
        return np.column_stack([ned[:, 1], ned[:, 0], -ned[:, 2]])

    def _build(self, w, h):
        """Build the scene once: axes, grey trajectory, empty trail and marker."""
        self._fig = _new_fig(w, h)
        ax = self._fig.add_subplot(1, 1, 1, projection="3d")
        self._ax = ax

        if self.path is not None and np.isfinite(self.path).any():
            P = self.path
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color="0.55", lw=1.1)
            lo, hi = np.nanmin(P, axis=0), np.nanmax(P, axis=0)
            if self.show_tether:                      # keep the ground station in view
                lo, hi = np.minimum(lo, 0.0), np.maximum(hi, 0.0)
            pad = 0.12 * np.maximum(hi - lo, 1.0)
            self._lims0 = [(lo[0] - pad[0], hi[0] + pad[0]),
                           (lo[1] - pad[1], hi[1] + pad[1]),
                           (max(0.0, lo[2] - pad[2]), hi[2] + pad[2])]
        else:
            ax.text2D(0.5, 0.5, "position columns missing",
                      ha="center", transform=ax.transAxes, fontsize=11)

        # only the data of these artists changes from frame to frame
        self._mesh = None
        if self.model is not None:
            self._mesh = Poly3DCollection([], animated=True, edgecolor=(0.1, 0.1, 0.1, 0.5),
                                          linewidths=0.3)
            ax.add_collection3d(self._mesh)
        if self.show_tether:
            self._tether, = ax.plot([], [], [], color="0.45", lw=1.0, animated=True)
            ax.plot([0], [0], [0], marker="^", color="0.25", ms=6, ls="none")   # ground station
        else:
            self._tether = None
        self._trail, = ax.plot([], [], [], color="#1f77b4", lw=2.8, animated=True)
        self._dot, = ax.plot([], [], [], "o", color="black", ms=6, animated=True)
        # body-axis triad as three plain lines: far cheaper than quiver, which
        # cannot be updated and would have to be recreated on every frame
        self._triad = [ax.plot([], [], [], color=c, lw=2.6, marker="o",
                               markevery=[-1], ms=5, animated=True)[0]
                       for c in ("#d62728", "#2ca02c", "#1f77b4")]

        # fewer ticks and some label padding: the default crowds the small panel
        ax.locator_params(nbins=4)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.labelpad = _fs(w, 6)
        ax.set_xlabel("east [m]", fontsize=_fs(w, 9))
        ax.set_ylabel("north [m]", fontsize=_fs(w, 9))
        ax.set_zlabel("height [m]", fontsize=_fs(w, 9))
        ax.set_title(self.title, fontsize=_fs(w, 11))
        ax.tick_params(labelsize=_fs(w, 7))
        self._apply_view(ax)
        ax.set_autoscale_on(False)
        self._fig.tight_layout()
        # snapshot of everything that never changes, so later frames only
        # have to draw the trail, the marker and the triad on top of it
        self._fig.canvas.draw()
        self._bg = self._fig.canvas.copy_from_bbox(self._fig.bbox)
        self._size = (w, h)

    def render(self, t: float, w: int, h: int) -> np.ndarray:
        if getattr(self, "_fig", None) is None or self._size != (w, h):
            self._build(w, h)
        self._fig.canvas.restore_region(self._bg)

        if self.path is not None and np.isfinite(self.path).any():
            P = self.path
            m = (self.t_all >= t - self.trail_s) & (self.t_all <= t)
            if m.any():
                self._trail.set_data_3d(P[m, 0], P[m, 1], P[m, 2])
            pos = np.array([self.tel.value_at(c, t) for c in self.POS])
            self._dot.set_data_3d([pos[0]], [pos[1]], [pos[2]])

            att = [self.tel.angle_at(c, t) for c in self.ATT]     # short way round
            att[2] = att[2] - self.yaw_offset
            ok = all(np.isfinite(att)) and np.all(np.isfinite(pos))
            if ok:
                B = self._body_axes_enu(*att)            # rows: body axes in ENU
                if self.mount is not None:
                    B = self.mount @ B
                for line, vec in zip(self._triad, B * self.L):
                    line.set_data_3d([pos[0], pos[0] + vec[0]],
                                     [pos[1], pos[1] + vec[1]],
                                     [pos[2], pos[2] + vec[2]])
                if self._mesh is not None:
                    V, F = self.model
                    W = pos + (V @ B) * self.model_size   # body -> ENU, placed at pos
                    tris = W[F]
                    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
                    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
                    light = np.array([0.35, -0.45, 0.82])
                    k = 0.35 + 0.65 * np.abs(n @ light)   # simple two-sided shading
                    base = np.array([0.22, 0.28, 0.40])
                    self._mesh.set_verts(list(tris))
                    self._mesh.set_facecolor(np.column_stack([np.outer(k, base), np.full(len(k), 0.95)]))
                    self._mesh.do_3d_projection()
                    self._ax.draw_artist(self._mesh)

            if self._tether is not None and np.all(np.isfinite(pos)):
                self._tether.set_data_3d([0.0, pos[0]], [0.0, pos[1]], [0.0, pos[2]])
            arts = [self._trail, self._dot]
            if self._tether is not None:
                arts.append(self._tether)
            arts += self._triad if (ok and self.show_triad) else []
            for art in arts:
                self._ax.draw_artist(art)

        self._fig.canvas.blit(self._fig.bbox)
        rgba = np.asarray(self._fig.canvas.buffer_rgba())
        return cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)

    def close(self):
        self._fig = None


class Points3DPanel(View3DMixin):
    """3D view of the point cloud (the counterpart of Pim's top-left panel)."""

    def __init__(self, points: PointCloudSource, title: str = "3D shape",
                 lims=None, elev: float = 20.0, azim: float = -60.0):
        self.pts = points
        self.title = title
        self.lims = lims
        self._init_view(elev, azim)

    def _build(self, w, h):
        self._fig = _new_fig(w, h)
        ax = self._fig.add_subplot(1, 1, 1, projection="3d")
        self._pts_line, = ax.plot([], [], [], "o", color="red", ms=2.8, ls="none",
                                  animated=True)
        if self.lims:
            self._lims0 = [tuple(l) for l in self.lims]
        ax.set_xlabel("x [m]", fontsize=_fs(w, 8))
        ax.set_ylabel("y [m]", fontsize=_fs(w, 8))
        ax.set_zlabel("z [m]", fontsize=_fs(w, 8))
        ax.set_title(self.title, fontsize=_fs(w, 11))
        ax.tick_params(labelsize=_fs(w, 7))
        self._apply_view(ax)
        ax.set_autoscale_on(False)
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._bg = self._fig.canvas.copy_from_bbox(self._fig.bbox)
        self._ax = ax
        self._size = (w, h)

    def render(self, t: float, w: int, h: int) -> np.ndarray:
        if getattr(self, "_fig", None) is None or self._size != (w, h):
            self._build(w, h)
        self._fig.canvas.restore_region(self._bg)
        fr = self.pts.points_at(t)
        if fr is not None and len(fr):
            P = fr[["x", "y", "z"]].to_numpy(float)
            self._pts_line.set_data_3d(P[:, 0], P[:, 1], P[:, 2])
        else:
            self._pts_line.set_data_3d([], [], [])
        self._ax.draw_artist(self._pts_line)
        self._fig.canvas.blit(self._fig.bbox)
        rgba = np.asarray(self._fig.canvas.buffer_rgba())
        return cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)

    def close(self):
        self._fig = None


class MetricsPanel:
    """Instantaneous numeric values. Takes a list of (col, label, unit, decimals)."""

    def __init__(self, telemetry: TelemetrySource, rows, title: str = ""):
        self.tel = telemetry
        self.rows = rows
        self.title = title

    def render(self, t: float, w: int, h: int) -> np.ndarray:
        img = np.full((h, w, 3), 255, np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        k = max(0.75, min(2.4, w / REF_PANEL_W))
        y = int(54 * k)
        if self.title:
            cv2.putText(img, self.title, (int(16 * k), y), font, 0.7 * k,
                        (0, 0, 0), max(1, int(2 * k)), cv2.LINE_AA)
            y += int(40 * k)
        for col, label, unit, prec in self.rows:
            v = self.tel.value_at(col, t)
            txt = f"{label}: " + (f"{v:.{prec}f} {unit}".strip() if np.isfinite(v) else "-")
            cv2.putText(img, txt, (int(16 * k), y), font, 0.6 * k,
                        (30, 30, 30), max(1, int(1.5 * k)), cv2.LINE_AA)
            y += int(34 * k)
            if y > h - 10:
                break
        return img


# ===================================================================
# 3. GRID + RENDER
# ===================================================================

@dataclass
class Slot:
    """Where a panel sits in the grid."""
    panel: object
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1


def _draw_footer(canvas, text: str, bar_h: int):
    """Dark status bar along the bottom edge with one line of text."""
    H, W = canvas.shape[:2]
    canvas[H - bar_h:, :] = (32, 32, 32)
    scale = bar_h / 42.0
    cv2.putText(canvas, text, (int(14 * scale), H - int(bar_h * 0.30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (240, 240, 240),
                max(1, int(round(1.6 * scale))), cv2.LINE_AA)


def _draw_separators(canvas, slots, cw, ch, panels_h):
    """Thin light lines along every panel edge that faces another panel."""
    W = canvas.shape[1]
    px = max(1, int(round(W / 900)))
    colour = (205, 205, 205)
    for s in slots:
        x0, y0 = s.col * cw, s.row * ch
        x1, y1 = x0 + cw * s.colspan, y0 + ch * s.rowspan
        if x1 < W - 1:                                   # right edge is shared
            canvas[y0:y1, x1 - px // 2 - px % 2:x1 + px // 2] = colour
        if y1 < panels_h - 1:                            # bottom edge is shared
            canvas[y1 - px // 2 - px % 2:y1 + px // 2, x0:x1] = colour


def compose_frame(slots: List[Slot], n_rows: int, n_cols: int,
                  t: float, out_size=(1920, 1080), clock_label=True,
                  footer: Optional[Callable[[float], str]] = None,
                  separators: bool = True) -> np.ndarray:
    """Render every panel at time t into the grid and return a single frame.

    footer: optional function t -> text, drawn as a status bar at the bottom
            (date, time and live values). The panels share the remaining height.
    """
    W, H = out_size
    bar_h = max(26, int(0.032 * H)) if footer else 0
    cw, ch = W // n_cols, (H - bar_h) // n_rows
    canvas = np.full((H, W, 3), 255, np.uint8)
    for s in slots:
        w, h = cw * s.colspan, ch * s.rowspan
        img = s.panel.render(t, w, h)
        if img.shape[0] != h or img.shape[1] != w:      # matplotlib can round a pixel off
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        x, y = s.col * cw, s.row * ch
        canvas[y:y + h, x:x + w] = img
    if separators and len(slots) > 1:
        _draw_separators(canvas, slots, cw, ch, ch * n_rows)
    if footer:
        _draw_footer(canvas, footer(t), bar_h)
    elif clock_label:
        _label(canvas, f"t = {t:8.3f} s", (16, H - 16), 0.8)
    return canvas


def render_video(slots: List[Slot], n_rows: int, n_cols: int,
                 t_start: float, t_end: float, fps: float,
                 out_path: str, out_size=(1920, 1080), clock_label=True,
                 footer: Optional[Callable[[float], str]] = None):
    W, H = out_size
    cw, ch = W // n_cols, H // n_rows
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("could not open VideoWriter")

    n = int(round((t_end - t_start) * fps))
    for i in range(n):
        t = t_start + i / fps
        writer.write(compose_frame(slots, n_rows, n_cols, t, out_size, clock_label, footer))
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{n} frames")
    writer.release()
    print(f"[+] {out_path}  ({n} frames)")
