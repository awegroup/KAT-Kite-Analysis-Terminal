#!/usr/bin/env python3
"""
KAT - Kite Analysis Terminal
Configuration: which files, which panels, where. Engine lives in kat.py.

Two layouts:

  DEFAULT (4 panels, 2x2) - works for any flight:
      airborne camera | 3D position + attitude (with trail)
      ground station  | time signals

  EXTENDED (adds the deformation panels) - only for flights that have
  photogrammetry data. Enabled with WITH_DEFORMATION = True.
"""

import importlib
import os
import sys

import pandas as pd

import numpy as np

from kat import (
    VideoSource, TelemetrySource, PointCloudSource, repair_time_axis,
    default_kite_mesh, load_obj, utc_string,
    VideoPanel, ScrollingTimeSeriesPanel, WireframePanel, MetricsPanel,
    Path3DPanel, Points3DPanel,
    Slot,
)

# ========================== CONFIGURATION =========================

WITH_DEFORMATION = False      # True -> add the wireframe panels
AIRBORNE = "kcu"              # which airborne camera, from the dict below

# --- Synchronisation -------------------------------------------------
# Offsets are not written one by one; they are derived from a shared
# anchor: the moment of take-off. If both videos drift together relative
# to the telemetry, TELEMETRY_TAKEOFF_S is the single number to correct,
# and the spacing between the videos stays intact.

TELEMETRY_TAKEOFF_S = 2208.3   # take-off instant in the flight log [s]
TELEMETRY_LANDING_S = 3930.3   # landing instant in the flight log [s]

CLIP_TO_AIRBORNE = True        # keep only the flight window, drop the long ground time
PRE_TAKEOFF_S = 150.0          # ground time kept before take-off [s]
POST_LANDING_S = 60.0          # time kept after landing [s]

# The synthetic complete dataset (deformation panels) has its own t = 0,
# at this second of the flight log; it is shifted onto the master clock.
DATASET_START_S = 2417.0

# One entry per camera. "takeoff_s" is the second of THAT video at which the
# kite leaves the ground; the offsets follow from it (see below).
VIDEOS = {
    "kcu":  {"path": "output/kcu_proxy.mp4", "takeoff_s": 168.5, "rotate_180": True},
    "wing": {"path": "output/wing_proxy.mp4", "takeoff_s": None, "rotate_180": False},
    "gs":   {"path": "output/gs_proxy.mp4", "takeoff_s": 399.5, "rotate_180": False},
}
GS_KEY = "gs"                  # which entry above is the ground station camera

# Telemetry source:
#   True  -> read the flight log directly, THE WHOLE FLIGHT is available
#   False -> synthetic complete dataset (required by the deformation panels)
USE_FLIGHT_LOG = True
FLIGHT_CSV = "flightdata/log_2025-10-09_58-33-00.csv"

DATASET_CSV = "output/synthetic_complete_dataset.csv"
OUT = "output/kat_output.mp4"

DURATION_S = float(os.environ.get("DURATION_S", 60.0))
FPS = 30000 / 1001            # render rate; matches the KCU proxy exactly
WINDOW_S = 60.0               # width of the window shown in the time plots
WARMUP_S = 0.75 * WINDOW_S    # start slightly later so the cursor has a past

LIMS = [(-4.7, 5.3), (-1.4, 1.6), (3.7, 7.8)]   # x, y, z (deformation panels)

# Output resolution. A video panel is OUT_W / n_cols * colspan pixels wide;
# the proxy videos should be generated at roughly that width, otherwise
# they are stretched.
#   2x2 default layout  : video panel = OUT_W / 2
#   3x4 extended layout : video panel = OUT_W / 2  (colspan = 2)
OUT_SIZE_DEFAULT = (2560, 1440)      # video panel 1280 px
OUT_SIZE_EXTENDED = (2560, 1440)     # video panel 1280 px

# Viewer: graphs are always composed at the window's own pixel size, so they
# stay sharp at any window size; only the videos come from the 720p proxies.
VIEW_SIZE = (1600, 900)              # initial window size
VIEW_FPS = 30.0                      # upper bound on frames drawn per second
VIEW_PLAY_SPEED = 1.0                # 1.0 = real time; [ and ] change it in the viewer

# --- 3D kite model --------------------------------------------------
# None -> built-in LEI-style canopy. A path -> that .obj file is loaded.
# KITE_OBJ_AXES says which .obj axis becomes each body axis
# (forward, right, down); change it if the model appears rotated.
KITE_OBJ = "models/V3_Kite.obj"      # None -> built-in canopy
KITE_OBJ_AXES = "-x +y -z"           # SpaceClaim export of the V3 CAD model
KITE_OBJ_MAX_FACES = 300             # CAD exports are simplified to this on load
KITE_MODEL_SIZE = 55.0               # displayed span [m]; enlarged so it is visible
VIDEO_FIT = "contain"                # "contain": the whole frame, never cropped
                                     # "cover": fill the cell by cropping the edges
SHOW_TRIAD = True                    # body-axis arrows on top of the model
SHOW_TETHER = True                   # draw the line from the ground station to the kite
KITE_MOUNT_RPY = (0.0, 0.0, 0.0)     # sensor-to-kite mounting correction, in degrees
                                     # (roll, pitch, yaw); use what check 5 reports

# --- Time-signal channels --------------------------------------------
# What the two time-signal panels can show. The label appears in the panel's
# drop-down; "cols" may list several columns to plot together.
SIGNAL_CHANNELS = [
    {"label": "Tether force", "cols": ["ground_tether_force"], "unit": "kg"},
    {"label": "Apparent wind", "cols": ["kite_measured_va"], "unit": "m/s"},
    {"label": "Flow angles", "cols": ["airspeed_angle_of_attack",
                                      "airspeed_sideslip_angle"], "unit": "deg",
     "names": ["angle of attack", "sideslip"]},
    {"label": "Height", "cols": ["kite_height"], "unit": "m"},
    {"label": "Reel-out speed", "cols": ["ground_tether_reelout_speed"], "unit": "m/s"},
    {"label": "Tether length", "cols": ["ground_tether_length"], "unit": "m"},
    {"label": "Depower / steering", "cols": ["kite_actual_depower",
                                             "kite_actual_steering"], "unit": "",
     "names": ["depower", "steering"]},
    {"label": "Attitude", "cols": ["kite_0_roll", "kite_0_pitch"], "unit": "deg",
     "names": ["roll", "pitch"]},
]
SIGNAL_1 = 0        # channel the upper (or only) signal panel starts on
SIGNAL_2 = 2        # channel the lower signal panel starts on (extended layout)

# --- Live values -----------------------------------------------------
# Shown in the status bar (all layouts) and in the extended Live values panel.
# (column, label, unit, decimals)
LIVE_VALUES = [
    ("kite_measured_va", "Va", "m/s", 1),
    ("kite_actual_depower", "Depower", "", 1),
    ("kite_actual_steering", "Steering", "", 1),
    ("ground_tether_force", "Force", "kg", 0),
    ("kite_height", "Height", "m", 0),
]

# ====================== FLIGHT SELECTION ==========================
# Everything specific to one flight lives in flights/<name>.py and overrides
# the defaults above. Switch flights without touching any other file:
#     $env:KAT_FLIGHT="2025-10-09"; python viewer.py

def available_flights():
    """Every flights/f*.py in the folder, as the names KAT accepts."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flights")
    out = []
    for f in sorted(os.listdir(here)) if os.path.isdir(here) else []:
        if f.startswith("f") and f.endswith(".py"):
            out.append(f[1:-3].replace("_", "-"))
    return out


def _flight_from_argv():
    """--flight 2026-03-15 or --flight=2026-03-15, anywhere on the command line."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("--flight", "-f") and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--flight="):
            return a.split("=", 1)[1]
    return None


FLIGHT = _flight_from_argv() or os.environ.get("KAT_FLIGHT") or "2025-10-09"

if "--list-flights" in sys.argv:
    print("flights defined in flights/:")
    for f in available_flights():
        print(f"   {f}")
    print("\nuse one with:  python viewer.py --flight <name>")
    sys.exit(0)

try:
    _mod = importlib.import_module("flights.f" + FLIGHT.replace("-", "_"))
except ModuleNotFoundError:
    sys.exit(f"[kat] no flight called {FLIGHT!r}.\n"
             f"      defined: {', '.join(available_flights()) or '(none)'}\n"
             f"      add one: copy flights/_template.py to "
             f"flights/f{FLIGHT.replace('-', '_')}.py and fill it in")
for _name, _value in vars(_mod).items():
    if _name.isupper():
        globals()[_name] = _value

# --- derived from the flight ----------------------------------------
# Master clock origin: t = 0 is the first instant shown, so t never goes
# negative. Changing PRE_TAKEOFF_S moves it and every offset follows.
FLIGHT_START_S = max(0.0, TELEMETRY_TAKEOFF_S - PRE_TAKEOFF_S) if CLIP_TO_AIRBORNE else 0.0


def _t0(video_key):
    """Which second of this video corresponds to master clock t = 0."""
    tk = VIDEOS[video_key].get("takeoff_s")
    if tk is None:
        return 0.0
    return tk - TELEMETRY_TAKEOFF_S + FLIGHT_START_S


def _entry(key):
    v = VIDEOS[key]
    return (v["path"], _t0(key), v.get("rotate_180", False))


AIRBORNE_VIDEOS = {k: _entry(k) for k in VIDEOS if k != GS_KEY}
GS_VIDEO = _entry(GS_KEY)

if AIRBORNE not in AIRBORNE_VIDEOS:
    AIRBORNE = next(iter(AIRBORNE_VIDEOS))

# a camera that started recording after the window opens shows "no video" at
# first; say so once rather than leaving it to be discovered on screen
_leads = {k: v["takeoff_s"] for k, v in VIDEOS.items() if v.get("takeoff_s") is not None}
_late = {k: v for k, v in _leads.items() if v < PRE_TAKEOFF_S}
if CLIP_TO_AIRBORNE and _late:
    _first = min(_late.values())
    print(f"[kat] PRE_TAKEOFF_S = {PRE_TAKEOFF_S:.0f} s, but {', '.join(_late)} "
          f"start{'s' if len(_late) == 1 else ''} only {_first:.0f} s before take-off; "
          f"set PRE_TAKEOFF_S <= {_first:.0f} to avoid a blank start")

def check_inputs(with_deformation=None, airborne=None):
    """List the files this flight needs but does not have."""
    missing = []
    if not os.path.exists(FLIGHT_CSV):
        missing.append((FLIGHT_CSV, "flight log"))
    keys = [airborne or AIRBORNE, GS_KEY]
    for k in keys:
        path = VIDEOS[k]["path"]
        if not os.path.exists(path):
            missing.append((path, f"{k} video proxy"))
        elif VIDEOS[k].get("takeoff_s") is None:
            missing.append((path, f"{k}: takeoff_s not filled in"))
    if (WITH_DEFORMATION if with_deformation is None else with_deformation) \
            and not os.path.exists(DATASET_CSV):
        missing.append((DATASET_CSV, "photogrammetry dataset"))
    return missing


def require_inputs(**kw):
    """Stop with an explanation rather than a traceback from deep inside pandas."""
    missing = check_inputs(**kw)
    if not missing:
        return
    print(f"[kat] flight {FLIGHT}: cannot start, {len(missing)} input(s) missing\n")
    for path, what in missing:
        print(f"      {what:28s} {path}")
    print(f"\n      These paths come from flights/f{FLIGHT.replace('-', '_')}.py.")
    print("      The flight log is not part of the repository; proxies are built with")
    print("      scripts/make_proxies.py. See docs/pipeline.md for the five steps.")
    print(f"\n      Other flights defined here: "
          f"{', '.join(f for f in available_flights() if f != FLIGHT) or '(none)'}")
    sys.exit(1)


# ============================== SETUP =============================

# name in the flight log -> name the panels expect
FLIGHT_RENAME = {"airspeed_apparent_windspeed": "kite_measured_va"}


_EPOCH_CACHE = {}


def flight_epoch(path=None):
    """Unix time [s] at master clock t = 0, read from the flight log.

    Returns None when the log's time column is not Unix time, in which case
    the status bar shows only the relative clock.
    """
    path = path or FLIGHT_CSV
    if path not in _EPOCH_CACHE:
        try:
            first = pd.read_csv(path, usecols=["time"], nrows=1)["time"].iloc[0]
            first = float(first)
            _EPOCH_CACHE[path] = first + FLIGHT_START_S if first > 1e9 else None
        except Exception:
            _EPOCH_CACHE[path] = None
    return _EPOCH_CACHE[path]


def make_footer(telemetry, epoch=None):
    """Status-bar text for master clock t: UTC date-time, t, and live values."""
    epoch = flight_epoch() if epoch is None else epoch

    def footer(t):
        parts = []
        stamp = utc_string(epoch, t, "%Y-%m-%d  %H:%M:%S")
        if stamp:
            frac = (epoch + t) % 1.0
            parts.append(f"{stamp}.{int(frac * 10)} UTC")
        parts.append(f"t = {t:7.1f} s")
        for col, label, unit, prec in LIVE_VALUES:
            v = telemetry.value_at(col, t)
            val = f"{v:.{prec}f}" if np.isfinite(v) else "-"
            parts.append(f"{label} {val}{(' ' + unit) if unit else ''}")
        return "    |    ".join(parts)
    return footer


def kite_model():
    """The mesh drawn in the 3D panel: the .obj if configured, else the built-in one."""
    if KITE_OBJ:
        return load_obj(KITE_OBJ, KITE_OBJ_AXES, KITE_OBJ_MAX_FACES)
    return default_kite_mesh()


def load_flight_telemetry(path=None, clip=None):
    """Turn the flight log itself into a telemetry source.

    Master clock: t = 0 corresponds to FLIGHT_START_S in the flight log, so
    the whole flight becomes available without changing the video offsets.
    """
    path = path or FLIGHT_CSV
    clip = CLIP_TO_AIRBORNE if clip is None else clip
    fd = pd.read_csv(path, low_memory=False).rename(columns=FLIGHT_RENAME)
    t = repair_time_axis(fd, "time")
    # concat rather than assign: inserting into a 180-column frame fragments it
    # and pandas warns about it
    fd = pd.concat([fd, pd.Series(t - t[0] - FLIGHT_START_S,
                                  index=fd.index, name="timestamp")], axis=1)
    if clip:
        lo = TELEMETRY_TAKEOFF_S - FLIGHT_START_S - PRE_TAKEOFF_S
        hi = TELEMETRY_LANDING_S - FLIGHT_START_S + POST_LANDING_S
        fd = fd[(fd["timestamp"] >= lo) & (fd["timestamp"] <= hi)]
    keep = [c for c in fd.columns
            if c == "timestamp" or pd.api.types.is_numeric_dtype(fd[c])]
    return fd[keep].reset_index(drop=True)


def build(with_deformation=None, airborne=None, dataset_csv=None, window_s=None):
    """Build the panels and the grid. Used by both render.py and viewer.py.

    The defaults are read here, not in the signature: a signature default is
    frozen when this module is imported, so a setting changed afterwards
    would be silently ignored.
    """
    with_deformation = WITH_DEFORMATION if with_deformation is None else with_deformation
    airborne = airborne or AIRBORNE
    dataset_csv = dataset_csv or DATASET_CSV
    window_s = WINDOW_S if window_s is None else window_s
    # Telemetry always covers the whole flight; the photogrammetry dataset only
    # feeds the deformation panels, so switching layouts never shortens the timeline.
    points_df = None
    if with_deformation:
        if not os.path.exists(dataset_csv):
            raise FileNotFoundError(
                f"{dataset_csv} not found - the deformation panels need a "
                "photogrammetry dataset. Build a stand-in with kat_dataset.py, "
                "or point DATASET_CSV in the flight file at the real one.")
        points_df = pd.read_csv(dataset_csv)
        points_df["timestamp"] = points_df["timestamp"] + (DATASET_START_S - FLIGHT_START_S)
    if USE_FLIGHT_LOG:
        df = load_flight_telemetry()
    else:
        df = points_df if points_df is not None else pd.read_csv(dataset_csv)
    telemetry = TelemetrySource(df)

    air_path, air_t0, air_rot = AIRBORNE_VIDEOS[airborne]
    air = VideoSource(air_path, t0_s=air_t0, rotate_180=air_rot)
    gs = VideoSource(GS_VIDEO[0], t0_s=GS_VIDEO[1], rotate_180=GS_VIDEO[2])

    epoch = flight_epoch()
    sig1 = ScrollingTimeSeriesPanel(telemetry, SIGNAL_CHANNELS, window_s=window_s,
                                    epoch_unix=epoch, index=SIGNAL_1)
    sig2 = ScrollingTimeSeriesPanel(telemetry, SIGNAL_CHANNELS, window_s=window_s,
                                    epoch_unix=epoch, index=SIGNAL_2)

    def path3d():
        return Path3DPanel(telemetry, "3D position + attitude", trail_s=25.0,
                           model=kite_model(), model_size=KITE_MODEL_SIZE,
                           show_triad=SHOW_TRIAD, mount_rpy=KITE_MOUNT_RPY,
                           show_tether=SHOW_TETHER)

    if not with_deformation:
        slots = [
            Slot(VideoPanel(air, f"{airborne.upper()} camera", VIDEO_FIT), 0, 0),
            Slot(path3d(), 0, 1),
            Slot(VideoPanel(gs, "Ground station", VIDEO_FIT), 1, 0),
            Slot(sig1, 1, 1),
        ]
        grid = (2, 2, OUT_SIZE_DEFAULT)
    else:
        points = PointCloudSource(points_df)
        metrics = LIVE_VALUES
        slots = [
            Slot(WireframePanel(points, "front", "Front view", LIMS, invert_x=True), 0, 0),
            Slot(WireframePanel(points, "side", "Side view", LIMS), 0, 1),
            Slot(WireframePanel(points, "bottom", "Bottom view", LIMS, invert_x=True), 0, 2),
            Slot(MetricsPanel(telemetry, metrics, "Live values"), 0, 3),
            Slot(VideoPanel(air, f"{airborne.upper()} camera", VIDEO_FIT), 1, 0, colspan=2),
            Slot(path3d(), 1, 2),
            Slot(sig1, 1, 3),
            Slot(VideoPanel(gs, "Ground station", VIDEO_FIT), 2, 0, colspan=2),
            Slot(Points3DPanel(points, "3D shape", LIMS), 2, 2),
            Slot(sig2, 2, 3),
        ]
        grid = (3, 4, OUT_SIZE_EXTENDED)

    return slots, grid, telemetry, [air, gs]
