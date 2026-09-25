#!/usr/bin/env python3
"""
KAT - tools for finding the video-to-telemetry offset.

Two commands:

  meta    : creation time, fps and duration from the camera metadata
            python kat_sync.py meta KCU.MP4 GS.MP4

  anchor  : scan offset candidates using labelled static frames
            python kat_sync.py anchor flight.csv

Note: cross-correlating image motion energy with telemetry (the former
"motion" / "align" / "profile" commands) returned the wrong answer twice on
this flight. A pumping flight is periodic, so the correlation curve carries
several peaks of almost equal height. The reliable way to find the offset is
to locate take-off in both the video and the telemetry and take the
difference (see VIDEO_TAKEOFF_S / TELEMETRY_TAKEOFF_S in config.py).
"""

import json
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd


# ------------------------------- meta -------------------------------

def cmd_meta(paths):
    for p in paths:
        print(f"\n=== {p} ===")
        cap = cv2.VideoCapture(p)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"resolution : {w}x{h}")
            print(f"fps        : {fps:.6f}   <- 29.97 or 30? this matters")
            print(f"frame count: {n}")
            print(f"duration   : {n/fps:.1f} s  ({n/fps/60:.1f} min)")
            cap.release()
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", p],
                capture_output=True, text=True, timeout=60).stdout
            info = json.loads(out)
            tags = info.get("format", {}).get("tags", {}) or {}
            for k in ("creation_time", "com.apple.quicktime.creationdate",
                      "date", "firmware", "model"):
                if k in tags:
                    print(f"{k:11}: {tags[k]}")
            for st in info.get("streams", []):
                t = (st.get("tags") or {})
                if st.get("codec_type") == "data" or "gpmd" in str(t).lower():
                    print(f"GPMF/data stream present (GPS time recoverable): "
                          f"stream #{st.get('index')}")
        except Exception as e:
            print(f"could not read ffprobe: {e}")
    print("\nIf creation_time is UTC: offset = video start - telemetry start")


# ------------------------------ anchor ------------------------------

def _flight_window(fd, tt):
    """The actual flight window in the telemetry: the 'pp-*' phases."""
    if "flight_phase" not in fd.columns:
        return None, None
    ph = pd.Series(fd["flight_phase"]).astype(str)
    fly = np.flatnonzero(ph.str.startswith("pp-").to_numpy())
    if len(fly) == 0:
        return None, None
    return float(tt[fly[0]]), float(tt[fly[-1]])



# taken from the static_test_output file names: frame number -> expected phase
ANCHORS = [
    (7182, "ro", "straight_flight_reelout"),
    (7362, "ro", "right_turn_reelout"),
    (7721, "ro", "straight_flight_reelout"),
    (7811, "ro", "left_turn_reelout"),
    (17372, "ri", "powered_state_reelin"),
    (17611, "ri", "depowered_state_reelin"),
    (17701, "ri", "depowered_state_reelin"),
]


def cmd_anchor(flight_csv, fps=30000.0 / 1001.0, step=0.2, top=6):
    """Find KCU offset candidates from the labelled static frames.

    Each file name carries both a frame number and a manoeuvre. The correct
    offset is the one that places all seven frames in the right flight_phase.
    """
    from kat import repair_time_axis

    fd = pd.read_csv(flight_csv, low_memory=False)
    tt = repair_time_axis(fd, "time")
    tt = tt - tt[0]
    if "flight_phase" not in fd.columns:
        sys.exit("no flight_phase column")
    ph = pd.Series(fd["flight_phase"]).astype(str).to_numpy()

    print(f"fps = {fps:.5f}")
    print("anchor frames -> video time:")
    for n, kind, name in ANCHORS:
        print(f"   {n:>6} -> {n/fps:8.2f} s   ({kind}, {name})")

    t_fly0, t_fly1 = _flight_window(fd, tt)
    v_times = np.array([n / fps for n, _, _ in ANCHORS])
    kinds = [k for _, k, _ in ANCHORS]

    # scan T0 so that every anchor falls inside the flight window
    lo = v_times.max() - t_fly1
    hi = v_times.min() - t_fly0
    cands = np.arange(lo, hi + step, step)

    results = []
    for T0 in cands:
        score = 0
        for v, k in zip(v_times, kinds):
            mt = v - T0
            i = int(np.searchsorted(tt, mt))
            if i <= 0 or i >= len(tt):
                continue
            p = ph[i]
            if p.startswith("pp-ro") and k == "ro":
                score += 1
            elif p.startswith("pp-ri") and k == "ri":
                score += 1
        results.append((score, T0))

    results.sort(key=lambda r: (-r[0], r[1]))
    best = results[0][0]
    print(f"\nbest score: {best}/{len(ANCHORS)}")

    # merge adjacent T0 values that share the best score into ranges
    groups = []
    for sc, T0 in sorted([r for r in results if r[0] == best], key=lambda r: r[1]):
        if groups and T0 - groups[-1][1] <= step * 1.5:
            groups[-1][1] = T0
        else:
            groups.append([T0, T0])
    print(f"{len(groups)} separate T0 ranges reach this score:")
    for a, b in groups[:top]:
        print(f"   T0 = {a:+9.1f} .. {b:+9.1f}   (centre {0.5*(a+b):+.1f}, width {b-a:.1f} s)")

    if len(groups) > 1:
        print("\nSeveral candidates - expected, because the pumping cycle is")
        print("periodic. Separate them by checking the candidates visually.")
    return groups


# -------------------------------- cli -------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "meta":
        cmd_meta(sys.argv[2:])
    elif cmd == "anchor":
        cmd_anchor(sys.argv[2])
    else:
        sys.exit(__doc__)
