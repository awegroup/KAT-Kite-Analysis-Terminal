#!/usr/bin/env python3
"""
KAT - find the second at which the kite leaves the ground, in a video and in
the flight log. Those two numbers are the whole synchronisation.

Why this and not a correlation? Cross-correlating image motion with the
telemetry was tried twice on this project and gave the wrong answer both
times: a pumping flight is periodic, so the correlation curve has several
peaks of nearly equal height. Take-off happens once.

In the flight log (prints the first moment the kite climbs past a height):

    python tools/find_takeoff.py log flightdata/log.csv

In a video (writes evenly spaced frames to a folder, look at them and
narrow the range down; repeat until one second is left):

    python tools/find_takeoff.py video output/kcu_proxy.mp4 --from 0 --to 600
    python tools/find_takeoff.py video output/kcu_proxy.mp4 --from 150 --to 190

Then put the results in flights/<flight>.py:

    TELEMETRY_TAKEOFF_S   the log value
    VIDEOS[...]["takeoff_s"]   the video value, per camera
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd


def from_log(path, height_col="kite_height", threshold=30.0, land=True):
    df = pd.read_csv(path, low_memory=False)
    if height_col not in df.columns:
        sys.exit(f"no column {height_col}; available height-like columns: "
                 + ", ".join(c for c in df.columns if "height" in c.lower()))
    sys.path.insert(0, os.getcwd())
    from kat import repair_time_axis                       # same time repair as KAT
    t = repair_time_axis(df, "time")
    t = t - t[0]
    h = pd.to_numeric(df[height_col], errors="coerce").to_numpy(float)
    above = np.isfinite(h) & (h > threshold)
    if not above.any():
        sys.exit(f"the kite never rises above {threshold} m")
    first, last = np.flatnonzero(above)[[0, -1]]
    print(f"log covers {t[-1]:.0f} s, {len(t)} rows")
    print(f"first above {threshold:.0f} m : {t[first]:8.1f} s   <- TELEMETRY_TAKEOFF_S")
    if land:
        print(f"last  above {threshold:.0f} m : {t[last]:8.1f} s   <- TELEMETRY_LANDING_S")
        print(f"airborne             : {t[last] - t[first]:8.1f} s")
    print("\nCheck the value against the video: at that instant the kite should")
    print("just be leaving the ground in every camera.")


def from_video(path, t_from, t_to, count, out_dir):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps if fps else 0.0
    t_to = min(t_to, dur)
    print(f"{os.path.basename(path)}: {fps:.4f} fps, {n} frames, {dur:.0f} s")
    if t_from >= t_to:
        sys.exit("empty range")

    os.makedirs(out_dir, exist_ok=True)
    times = np.linspace(t_from, t_to, count)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)))
        cv2.putText(frame, f"t = {t:.2f} s", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"t{t:08.2f}.jpg"), frame)
    cap.release()
    step = times[1] - times[0] if len(times) > 1 else 0.0
    print(f"wrote {len(times)} frames to {out_dir}, one every {step:.2f} s")
    print("Open them, find the last frame still on the ground and the first in")
    print("the air, then run again over that shorter range until the gap is ~1 s.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    lg = sub.add_parser("log", help="take-off and landing from the flight log")
    lg.add_argument("csv")
    lg.add_argument("--column", default="kite_height")
    lg.add_argument("--threshold", type=float, default=30.0)

    vd = sub.add_parser("video", help="export frames to look through")
    vd.add_argument("video")
    vd.add_argument("--from", dest="t_from", type=float, default=0.0)
    vd.add_argument("--to", dest="t_to", type=float, default=600.0)
    vd.add_argument("--count", type=int, default=20)
    vd.add_argument("--out-dir", default="output/takeoff_frames")

    args = ap.parse_args()
    if args.mode == "log":
        from_log(args.csv, args.column, args.threshold)
    else:
        from_video(args.video, args.t_from, args.t_to, args.count, args.out_dir)


if __name__ == "__main__":
    main()
