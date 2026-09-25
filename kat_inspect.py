#!/usr/bin/env python3
"""
KAT - get to know a flight log: columns, sampling rate and flight_phase
segments. The point is to pick a good value for --flight-start-s.

Usage:
    python kat_inspect.py flightdata\\<file>.csv
"""

import sys
import numpy as np
import pandas as pd

WANTED = [
    "kite_measured_va", "airspeed_apparent_windspeed",
    "kite_actual_depower", "kite_actual_steering",
    "ground_tether_force", "ground_tether_reelout_speed",
    "ground_tether_length",
    "airspeed_angle_of_attack", "airspeed_sideslip_angle",
    "kite_pos_east", "kite_pos_north", "kite_height",
    "kite_0_roll", "kite_0_pitch", "kite_0_yaw",
    "kite_0_latitude", "kite_0_longitude",
    "flight_phase",
]


def main(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"size: {df.shape[0]} rows x {df.shape[1]} columns\n")

    tcol = "time" if "time" in df.columns else df.columns[0]
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    t = t - t[0]
    dt = np.median(np.diff(t))
    print(f"time column  : {tcol}")
    print(f"duration     : {t[-1]:.0f} s  ({t[-1]/60:.1f} min)")
    print(f"sampling     : ~{1/dt:.1f} Hz (dt={dt:.4f} s)\n")

    print("columns we look for:")
    for c in WANTED:
        print(f"   {'YES' if c in df.columns else 'NO '} {c}")

    extra = [c for c in df.columns if c not in WANTED and c != tcol]
    print(f"\nother {len(extra)} columns:")
    print("   " + ", ".join(extra[:40]) + (" ..." if len(extra) > 40 else ""))

    if "flight_phase" in df.columns:
        ph = df["flight_phase"].astype(str).to_numpy()
        change = np.flatnonzero(ph[1:] != ph[:-1]) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(ph)]
        print("\nflight_phase segments longer than 20 s:")
        print(f"   {'start [s]':>14}  {'length [s]':>10}  phase")
        for s, e in zip(starts, ends):
            dur = t[e - 1] - t[s]
            if dur >= 20:
                print(f"   {t[s]:>14.1f}  {dur:>9.1f}  {ph[s]}")
        print("\nPick one of the reel-out rows above for --flight-start-s.")
    else:
        print("\nNo flight_phase column - choose the window from the tether force.")

    if "ground_tether_force" in df.columns:
        f = pd.to_numeric(df["ground_tether_force"], errors="coerce")
        print(f"\nground_tether_force: min {f.min():.1f}  max {f.max():.1f}  "
              f"avg {f.mean():.1f}")
        print("   (above 1000 it is newtons, around 100 it is kilograms)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python kat_inspect.py <flight_log.csv>")
    main(sys.argv[1])
