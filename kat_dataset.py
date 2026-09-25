#!/usr/bin/env python3
"""
KAT - build a synthetic TIME SERIES in complete-dataset format from the
REAL point cloud stored in static_test_output/*.csv.

Real      : the kite geometry (marker positions, span, camber, scale)
Synthetic : the motion over time (rotation + twist + noise)
Optional  : telemetry filled with real values from the flight log CSV

Usage:
    python kat_dataset.py
    python kat_dataset.py --flight-csv "path/to/flight_log.csv"
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ----------------------------- settings ----------------------------
STATIC_CSV = os.path.join(
    "Photogrammetry", "static_experiments", "static_test_output",
    "straight_flight_reelout_frame_7182.csv",
)
OUT_CSV = os.path.join("output", "synthetic_complete_dataset.csv")

N_FRAMES = 300       # ~10 s @ 30 Hz
FPS = 30.0
NOISE_STD = 0.004    # marker detection noise [m]

# motion amplitudes (deg)
ROLL_AMP, PITCH_AMP, YAW_AMP, TWIST_AMP = 4.0, 3.0, 2.0, 2.5

# column names in the flight log -> names the panels expect
FLIGHT_RENAME = {"airspeed_apparent_windspeed": "kite_measured_va"}

# telemetry columns the panels look for
TELEMETRY_COLS = [
    "kite_measured_va", "kite_actual_depower", "kite_actual_steering",
    "ground_tether_force", "ground_tether_reelout_speed",
    "airspeed_angle_of_attack", "airspeed_sideslip_angle",
    "kite_0_latitude", "kite_0_longitude", "ground_tether_length",
    "uwb_distance_m",
    "kite_pos_east", "kite_pos_north", "kite_height",
    "kite_0_roll", "kite_0_pitch", "kite_0_yaw",
]

G0 = 9.80665
rng = np.random.default_rng(42)


# --------------------------- real geometry -------------------------
def geometry_from_model(obj_path, n_struts=8, n_per_strut=6, n_le=13):
    """Marker positions sampled from the kite's CAD model, in metres.

    Used for the demonstration dataset shipped with the repository, so that
    the deformation panels can be shown without the photogrammetry data,
    which is not public. The layout imitates the real marker set: one row
    along the leading edge and one line of markers along each strut.
    """
    from kat import load_obj
    V, _ = load_obj(obj_path, "-x +y -z", max_faces=10 ** 9)
    V = V * 8.32                                  # normalised span -> metres
    # KAT's deformation panels expect x across the span, z upward
    P = np.column_stack([V[:, 1], V[:, 0], -V[:, 2]])
    P[:, 2] += 6.0                                # sit the wing at a plausible height

    geom = {}
    stations = np.linspace(P[:, 0].min(), P[:, 0].max(), n_struts + 2)[1:-1]
    half = 0.5 * (stations[1] - stations[0])
    for i, xs in enumerate(stations):
        band = P[np.abs(P[:, 0] - xs) < half * 0.6]
        if len(band) < n_per_strut:
            continue
        order = np.argsort(band[:, 1])            # along the chord
        idx = np.linspace(0, len(order) - 1, n_per_strut).astype(int)
        geom[str(i)] = band[order][idx]

    le = []
    for xs in np.linspace(P[:, 0].min(), P[:, 0].max(), n_le):
        band = P[np.abs(P[:, 0] - xs) < half]
        if len(band):
            le.append(band[np.argmax(band[:, 1])])   # furthest forward = leading edge
    geom["LE"] = np.array(le)
    print(f"geometry sampled from {os.path.basename(obj_path)}: "
          f"{sum(len(v) for v in geom.values())} markers in {len(geom)} groups")
    return geom


def resolve_static(path):
    """Locate the static point-cloud CSV, or explain how to get it."""
    if os.path.exists(path):
        return path
    folder = os.path.dirname(path) or "."
    if os.path.isdir(folder):
        found = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
        if found:
            print(f"[kat] {os.path.basename(path)} is not in {folder}. Available there:")
            for f in found:
                print(f"      {f}")
            print("\n      Pick one with --static-csv <path>.")
            raise SystemExit(1)
    raise SystemExit(
        f"[kat] static point cloud not found: {path}\n\n"
        "      This file is not part of the KAT repository. It comes from the\n"
        "      photogrammetry work, in\n"
        "          Photogrammetry/static_experiments/static_test_output/\n"
        "      of pimjhaanen/photogrammetry_thesis. Copy that folder next to\n"
        "      this script, or point at it directly:\n\n"
        "          python kat_dataset.py --static-csv <a frame csv>"
        " --flight-csv <log> --flight-start-s 2417\n\n"
        "      The file holds one frozen frame of marker positions: columns\n"
        "      group, idx_in_group, x, y, z. KAT only uses it for the shape;\n"
        "      the motion over time is synthesised.")


def load_geometry(path):
    """static CSV -> {marker_id: (N,3)}. Maps the 'group' column to marker_id."""
    df = pd.read_csv(path)
    for col in ("group", "x", "y", "z"):
        if col not in df.columns:
            raise ValueError(f"column '{col}' not found. Available: {df.columns.tolist()}")

    if "idx_in_group" in df.columns:
        df = df.sort_values(["group", "idx_in_group"])

    groups = list(dict.fromkeys(df["group"].astype(str)))
    le_names = [g for g in groups if g.upper() == "LE"]
    others = [g for g in groups if g.upper() != "LE"]

    mapping = {g: "LE" for g in le_names}
    for i, g in enumerate(others):
        # keep 0..7 as they are, otherwise number them in order
        mapping[g] = g if g.isdigit() and int(g) < 8 else str(i)

    print("group -> marker_id mapping:")
    for g in groups:
        n = (df["group"].astype(str) == g).sum()
        print(f"   {g!r:>6} -> {mapping[g]!r:>5}  ({n} points)")

    geom = {}
    for g in groups:
        pts = df.loc[df["group"].astype(str) == g, ["x", "y", "z"]].to_numpy(float)
        mid = mapping[g]
        geom[mid] = np.vstack([geom[mid], pts]) if mid in geom else pts
    return geom


def sanity_check(geom):
    """Simulate the isolated-point filter (0.8 m) used downstream."""
    P = np.vstack(list(geom.values()))
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    frac = (D.min(axis=1) <= 0.8).mean()
    span = np.ptp(P[:, 0])
    print(f"\n{len(P)} points in total | x span {span:.2f} m "
          f"| z range {P[:,2].min():.2f}-{P[:,2].max():.2f} m")
    print(f"passing the 0.8 m isolated-point filter: {100*frac:.1f}%")
    if frac < 1.0:
        print("   !! some points would be dropped - check the matching radius")
    if "3" not in geom or "4" not in geom:
        print("   !! struts 3/4 missing - the reference frame cannot be built")
    return P.mean(axis=0)


# ------------------------------ motion -----------------------------
def rot_xyz(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def deform(P, t, centre, half_span):
    """Rigid rotation + twist growing linearly along the span + noise."""
    twist = np.deg2rad(TWIST_AMP) * np.sin(2 * np.pi * 0.35 * t)
    frac = (P[:, 0] - centre[0]) / max(half_span, 1e-9)

    Q = P.copy()
    for i in range(Q.shape[0]):
        Q[i] = centre + rot_xyz(0.0, twist * frac[i], 0.0) @ (Q[i] - centre)

    R = rot_xyz(
        np.deg2rad(ROLL_AMP) * np.sin(2 * np.pi * 0.25 * t),
        np.deg2rad(PITCH_AMP) * np.sin(2 * np.pi * 0.15 * t + 0.6),
        np.deg2rad(YAW_AMP) * np.sin(2 * np.pi * 0.10 * t),
    )
    Q = centre + (Q - centre) @ R.T
    return Q + rng.normal(0.0, NOISE_STD, Q.shape)


# ---------------------------- telemetry ----------------------------
def kite_pose(t, radius=250.0):
    """A synthetic figure-eight trajectory with matching Euler angles.

    Position: ENU [m], centred on the ground station
    Angles  : NED->FRD, 3-2-1 (yaw, pitch, roll) [deg]
    """
    w = 2 * np.pi * 0.04                      # a figure eight of about 25 s
    azim = 0.50 * np.sin(w * t)               # lateral swing [rad]
    elev = 0.55 + 0.14 * np.sin(2 * w * t)    # lower/upper arms of the eight [rad]

    east = radius * np.cos(elev) * np.sin(azim)
    north = radius * np.cos(elev) * np.cos(azim)
    height = radius * np.sin(elev)

    # yaw: direction of the velocity vector, by finite difference
    dt = 0.05
    a2 = 0.50 * np.sin(w * (t + dt))
    e2 = 0.55 + 0.14 * np.sin(2 * w * (t + dt))
    de = radius * np.cos(e2) * np.sin(a2) - east
    dn = radius * np.cos(e2) * np.cos(a2) - north
    yaw = np.degrees(np.arctan2(de, dn))

    # roll: bank proportional to the turn rate
    a3 = 0.50 * np.sin(w * (t + 2 * dt))
    e3 = 0.55 + 0.14 * np.sin(2 * w * (t + 2 * dt))
    yaw2 = np.degrees(np.arctan2(
        radius * np.cos(e3) * np.sin(a3) - radius * np.cos(e2) * np.sin(a2),
        radius * np.cos(e3) * np.cos(a3) - radius * np.cos(e2) * np.cos(a2)))
    rate = (yaw2 - yaw + 180) % 360 - 180
    roll = float(np.clip(rate * 1.2, -55, 55))

    pitch = 6.0 + 3.0 * np.sin(2 * w * t)
    return east, north, height, roll, pitch, float(yaw)


def synthetic_telemetry(t, span):
    return {
        "kite_measured_va": 21.0 + 3.0 * np.sin(2 * np.pi * 0.12 * t),
        "kite_actual_depower": 32.0 + 5.0 * np.sin(2 * np.pi * 0.08 * t),
        "kite_actual_steering": 10.0 * np.sin(2 * np.pi * 0.25 * t),
        "ground_tether_force": 2600.0 + 400.0 * np.sin(2 * np.pi * 0.10 * t),  # N
        "ground_tether_reelout_speed": 1.8 + 0.6 * np.sin(2 * np.pi * 0.07 * t),
        "airspeed_angle_of_attack": 8.0 + 2.5 * np.sin(2 * np.pi * 0.15 * t + 0.6),
        "airspeed_sideslip_angle": 1.5 * np.sin(2 * np.pi * 0.20 * t),
        "kite_0_latitude": 54.1265 + 0.0009 * np.sin(2 * np.pi * 0.05 * t),
        "kite_0_longitude": -9.7811 + 0.0012 * np.cos(2 * np.pi * 0.05 * t),
        "ground_tether_length": 230.0 + 12.0 * np.sin(2 * np.pi * 0.04 * t),
        "uwb_distance_m": span + 0.03 * np.sin(2 * np.pi * 0.30 * t),
        **dict(zip(("kite_pos_east", "kite_pos_north", "kite_height",
                    "kite_0_roll", "kite_0_pitch", "kite_0_yaw"), kite_pose(t))),
    }


def real_telemetry(flight_csv, n_frames, fps, start_s, span):
    """Take an N_FRAMES window from the flight log and resample it to 30 Hz."""
    fd = pd.read_csv(flight_csv).rename(columns=FLIGHT_RENAME)
    if "time" not in fd.columns:
        raise ValueError("no 'time' column - is this the right flight CSV?")

    from kat import repair_time_axis
    t_rel = repair_time_axis(fd, "time")
    t_rel -= t_rel[0]
    t_new = start_s + np.arange(n_frames) / fps
    if t_new[-1] > t_rel[-1]:
        raise ValueError(f"window falls outside the data (max {t_rel[-1]:.0f} s)")

    out = {}
    for col in TELEMETRY_COLS:
        if col in fd.columns:
            out[col] = np.interp(t_new, t_rel, pd.to_numeric(fd[col], errors="coerce")
                                 .ffill().bfill().to_numpy(float))
        else:
            out[col] = np.full(n_frames, np.nan)

    # tether force is logged in kilograms - kept as recorded

    # UWB distance comes from the photogrammetry side; the flight log has none
    out["uwb_distance_m"] = span + 0.03 * np.sin(2 * np.pi * 0.30 * (t_new - start_s))

    missing = [c for c in TELEMETRY_COLS if c not in fd.columns and c != "uwb_distance_m"]
    if missing:
        print(f"   not found in the flight CSV, left as NaN: {missing}")
    return pd.DataFrame(out)


# ------------------------------ main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-csv", default=STATIC_CSV)
    ap.add_argument("--from-model", default=None,
                    help="sample the markers from a kite .obj instead of a static CSV "
                         "(used for the demonstration dataset; no photogrammetry needed)")
    ap.add_argument("--flight-csv", default=None,
                    help="flight log CSV; without it the telemetry is synthetic")
    ap.add_argument("--flight-start-s", type=float, default=600.0,
                    help="second of the flight log to start from")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    if args.flight_csv and not os.path.exists(args.flight_csv):
        raise SystemExit(f"[kat] flight log not found: {args.flight_csv}\n"
                         "      give the path to the log CSV, e.g.\n"
                         "          --flight-csv flightdata/log_2025-10-09_58-33-00.csv\n"
                         "      or leave --flight-csv out to use synthetic telemetry.")
    geom = (geometry_from_model(args.from_model) if args.from_model
            else load_geometry(resolve_static(args.static_csv)))
    centre = sanity_check(geom)
    all_pts = np.vstack(list(geom.values()))
    span = float(np.ptp(all_pts[:, 0]))
    half_span = span / 2.0

    if args.flight_csv:
        print(f"\ntelemetry: real ({os.path.basename(args.flight_csv)})")
        tel = real_telemetry(args.flight_csv, args.frames, FPS, args.flight_start_s, span)
    else:
        print("\ntelemetry: synthetic (pass --flight-csv for the real thing)")
        tel = pd.DataFrame([synthetic_telemetry(f / FPS, span) for f in range(args.frames)])

    t0 = datetime(2025, 10, 9, 17, 40, 0, tzinfo=timezone.utc)
    frames = []
    for f in range(args.frames):
        t = f / FPS
        row_tel = tel.iloc[f].to_dict()
        utc = (t0 + timedelta(seconds=t)).strftime("%H:%M:%S.%f")[:-3]
        for marker_id, pts in geom.items():
            moved = deform(pts, t, centre, half_span)
            blk = pd.DataFrame(moved, columns=["x", "y", "z"])
            blk["frame_idx"] = f
            blk["marker_id"] = marker_id
            blk["timestamp"] = t
            blk["utc"] = utc
            for k, v in row_tel.items():
                blk[k] = v
            frames.append(blk)

    df = pd.concat(frames, ignore_index=True)
    df = df[["frame_idx", "marker_id", "x", "y", "z", "timestamp", "utc"] + TELEMETRY_COLS]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n[+] {args.out}  ({len(df)} rows, {args.frames} frames)")
    print("\nsuggested axis limits for config.py LIMS:")
    for ax, i in (("X_LIM", 0), ("Y_LIM", 1), ("Z_LIM", 2)):
        lo, hi = all_pts[:, i].min(), all_pts[:, i].max()
        pad = 0.15 * (hi - lo)
        print(f"   {ax} = ({lo-pad:.1f}, {hi+pad:.1f})")


if __name__ == "__main__":
    main()
