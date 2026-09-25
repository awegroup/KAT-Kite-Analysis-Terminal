#!/usr/bin/env python3
"""
KAT - independent checks on the data behind the 3D panel.

  1. POSITION : does the |pos| radius agree with ground_tether_length
  2. ATTITUDE : does kite_0_yaw agree with kite_heading / kite_course
  3. PHYSICS  : does tether force scale with apparent wind speed squared
  4. PATTERN  : does force really rise as the elevation angle drops
  5. DRAWING  : do the body axes KAT draws agree with the physics - the nose
                with the direction of travel, the down axis with the tether

Usage:
    python kat_validate.py flightdata\\log_2025-10-09_58-33-00.csv
    python kat_validate.py <csv> 2417 3244        # restrict to this window
    python kat_validate.py <csv> --windows 30     # also fit check 5 per 30 s window
"""

import sys
import numpy as np
import pandas as pd

from kat import repair_time_axis


def col(df, name):
    return pd.to_numeric(df[name], errors="coerce").to_numpy(float) \
        if name in df.columns else None


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def main(path, t_lo=None, t_hi=None, window_s=None):
    df = pd.read_csv(path, low_memory=False)
    t = repair_time_axis(df, "time")
    t = t - t[0]
    if t_lo is not None:
        m = (t >= t_lo) & (t <= t_hi)
        df, t = df.loc[m].reset_index(drop=True), t[m]
        print(f"window under review: {t_lo} - {t_hi} s ({len(t)} samples)\n")

    e, n, h = col(df, "kite_pos_east"), col(df, "kite_pos_north"), col(df, "kite_height")
    L = col(df, "ground_tether_length")
    F = col(df, "ground_tether_force")
    va = col(df, "airspeed_apparent_windspeed")
    yaw = col(df, "kite_0_yaw")

    # --------------------------- 1. position --------------------------
    print("1. POSITION")
    if e is None or L is None:
        print("   required columns missing")
    else:
        r = np.sqrt(e**2 + n**2 + h**2)
        ok = np.isfinite(r) & np.isfinite(L) & (L > 20)
        d = r[ok] - L[ok]
        print(f"   radius   mean {np.nanmean(r[ok]):7.1f} m")
        print(f"   tether   mean {np.nanmean(L[ok]):7.1f} m")
        print(f"   diff     mean {np.nanmean(d):+7.1f} m,  std {np.nanstd(d):.1f} m")
        rel = np.nanmean(np.abs(d)) / np.nanmean(L[ok]) * 100
        print(f"   relative deviation {rel:.1f}%")
        if rel < 10:
            print("   -> position is consistent (tether sag explains a few percent)")
        else:
            print("   -> !! large deviation: position data or units are suspect")

        kd = col(df, "kite_distance")
        if kd is not None:
            dd = r[ok] - kd[ok]
            print(f"   difference from kite_distance: mean {np.nanmean(dd):+.1f} m")

    # --------------------------- 2. attitude --------------------------
    print("\n2. ATTITUDE")
    ref_name = next((c for c in ("kite_heading", "kite_course") if c in df.columns), None)
    if yaw is None or ref_name is None:
        print("   no column available to compare against")
    else:
        ref = col(df, ref_name)
        ok = np.isfinite(yaw) & np.isfinite(ref)
        d = wrap180(yaw[ok] - ref[ok])
        print(f"   kite_0_yaw - {ref_name}:")
        print(f"   mean {np.nanmean(d):+7.1f} deg,  median {np.nanmedian(d):+7.1f} deg")
        print(f"   share with |diff| < 30 deg: {100*np.nanmean(np.abs(d) < 30):.0f}%")

        # circular statistics: plain mean/median mislead on angular data
        ang = np.radians(d)
        R = np.hypot(np.nanmean(np.cos(ang)), np.nanmean(np.sin(ang)))
        cmean = np.degrees(np.arctan2(np.nanmean(np.sin(ang)), np.nanmean(np.cos(ang))))
        print(f"   circular mean: {cmean:+.1f} deg,  consistency R = {R:.2f}")
        print("   (R near 1: constant offset. R near 0: unrelated or bimodal)")

        print("   distribution:")
        hist, edges = np.histogram(d, bins=12, range=(-180, 180))
        for c, lo_e in zip(hist, edges[:-1]):
            bar = "#" * int(40 * c / max(hist.max(), 1))
            print(f"     {lo_e:+5.0f}..{lo_e+30:+5.0f}  {bar}")

        print("   candidate corrections (median of the remaining deviation):")
        for off in (0, 90, -90, 180):
            rem = np.abs(wrap180(d - off))
            print(f"     {off:+4d} deg -> median |diff| {np.nanmedian(rem):5.1f} deg, "
                  f"{100*np.nanmean(rem < 30):3.0f}% within 30 deg")

        if R > 0.7 and abs(cmean) > 25:
            print("   -> a constant offset is present; pass the best candidate")
            print("      to Path3DPanel as yaw_offset_deg")
        elif R < 0.4:
            print("   -> !! no consistent relation; yaw and heading may measure")
            print("      different things - the convention needs to be confirmed")

    # ------------------ 2b. comparison with the velocity ---------------
    print("\n2b. YAW vs VELOCITY VECTOR (NED, no convention ambiguity)")
    vx, vy = col(df, "kite_0_vx"), col(df, "kite_0_vy")
    if yaw is None or vx is None or vy is None:
        print("   kite_0_vx / kite_0_vy missing")
    else:
        spd = np.hypot(vx, vy)
        for lbl, crs in (("atan2(vy,vx)  [x=north]", np.degrees(np.arctan2(vy, vx))),
                         ("atan2(vx,vy)  [x=east ]", np.degrees(np.arctan2(vx, vy)))):
            ok = np.isfinite(yaw) & np.isfinite(crs) & (spd > 5)
            if ok.sum() < 100:
                continue
            d = wrap180(yaw[ok] - crs[ok])
            ang = np.radians(d)
            R = np.hypot(np.nanmean(np.cos(ang)), np.nanmean(np.sin(ang)))
            cm = np.degrees(np.arctan2(np.nanmean(np.sin(ang)), np.nanmean(np.cos(ang))))
            print(f"   {lbl}: R = {R:.2f}, circular mean {cm:+6.1f} deg, "
                  f"{100*np.nanmean(np.abs(wrap180(d - cm)) < 30):3.0f}% within +-30")
        print("   the row with the higher R shows the correct axis order;")
        print("   its circular mean can be used as yaw_offset_deg")

    # -------------- 2c. does the deviation follow position -------------
    print("\n2c. DOES THE DEVIATION DEPEND ON KITE POSITION")
    az = col(df, "kite_azimuth")
    if yaw is None or ref_name is None or az is None:
        print("   kite_azimuth missing")
    else:
        ref = col(df, ref_name)
        ok = np.isfinite(yaw) & np.isfinite(ref) & np.isfinite(az)
        d = wrap180(yaw[ok] - ref[ok])
        a = az[ok]
        if np.nanmax(np.abs(a)) < 3.2:
            a = np.degrees(a)
        c = np.corrcoef(a, d)[0, 1]
        print(f"   correlation of (yaw - {ref_name}) with kite_azimuth: {c:+.2f}")
        if abs(c) > 0.5:
            print("   -> the deviation varies with the kite position in the sky:")
            print(f"      {ref_name} is probably defined in the wind reference frame")
            print("      and cannot be compared directly with yaw in NED")
        else:
            print("   -> position does not explain it; something else differs")

    # ---------------------------- 3. physics ---------------------------
    print("\n3. PHYSICS (force ~ speed^2)")
    if F is None or va is None:
        print("   required columns missing")
    else:
        ok = np.isfinite(F) & np.isfinite(va) & (va > 3) & (F > 5)
        if ok.sum() > 100:
            p = np.polyfit(np.log(va[ok]), np.log(F[ok]), 1)
            c = np.corrcoef(va[ok]**2, F[ok])[0, 1]
            print(f"   log-log slope (exponent): {p[0]:.2f}   (theory 2.0)")
            print(f"   correlation of F with va^2: {c:.2f}")
            if 1.3 < p[0] < 2.7:
                print("   -> force scales with speed squared, as expected")
            else:
                print("   -> !! exponent far from the expected value")

    # ---------------------------- 4. pattern ---------------------------
    print("\n4. DOES FORCE RISE WHILE THE KITE DESCENDS")
    if h is None or F is None:
        print("   required columns missing")
    else:
        el = np.degrees(np.arctan2(h, np.sqrt(e**2 + n**2)))
        ok = np.isfinite(el) & np.isfinite(F) & (h > 30)
        if ok.sum() > 100:
            c = np.corrcoef(el[ok], F[ok])[0, 1]
            lo = np.nanpercentile(el[ok], 25)
            hi = np.nanpercentile(el[ok], 75)
            f_lo = np.nanmean(F[ok][el[ok] <= lo])
            f_hi = np.nanmean(F[ok][el[ok] >= hi])
            print(f"   elevation angle vs force correlation: {c:+.2f}")
            print(f"   low elevation (<{lo:.0f} deg) mean force: {f_lo:7.1f}")
            print(f"   high elevation (>{hi:.0f} deg) mean force: {f_hi:7.1f}")
            if f_lo > f_hi:
                print("   -> force is higher down low: apparent wind is strongest")
                print("      at the centre of the wind window. Expected.")
            else:
                print("   -> the opposite showed up, which is unexpected")

    check_drawing(df, t, window_s)


def check_drawing(df, t, window_s=None):
    """Check 5: the axes exactly as Path3DPanel draws them, against physics."""
    from kat import Path3DPanel, mount_matrix
    print("\n5. DRAWN BODY AXES vs PHYSICS")
    need = ["kite_pos_east", "kite_pos_north", "kite_height",
            "kite_0_roll", "kite_0_pitch", "kite_0_yaw"]
    if any(c not in df.columns for c in need):
        print("   required columns missing")
        return
    P = np.column_stack([col(df, c) for c in need[:3]])
    A = np.column_stack([col(df, c) for c in need[3:]])
    ok = np.all(np.isfinite(P), 1) & np.all(np.isfinite(A), 1) & (P[:, 2] > 30)
    P, A, tt = P[ok], A[ok], t[ok]
    if len(P) < 100:
        print("   too few airborne samples")
        return
    idx = np.linspace(0, len(P) - 1, min(len(P), 4000)).astype(int)
    V = np.gradient(P, tt, axis=0)[idx]
    Pi = P[idx]
    B = np.array([Path3DPanel._body_axes_enu(*A[i]) for i in idx])   # sensor frame, ENU

    # the frame physics expects: down along the tether, nose along the travel
    z = -Pi / np.linalg.norm(Pi, axis=1, keepdims=True)
    x = V - np.sum(V * z, 1, keepdims=True) * z
    n = np.linalg.norm(x, axis=1, keepdims=True)
    good = (n[:, 0] > 1e-6)
    x = x / np.where(n > 1e-6, n, 1.0)
    y = np.cross(z, x)
    Bp = np.stack([x, y, z], axis=1)[good]
    B, Pi, V = B[good], Pi[good], V[good]

    def angles(Bx):
        return [np.degrees(np.arccos(np.clip(np.sum(Bx[:, i] * Bp[:, i], 1), -1, 1)))
                for i in range(3)]

    def fit_rotation(Bs, Bt):
        """The single rotation that best maps frame Bs onto frame Bt."""
        M = np.einsum("nij,nkj->ik", Bt, Bs) / len(Bs)
        U, _, Vt = np.linalg.svd(M)
        M = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        pitch = np.degrees(-np.arcsin(np.clip(M[2, 0], -1, 1)))
        yaw = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        roll = np.degrees(np.arctan2(M[2, 1], M[2, 2]))
        return (-roll, -pitch, -yaw)          # in mount_matrix convention

    def report(label, Bx):
        a_nose, a_right, a_down = angles(Bx)
        print(f"   {label:22s} nose {np.median(a_nose):5.1f}   right wing {np.median(a_right):5.1f}"
              f"   down {np.median(a_down):5.1f}  [median deg]")
        return np.median(a_nose) + np.median(a_right) + np.median(a_down)

    report("as logged:", B)

    fit = fit_rotation(B, Bp)
    after = report("best fit applied:", np.einsum("ij,njk->nik", mount_matrix(*fit), B))

    try:
        import config as _cfg
        cur = tuple(float(v) for v in getattr(_cfg, "KITE_MOUNT_RPY", (0.0, 0.0, 0.0)))
    except Exception:
        cur = (0.0, 0.0, 0.0)
    if any(cur):
        report("config setting now:", np.einsum("ij,njk->nik", mount_matrix(*cur), B))

    print(f"\n   KITE_MOUNT_RPY = ({fit[0]:.1f}, {fit[1]:.1f}, {fit[2]:.1f})"
          "      # roll, pitch, yaw [deg]")
    if after / 3 < 12:
        print("   -> a single fixed rotation explains the difference: the sensor frame")
        print("      is mounted at that angle relative to the kite's own axes")
    else:
        print("   -> a fixed rotation does not fully explain it; what remains is")
        print("      either real (angle of attack, sideslip) or a deeper mismatch")
    print("   Note: the reference itself is approximate - a kite does not fly exactly")
    print("   along its tether or exactly into its velocity, so a few degrees remain.")

    if window_s:
        tw = tt[idx][good]
        print(f"\n   The same fit over {window_s:.0f} s windows"
              " (to see whether the offset is really fixed):")
        print(f"   {'start [s]':>10} {'roll':>8} {'pitch':>8} {'yaw':>8}")
        rows = []
        w0 = tw[0]
        while w0 < tw[-1]:
            m = (tw >= w0) & (tw < w0 + window_s)
            if m.sum() > 50:
                r = fit_rotation(B[m], Bp[m])
                rows.append((w0, *r))
                print(f"   {w0:10.0f} {r[0]:8.1f} {r[1]:8.1f} {r[2]:8.1f}")
            w0 += window_s
        if len(rows) > 2:
            arr = np.array(rows)
            sd = arr[:, 1:].std(axis=0)
            print(f"   {'spread':>10} {sd[0]:8.1f} {sd[1]:8.1f} {sd[2]:8.1f}   [std over windows]")
            span = arr[0, 0] - arr[-1, 0]
            for j, axis in enumerate(("roll", "pitch", "yaw")):
                y = arr[:, 1 + j]
                if sd[j] < 2.0:
                    print(f"   {axis:>10}: steady, a fixed mounting angle covers it")
                    continue
                slope = np.polyfit(arr[:, 0], y, 1)[0]
                trend = slope * abs(span)
                resid = y - np.polyval(np.polyfit(arr[:, 0], y, 1), arr[:, 0])
                straight = resid.std() < 0.4 * sd[j]
                if straight and abs(trend) > 3 * resid.std():
                    print(f"   {axis:>10}: drifts steadily, {trend:+.1f} deg over the flight")
                else:
                    print(f"   {axis:>10}: moves about without a trend")
            med = tuple(np.median(arr[:, 1:], axis=0))
            r_med = report("median of windows:", np.einsum("ij,njk->nik", mount_matrix(*med), B))
            print(f"\n   KITE_MOUNT_RPY = ({med[0]:.1f}, {med[1]:.1f}, {med[2]:.1f})"
                  "      # robust: median of the windows")
            if r_med < after:
                print("   -> prefer this one: the single global fit is pulled by the few")
                print("      windows where the kite really was at a different attitude")

            print("\n   Read it this way: a steady axis is a mounting angle and belongs in")
            print("   KITE_MOUNT_RPY. A steady drift is most likely sensor drift, and can be")
            print("   corrected over time. Scatter is the kite's own changing attitude - the")
            print("   very thing the panel should show, so it must NOT be corrected away.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    args = sys.argv[1:]
    window = None
    if "--windows" in args:                       # e.g. --windows 30
        i = args.index("--windows")
        window = float(args[i + 1]) if len(args) > i + 1 else 30.0
        del args[i:i + 2]
    if len(args) >= 3:
        main(args[0], float(args[1]), float(args[2]), window)
    else:
        main(args[0], window_s=window)
