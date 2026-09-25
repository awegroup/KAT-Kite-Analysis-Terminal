# Pipeline — putting a new flight into KAT

Five measured numbers turn a set of raw files into a synchronised view. This
document is the procedure and, more importantly, the reasoning: every step
below exists because the obvious alternative failed on real data.

The full design notes — architecture, the pitfalls in detail, the validation
results — are in [design.md](design.md).

---

## 1. Get to know the flight log

```bash
python kat_inspect.py flightdata/<log>.csv
```

Prints the columns, the sampling rate, the `flight_phase` segments and the
range of the tether force.

Three things to read carefully:

- **The sampling rate may look wrong.** For the reference flight the tool
  reports an infinite rate, because the `time` column holds whole seconds
  while about ten rows share each second. The file is fine; KAT spreads those
  rows evenly across the second (`repair_time_axis`).
- **The force unit.** Around 100 means kilograms, above 1000 means newtons.
  KAT keeps the column as recorded and labels it; it never converts silently.
- **`flight_phase` does not cover the whole flight.** The `pp-*` segments
  (power production) spanned 827 s while the kite was airborne for 1722 s.
  `unknown` means "not pumping", not "on the ground".

## 2. Take-off and landing

```bash
python scripts/find_takeoff.py log flightdata/<log>.csv
```

The first and last moment `kite_height` exceeds 30 m. These two numbers anchor
everything else, so take them from the height column, never from
`flight_phase`.

→ `TELEMETRY_TAKEOFF_S`, `TELEMETRY_LANDING_S`

## 3. Video proxies

```bash
python scripts/make_proxies.py raw/KCU_1.MP4 raw/Ground_Station.MP4 --name kcu --name gs
```

KAT finds a frame with one calculation: `frame = time × frame rate`. That is
only valid when the rate is constant and known, which cameras do not
guarantee — the ground station file in the reference campaign reported no real
`r_frame_rate` and an average of 19.797 fps.

The script checks each source, keeps a constant rate as it is (so frame
numbers survive) and forces a variable one to a round rate near its average.
It also encodes with short keyframe intervals and no B-frames, which is what
makes scrubbing usable: a random jump costs about 20 ms instead of 250–630 ms.
The script's docstring has the measurements.

Cameras do **not** need the same rate or resolution. Each source carries its
own.

## 4. Take-off in each video

```bash
python scripts/find_takeoff.py video output/kcu_proxy.mp4 --from 0 --to 600
python scripts/find_takeoff.py video output/kcu_proxy.mp4 --from 150 --to 190
```

Export frames, look at them, narrow the range, repeat until one second is
left. Tedious, but it is the one method that worked.

**What did not work, and why:**

| Method | Outcome |
|---|---|
| Camera metadata | The KCU timecode matches nothing; concatenation dropped `creation_time` |
| Correlating image motion with telemetry | Correlation 0.094, second peak 0.093 — a pumping flight is periodic, so the curve has many equal peaks |
| Matching a fixed activity window | The `pp-*` window is less than half the real airborne time |
| Labelled static frames | Three candidates 190 s apart — one pumping cycle. Periodicity again |

Take-off is a single unambiguous event, visible in both the video and the
telemetry. Cross-check at the other end: take-off plus the airborne duration
should show the landing.

→ `VIDEOS[...]["takeoff_s"]` for each camera

**Residual uncertainty:** the telemetry anchor is a 30 m height threshold, the
video anchor is the visible lift-off. About ±1 s. For the next campaign,
record a synchronisation marker — a flash, a horn, a stopwatch in front of
both cameras. Ten seconds in the field replaces hours of searching.

## 5. The sensor mounting correction

```bash
python kat_validate.py flightdata/<log>.csv --windows 30
```

Check 5 compares the body axes exactly as the 3D panel draws them against two
references that do not come from the attitude sensor: the direction of travel
(from the position derivative) and the direction of the tether (from the kite
position to the ground station, because the kite hangs on its line).

For the reference flight the nose was right but the down axis sat 88.5° from
the tether. A single fixed rotation of about (87°, 7°, 5°) in roll, pitch and
yaw accounts for it, steady across the whole flight — a mounting or convention
offset, not noise. What remains after the correction (10–20°) is physical: a
kite flies at an angle of attack and with some sideslip.

Prefer the **median of the windows** over the single global fit. The global
fit is pulled by the few windows where the kite really was at a different
attitude, typically during reel-in.

→ `KITE_MOUNT_RPY`

Read the per-window table before trusting any of it:

- an axis that is **steady** is a mounting angle — correct it;
- an axis that **drifts smoothly** is most likely sensor drift — correctable,
  but over time rather than as a constant;
- an axis that **scatters** is the kite's own changing attitude — the very
  thing the panel exists to show, so it must not be corrected away.

## 6. Fill in the flight file

```bash
cp flights/_template.py flights/f2026_03_15.py
```

```python
FLIGHT_CSV = "flightdata/<log>.csv"
TELEMETRY_TAKEOFF_S = ...        # step 2
TELEMETRY_LANDING_S = ...        # step 2
PRE_TAKEOFF_S = 150.0            # ground time to keep before take-off
VIDEOS = {                       # step 3 and 4
    "kcu": {"path": "output/kcu_proxy.mp4", "takeoff_s": ..., "rotate_180": True},
    "gs":  {"path": "output/gs_proxy.mp4", "takeoff_s": ..., "rotate_180": False},
}
GS_KEY = "gs"
KITE_MOUNT_RPY = (..., ..., ...) # step 5
```

Then:

```bash
python viewer.py --flight 2026-03-15
python viewer.py --list-flights
```

The name is the file name without the leading `f`, with underscores as
dashes: `flights/f2026_03_15.py` is `2026-03-15`. `KAT_FLIGHT` in the
environment does the same job.

Before building anything, KAT checks that the flight log and the proxies of
the selected cameras exist, and that each has its `takeoff_s` filled in. If
not, it names the missing files and the flight file they came from.

Nothing else changes. `config.py` holds only what belongs to the tool.

### How the numbers combine

The master clock's `t = 0` is the first instant shown, so `t` never goes
negative:

```
FLIGHT_START_S = TELEMETRY_TAKEOFF_S − PRE_TAKEOFF_S
T0(camera)     = takeoff_s(camera) − TELEMETRY_TAKEOFF_S + FLIGHT_START_S
```

Offsets are derived, not typed in twice. If both videos are synchronised with
each other but drift together against the telemetry, `TELEMETRY_TAKEOFF_S` is
the single number to correct, and the spacing between cameras is preserved. If
one camera alone is off, correct its own `takeoff_s`.

**Direction:** if the plots and the 3D panel run *ahead* of the video,
decrease `TELEMETRY_TAKEOFF_S`; if the videos run ahead, increase it.

Keep `PRE_TAKEOFF_S` below the shortest camera lead — a camera that starts
recording later shows "no video" at the beginning. KAT prints a warning with
the right value when this happens.

## 7. Check before trusting

Run the whole validation once and read all five checks:

```bash
python kat_validate.py flightdata/<log>.csv 2417 3244
```

1. **Position** — the radius from `kite_pos_*` against `ground_tether_length`.
   A constant difference is expected (bridle and KCU below the kite); a
   varying one is not.
2. **Attitude** — `kite_0_yaw` against `kite_heading`, and against the
   velocity vector. On the reference flight the second comparison is the
   trustworthy one (R = 0.94 versus 0.34).
3. **Physics** — tether force against apparent wind speed squared.
4. **Force versus elevation** — force should be higher at low elevation, where
   the kite crosses the centre of the wind window fastest.
5. **The drawn axes** — see step 5.

A check that fails is not necessarily a broken tool; it is usually a column
that means something other than its name suggests. Four of the five findings
in the README came out of exactly that.
