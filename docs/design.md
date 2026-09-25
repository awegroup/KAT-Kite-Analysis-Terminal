# KAT — Kite Analysis Terminal

## Pipeline and Design Notes

Sep 18, 2026 · @Abdullah

## 1. Purpose and scope

KAT (Kite Analysis Terminal) is a visualisation tool for kite flight campaigns. It shows several data streams side by side on a single time axis: camera video, kite position and attitude in 3D, sensor time histories, and — when photogrammetry data exists — the deformed shape of the wing.

The starting point was `video_generator.py` in the `pimjhaanen/photogrammetry_thesis` repository. That script renders a composite video with four projection views of the point cloud, one camera stream and a metrics block. It works, but three properties made it impossible to extend:

- **One video source only.** The script holds a single `VideoCapture` and a single `input_video_start_s`. A second camera cannot be added without rewriting the render loop.
- **The layout is hard-coded.** The composition function is literally named `_compose_all_frame_top4`; the 2×4 arrangement is baked into pixel arithmetic.
- **No time-history plots.** Every panel draws only the current instant, so a sensor signal cannot be shown as a curve with a moving cursor.

The requirement set from Roland Schmehl was: one airborne video channel (selectable between KCU and a wing-mounted camera), one ground station channel, a 3D position and orientation panel with an animated tail, and a few time signals — all time synchronised, with selector buttons, and a view that keeps only four panels. The wireframe deformation panels should be an optional addition, because photogrammetry data exists only for this one flight while everything else exists for every flight.

KAT was therefore written as a new tool rather than a patch. It reuses the ideas of the original (composite frame, projection views, metrics block) but replaces the three structural limitations above.

## 2. Input data

The reference campaign is the flight of **9 October 2025** at the Kitepower test site in Bangor Erris, Ireland, with a 25 m² V3 kite.

### What exists

| Item | Source | Notes |
| --- | --- | --- |
| Flight log CSV | `awegroup/Flightdata09102025` (public) | 39 967 rows × 184 columns, 3 997 s, \~10 Hz |
| KCU camera video | Local (from Pim) | `KCU_1.MP4`, 5120×2880, 29.97 fps, 57 948 frames, 32.2 min |
| Ground station video | Local | `Ground_Station.MP4`, 2304×1536, variable frame rate, 36.5 min |
| Static photogrammetry frames | Repository, `Photogrammetry/static_experiments/static_test_output/` | 17 CSV files, one frozen frame each |
| Stereo calibration | Repository, `Photogrammetry/Calibration/` | 17 `.pkl` files |

The static CSV files are the only data committed to the repository, because its `.gitignore` excludes `*.csv` globally and then re-admits exactly that one folder with `!Photogrammetry/static_experiments/static_test_output/*.csv`. Video files are excluded by `*.MP4` / `*.mp4` and by the `Photogrammetry/input/left_videos/` rule.

### What is still missing

- **The photogrammetry time series** (the "complete dataset" produced by `combine_all_results.py`). Without it the deformation panels cannot show real motion. It can only come from Pim; regenerating it would require the raw stereo frames, which are not in the repository.
- **The wing-mounted camera**, if such a recording exists for this flight.
- **Frame offsets from Pim's synchronisation subsystem**, which would have removed the need for the manual offset search described in section 5.

### A naming inconsistency worth knowing

`video_generator.py` refers to `output/left_turn_frame_7182_complete_dataset.csv`, but the static folder contains `straight_flight_reelout_frame_7182.csv` — the same frame number under a different manoeuvre name. Do not search for the first filename; it is a leftover from an earlier naming scheme.

## 3. Step 1 — Understand the flight log

```
python kat_inspect.py flightdata\log_2025-10-09_58-33-00.csv
```

This prints the number of rows and columns, the time column, the duration, the sampling rate, which of the expected columns exist, and the `flight_phase` segments longer than 20 s. It also reports the mean and maximum tether force, which tells you the unit.

### What we found

**Sampling rate reported as infinite.** The tool printed `~inf Hz (dt=0.0000)`. The median time difference between consecutive rows is zero, because the `time` column holds whole seconds while roughly ten rows share each second. This is not a corrupt file; it is how the log is written. Section 11 explains why this matters and how KAT handles it.

**Tether force is in kilograms.** Mean 96.7, maximum 672.5. A force in newtons would be an order of magnitude larger. `video_generator.py` divides this column by 9.80665 in one place and not in another, so those two panels disagree with each other — at least one of them is wrong. KAT keeps the column as recorded and labels it kg.

**All the columns we need exist.** `kite_pos_east`, `kite_pos_north`, `kite_height`, `kite_0_roll/pitch/yaw`, `kite_0_vx/vy/vz`, `ground_tether_force`, `ground_tether_length`, `ground_tether_reelout_speed`, `airspeed_angle_of_attack`, `airspeed_sideslip_angle`, `flight_phase`. One rename is needed: the panels expect `kite_measured_va`, the log calls it `airspeed_apparent_windspeed`.

**`flight_phase` does not mark the whole flight.** The `pp-ro` and `pp-ri` segments (power production reel-out and reel-in) run only from 2417 s to 3244 s. Everything before and after is labelled `unknown`. But `kite_height` shows the kite airborne from about 2208 s to 3930 s, so `unknown` means "not pumping", not "on the ground". Assuming otherwise cost us a wrong offset estimate; see section 5.

### Deriving take-off and landing

```powershell
python -c "import pandas as pd, numpy as np; from kat import repair_time_axis; d=pd.read_csv(r'flightdata\log_2025-10-09_58-33-00.csv',low_memory=False); t=repair_time_axis(d,'time'); t=t-t[0]; h=pd.to_numeric(d['kite_height'],errors='coerce').ffill().bfill().to_numpy(); i=np.flatnonzero(h>30); print('take-off:',round(float(t[i[0]]),1)); print('landing:',round(float(t[i[-1]]),1))"
```

For this flight: **take-off 2208.3 s, landing 3930.3 s**, so 1722 s airborne. These two numbers anchor everything in section 5.

## 4. Step 2 — Video proxies

### Why proxies are mandatory

KAT locates a video frame with one calculation: `frame index = time × frame rate`. This is only valid if the frame rate is constant and known. A proxy is a re-encoded copy that guarantees both. It also solves three other problems at once:

- **Decoding cost.** The KCU source is 5120×2880 H.265. Decoding that per frame is far slower than 1280-wide H.264, and the panel is only about 1280 px wide anyway, so the extra resolution is never visible.
- **Corrupt frames.** Re-encoding lets ffmpeg conceal the damaged frames in the ground station recording.
- **The originals stay untouched.** They remain the archive.

### Checking the source first

```powershell
ffprobe -v error -show_entries stream=r_frame_rate,avg_frame_rate -of default "KCU_1.MP4"
```

|  | KCU\_1.MP4 | Ground\_Station.MP4 |
| --- | --- | --- |
| `r_frame_rate` | 30000/1001 = 29.97 | 90000/1 (placeholder) |
| `avg_frame_rate` | 30000/1001 = 29.97 | 2035153787/102800889 = 19.797 |
| Verdict | constant | **variable** |

When the two values agree and read as a real rate, the file is constant. The ground station file gives no real `r_frame_rate` and an average of 19.797 fps, which is not a frame rate any camera offers.

### Generating the proxies

```powershell
ffmpeg -i "KCU_1.MP4"          -vf "scale=1280:-2" -r 30000/1001 -fps_mode cfr -c:v libx264 -crf 20 -g 10 -bf 0 -tune fastdecode -an "output\kcu_proxy.mp4"
ffmpeg -i "Ground_Station.MP4" -vf "scale=1280:-2" -r 20         -fps_mode cfr -c:v libx264 -crf 20 -g 10 -bf 0 -tune fastdecode -an "output\gs_proxy.mp4"
```

Every flag has a reason:

| Flag | Purpose |
| --- | --- |
| `scale=1280:-2` | match the panel width; the viewer and the 2560×1440 render both show videos at about 1280 px |
| `-r …` `-fps_mode cfr` | force a constant frame rate (on older ffmpeg builds: `-vsync cfr`) |
| `-crf 20` | visually near-lossless quality |
| `-g 10` | a keyframe every 10 frames, so a jump never has to decode far |
| `-bf 0` | no B-frames, so frames are stored in display order |
| `-tune fastdecode` | switch off the compression tools that are expensive to decode |
| `-an` | drop the audio track |

Two rules govern the frame rate:

- **Match the source rate when it is already constant.** KCU was re-encoded at 29.97, so ffmpeg copied and dropped nothing: 57 948 frames in, 57 948 frames out. Frame indices are preserved, which keeps any frame-number reference valid.
- **Pick a rate near the average when it is not.** The ground station was forced to 20 fps, which required duplicating 480 frames and dropping 34. Choosing 30 instead would have duplicated far more frames without adding information.

The cameras do **not** need the same frame rate or the same resolution. Each source carries its own rate, and each panel is sized independently. What matters is that every rate is constant and known. Re-encoding with different flags does not change the offsets, as long as the frame rate and frame count stay the same.

### Why the keyframe and B-frame flags matter

A video stores a complete picture only now and then — a keyframe — and in between only what changed. To show an arbitrary frame, the decoder has to start from the keyframe before it and decode forward. The first proxies used ffmpeg's default of one keyframe every 250 frames, so every jump on the time slider decoded up to 250 frames.

B-frames add a second problem: they are stored out of display order, and OpenCV's seek then overshoots, steps back and retries. That produced erratic seek times — usually 25 ms, occasionally 300–450 ms.

Measured on 1280×720 H.264, random jumps:

| Encoding | Median | 90th percentile | Worst | Size |
| --- | --- | --- | --- | --- |
| default (keyframe every 250) | 245 ms | 592 ms | 628 ms | 40 MB |
| keyframe every 5, B-frames on | 25 ms | 162 ms | 324 ms | 50 MB |
| keyframe every 5, no B-frames | 19 ms | 21 ms | 22 ms | 53 MB |
| **keyframe every 10, no B-frames** | **20 ms** | **23 ms** | **25 ms** | **48 MB** |

The last row is the recommended recipe: jumps become consistently about 20 ms at a cost of roughly 20 % in file size. Lowering the resolution to 480p, which was also considered, only brought the default encoding down to 139 ms — the keyframe interval, not the resolution, was the bottleneck.

### Verify

```powershell
python kat_sync.py meta "output\kcu_proxy.mp4" "output\gs_proxy.mp4"
```

Expected: KCU 29.970030 with 57 948 frames, GS exactly 20.000000.

## 5. Step 3 — Finding the offsets

An offset answers one question: **at master clock t = 0, which second of this video are we at?** Every camera needs its own.

### Methods that did not work

**Camera metadata.** The KCU file carries a timecode of `23:27:40:00`. The flight took place between 17:35 and 18:05 UTC, and Ireland was UTC+1, so local time was about 18:35–19:05. The timecode matches nothing, most likely because the camera clock was never set or the tag survived from the first segment of a concatenation. Neither file carries a `creation_time` tag; the concatenation (`encoder = Lavf61.7.100`) dropped it.

**Motion-energy cross-correlation.** We extracted a per-frame image-difference signal and correlated it against `kite_turn_rate`. It returned a correlation of 0.094 with a second peak at 0.093 — a flat curve carrying no information. One result landed exactly on the search boundary, which is the classic sign that no true optimum exists inside the window. The underlying reason is that a pumping flight is periodic: shifting by one cycle reproduces almost the same pattern, so the correlation curve has many peaks of near-equal height. These commands were removed from `kat_sync.py` rather than left in place to mislead a future user.

**Fixed-duration activity window.** We searched the video for the most active stretch matching the length of the `pp-*` phases. It failed because those phases cover 827 s while the kite was actually airborne for 1722 s — the window we were matching was less than half the real one.

**Labelled static frames (partly useful).** The static CSV file names carry both a frame number and a manoeuvre: `..._frame_7182` is `straight_flight_reelout`, `..._frame_17611` is `depowered_state_reelin`, and so on. `python kat_sync.py anchor <flight.csv>` converts each frame number to a video time (frame ÷ 29.97) and scans for offsets that place all seven anchors in the correct `flight_phase`. All seven matched for three candidates, roughly 190 s apart — one pumping cycle. Periodicity again. The candidates were also mutually inconsistent with `kite_height`, which means the frame numbering of `KCU_1.MP4` does not align with Pim's photogrammetry indexing. **This is an open question for Pim.**

### The method that worked

Take-off is a single unambiguous event visible in both the video and the telemetry.

1. **Telemetry side.** The first moment `kite_height` exceeds 30 m: **2208.3 s**.
2. **Video side.** Extract candidate frames and find where the kite leaves the ground:

```powershell
160,170,180,190,200,210 | ForEach-Object { ffmpeg -v error -y -ss $_ -i "output\kcu_proxy.mp4" -vframes 1 "output\tk_$_.png" }
```

Narrow down by bisection. For KCU the kite is still on the ground at 168 s and clearly airborne at 169 s, so take-off is **168.5 s**. For the ground station the same procedure gives about **399.5 s**.

3. **Sanity check at the other end.** Take-off + 1722 s should show the landing. For the ground station that is 2121 s, and the kite is indeed descending through 20–30 m there. Two independent events agreeing makes a wrong answer very unlikely.

### Results for this flight

| Camera | Take-off in video | Offset T0 | Notes |
| --- | --- | --- | --- |
| KCU | 168.5 s | 18.5 | rotate 180°, camera mounted upside down |
| Ground station | 399.5 s | 249.5 | later refined by \~1 s while watching playback |

The offset follows from `T0 = video_takeoff − telemetry_takeoff + FLIGHT_START_S`. The values above assume the current clock origin `FLIGHT_START_S = 2058.3` (section 6). The offset numbers change whenever the origin moves, but the take-off seconds in the second column do not — those are the measured quantities, and they are what `config.py` stores.

### Residual uncertainty

The telemetry anchor uses a 30 m height threshold while the video anchor is the visible moment of lift-off. These are not the same instant, so the result carries roughly ±1 s. At 20 m/s that is 20 m of kite travel. It is acceptable for viewing but not for frame-accurate analysis.

For future campaigns this can be removed entirely: record a **synchronisation marker** at the start — a flash, a horn, a clapperboard, or a stopwatch held in front of both cameras. Ten seconds of effort at the field replaces hours of searching afterwards.

## 6. Step 4 — The configuration

Everything a user changes lives in `config.py`. Both `render.py` and `viewer.py` call its `build()` function, so a change applies to the video output and the interactive window at once.

### The clock origin and the shared take-off anchor

```python
TELEMETRY_TAKEOFF_S = 2208.3   # take-off instant in the flight log [s]
TELEMETRY_LANDING_S = 3930.3   # landing instant in the flight log [s]

CLIP_TO_AIRBORNE = True        # keep only the flight window
PRE_TAKEOFF_S = 150.0          # ground time kept before take-off [s]
POST_LANDING_S = 60.0          # time kept after landing [s]

FLIGHT_START_S = max(0.0, TELEMETRY_TAKEOFF_S - PRE_TAKEOFF_S) if CLIP_TO_AIRBORNE else 0.0

VIDEO_TAKEOFF_S = {            # second at which the kite leaves the ground
    "kcu": 168.5,
    "wing": 0.0,               # not available yet
    "gs": 399.5,
}

def _t0(video_key):
    return VIDEO_TAKEOFF_S[video_key] - TELEMETRY_TAKEOFF_S + FLIGHT_START_S
```

**Master clock `t = 0` is the first instant shown**, so `t` never goes negative. `FLIGHT_START_S` is not typed in; it is computed as take-off minus `PRE_TAKEOFF_S`. For this flight that is 2058.3 s in the log, and take-off falls at exactly `t = 150`. Changing `PRE_TAKEOFF_S` moves the origin and every offset follows automatically.

Keep `PRE_TAKEOFF_S` below the shortest camera lead. The KCU camera starts 168.5 s before take-off; a larger value makes its panel show "no video" at the start.

The offsets are derived, not written as two independent numbers. If both videos are synchronised with each other but drift together against the telemetry, only `TELEMETRY_TAKEOFF_S` is wrong, and correcting that single number shifts both videos without disturbing their spacing. If one camera alone is off, correct its entry in `VIDEO_TAKEOFF_S`.

**Direction of correction.** If the 3D panel and the plots run *ahead* of the video, decrease `TELEMETRY_TAKEOFF_S`. If the videos run ahead, increase it. Each second of change moves the videos by one second.

### Camera selection

```python
AIRBORNE = "kcu"    # which airborne camera
AIRBORNE_VIDEOS = {
    "kcu":  ("output/kcu_proxy.mp4", _t0("kcu"), True),   # True = rotate 180°
    "wing": ("output/wing_proxy.mp4", _t0("wing"), False),
}
GS_VIDEO = ("output/gs_proxy.mp4", _t0("gs"), False)
```

The KCU camera is mounted upside down — the ground appears at the top of the raw frame — hence the rotation flag. Adding the wing camera means generating its proxy and filling in its take-off second.

### Telemetry source

```python
USE_FLIGHT_LOG = True
FLIGHT_CSV = "flightdata/log_2025-10-09_58-33-00.csv"
DATASET_START_S = 2417.0       # the synthetic dataset's own t = 0 in the log
```

With `USE_FLIGHT_LOG = True` the default layout reads the flight log directly, so the **whole flight** is available with no intermediate file. When the deformation panels are switched on, KAT reads the synthetic complete dataset instead, because that is where the point cloud lives. That file was generated with `kat_dataset.py --flight-start-s 2417`, so its timestamps are shifted by `DATASET_START_S − FLIGHT_START_S` on loading to land on the same master clock. It does not need to be regenerated when the origin moves.

### One file per flight

Everything specific to a flight lives in `flights/fYYYY_MM_DD.py`; `config.py` holds only what belongs to the tool. The flight file is selected on the command line and its values override the defaults:

```bash
python viewer.py --flight 2025-10-09
python viewer.py --list-flights
```

```python
# flights/f2025_10_09.py
FLIGHT_CSV = "flightdata/log_2025-10-09_58-33-00.csv"
TELEMETRY_TAKEOFF_S = 2208.3
TELEMETRY_LANDING_S = 3930.3
PRE_TAKEOFF_S = 150.0
VIDEOS = {
    "kcu":  {"path": "output/kcu_proxy.mp4", "takeoff_s": 168.5, "rotate_180": True},
    "wing": {"path": "output/wing_proxy.mp4", "takeoff_s": None, "rotate_180": False},
    "gs":   {"path": "output/gs_proxy.mp4", "takeoff_s": 399.5, "rotate_180": False},
}
GS_KEY = "gs"
KITE_MOUNT_RPY = (87.1, 7.2, 5.3)
```

Camera names are free; `GS_KEY` says which entry is the ground station and the rest appear in the viewer's camera selector. `flights/_template.py` carries the same fields with the command that produces each value written beside it.

Before building anything, KAT checks that the flight log and the proxies exist and that each camera has its take-off second; if not it names the missing files and the flight file they came from, rather than failing somewhere inside pandas. A camera that starts recording after the window opens is reported at startup with the largest `PRE_TAKEOFF_S` that avoids a blank start.

### Live values and the kite model

```python
LIVE_VALUES = [                     # (column, label, unit, decimals)
    ("kite_measured_va", "Va", "m/s", 1),
    ("kite_actual_depower", "Depower", "", 1),
    ("kite_actual_steering", "Steering", "", 1),
    ("ground_tether_force", "Force", "kg", 0),
    ("kite_height", "Height", "m", 0),
]

KITE_OBJ = "models/V3_Kite.obj"     # None -> built-in canopy
KITE_OBJ_AXES = "-x +y -z"          # which .obj axis becomes forward, right, down
KITE_OBJ_MAX_FACES = 300            # CAD exports are simplified to this on load
KITE_MODEL_SIZE = 55.0              # displayed span [m], enlarged for visibility
SHOW_TRIAD = True                   # body-axis arrows on top of the model
SHOW_TETHER = True                  # line from the ground station to the kite
VIDEO_FIT = "contain"               # whole frame; "cover" fills the cell by cropping
KITE_MOUNT_RPY = (87.1, 7.2, 5.3)   # sensor-to-kite correction: roll, pitch, yaw [deg]
```

`LIVE_VALUES` feeds both the status bar and the Live values panel of the extended layout; adding a quantity is one line. Depower and steering are logged with values such as 41.5 and −4.7, which look like percentages, but the unit is not documented, so they are shown without one.

`KITE_MOUNT_RPY` turns the drawn body frame by a fixed rotation in roll, pitch and yaw. The attitude sensor is not aligned with the kite's own axes: for this flight the difference is about (87°, 7°, 5°), measured from the tether direction by `kat_validate.py` (section 10, check 5). Without it the kite is drawn lying on its side. It affects only the drawing; the logged angles are left untouched.

The model itself is the V3 wing CAD geometry from `awegroup/TUDELFT_V3_KITE`, exported to `.obj`. An `.obj` file carries no agreed axis convention — one program exports Y-up and −Z-forward, another Z-up — so `KITE_OBJ_AXES` maps the file's axes onto the body axes; for this export the value is `"-x +y -z"`, derived from the geometry itself: the wing tips sit below and behind the centre, and the thickest chordwise section is the inflated leading edge. A CAD export carries tens of thousands of triangles and is simplified on load to `KITE_OBJ_MAX_FACES` by vertex clustering — matplotlib draws 3D faces slowly (200 triangles cost 14 ms per frame, 1400 cost 64 ms) and the kite is a small object on screen, so the detail is invisible either way. With `KITE_OBJ = None` a built-in canopy is used instead, shaped to the published V3 proportions.

### Layout and output

```python
WITH_DEFORMATION = False       # True -> add the wireframe panels
WINDOW_S = 60.0                # width of the scrolling time window
WARMUP_S = 0.75 * WINDOW_S     # render.py starts this much later so the plot has a past
DURATION_S = 60.0              # length of the rendered video
FPS = 30000 / 1001             # render rate; matches the KCU proxy exactly
OUT_SIZE_DEFAULT = (2560, 1440)
VIEW_SIZE = (1600, 900)        # initial viewer window size
VIEW_FPS = 30.0                # upper bound on frames drawn per second
VIEW_PLAY_SPEED = 1.0          # 1.0 = real time; [ and ] change it in the viewer

SIGNAL_CHANNELS = [            # what the signal panels can show, via their drop-down
    {"label": "Tether force", "cols": ["ground_tether_force"], "unit": "kg"},
    {"label": "Flow angles", "cols": ["airspeed_angle_of_attack",
                                      "airspeed_sideslip_angle"], "unit": "deg",
     "names": ["angle of attack", "sideslip"]},
    # ... apparent wind, height, reel-out speed, tether length, depower, attitude
]
SIGNAL_1 = 0                   # channel the first signal panel starts on
SIGNAL_2 = 2                   # and the second, in the extended layout
```

`VIEW_SIZE` only sets the size at which the viewer window opens; the graphs are always composed at the window's actual pixel size, so they stay sharp however the window is resized. `VIEW_FPS` is a ceiling, not a rate: playback follows the wall clock, so a slow machine drops frames rather than running in slow motion. `WARMUP_S` affects only `render.py`. The viewer's slider always spans the whole window, from `t = 0`.

Adding a signal is one entry in `SIGNAL_CHANNELS`. Several columns in `cols` are drawn together and `names` gives them readable labels; without `names` the column name is shortened automatically (`kite_0_roll` becomes "roll"). An entry may also carry `"ylim": (lo, hi)` to fix the axis instead of letting KAT choose it.

## 7. Architecture

### The master clock

Every source keeps its own stopwatch; KAT puts one clock on the wall and makes everybody read it. The master clock `t` advances, and each source translates it into its own index:

| Source | Translation |
| --- | --- |
| KCU video | `frame = (t0_kcu + t) × 29.97` |
| Ground station video | `frame = (t0_gs + t) × 20` |
| Telemetry | interpolate at `t` |
| Point cloud | nearest frame at `t` |

Nothing requires the rates to be equal, and nothing requires the sources to know about each other. Adding a third camera costs one line. This is exactly what the original script could not do, because it carried a single global `input_video_start_s`.

### Three layers

**Sources** answer "what is your data at time t?" — `VideoSource`, `TelemetrySource`, `PointCloudSource`. Each holds its own time mapping.

**Panels** answer "what do you look like at time t?" and return a BGR image of the requested size. A panel never knows where it sits on screen.

**The grid** places panels and produces one frame. `compose_frame()` does this for a single instant; `render_video()` loops it into an mp4; the viewer calls the same function on demand from a slider.

Because a panel is just a function of `t`, the same objects serve both the offline render and the interactive window. That is why the viewer needed almost no new code.

### Layout as configuration

```python
Slot(VideoPanel(air, "KCU camera"), row=0, col=0)
Slot(Path3DPanel(telemetry, "3D position + attitude", trail_s=25.0), row=0, col=1)
Slot(VideoPanel(gs, "GS camera"), row=1, col=0)
Slot(SIGNALS, row=1, col=1)
```

Each entry carries a panel, a row, a column and optional spans. Moving a panel means changing two integers. Growing the grid means changing `n_rows` and `n_cols`. When Roland asked for a different arrangement three times during development, each change was a few lines in `config.py` and nothing in the engine.

## 8. Panels and layouts

### The default layout — four panels, 2×2, plus a status bar

This is the standard view and works for any flight, because it needs nothing but the flight log and the videos.

| Position | Panel | Data source |
| --- | --- | --- |
| top left | Airborne camera | proxy video, own offset |
| top right | 3D position + attitude, with kite model | `kite_pos_east/north/height`, `kite_0_roll/pitch/yaw` |
| bottom left | Ground station camera | proxy video, own offset |
| bottom right | Time signals, UTC time axis | telemetry columns |
| bottom strip | Status bar: UTC date-time, `t`, live values | flight log time + `LIVE_VALUES` |

### The extended layout — ten panels, 3×4

Enabled with `WITH_DEFORMATION = True`. Adds the three projection views, the 3D shape view and a Live values panel. Requires photogrammetry data, which currently exists only for the 9 October flight and only as one static frame.

### Panel details

**Status bar.** A dark strip along the bottom edge of every frame, in both layouts:

`2025-10-09  15:35:21.4 UTC | t = 150.0 s | Va … | Depower … | Steering … | Force … | Height …`

The date and time are the flight log's own Unix time at that instant, so they are absolute and unaffected by the video offsets or by where the master clock starts. The strip is reserved space: the panels share the remaining height and nothing is covered.

**`Path3DPanel`** draws the full trajectory in faint grey, the last `trail_s` seconds in blue, the current position as a black marker, and **the kite itself** as a shaded surface placed at that position and turned by the Euler angles. With no `.obj` configured, a built-in LEI-style canopy is used: an arc whose tips curve towards the tether, as on the real wing. The body-axis triad is drawn on top — red for the nose, green for the right wing, blue for down.

The attitude comes from a 3-2-1 sequence (yaw, pitch, roll) giving the body-to-NED rotation, converted to ENU for plotting: east = v\_E, north = v\_N, up = −v\_D. Each model vertex is mapped as `position + (vertex @ body_axes) × KITE_MODEL_SIZE`. This is the panel Roland noted Pim had not managed to finish; the original tried to derive position from geodetic latitude and longitude treated as spherical angles about the ground station, which is not what those quantities are. Using the pre-derived ENU columns removes the problem entirely.

**`ScrollingTimeSeriesPanel`** keeps the cursor fixed and scrolls the axis leftwards past it, as Roland requested. The x axis shows **UTC clock time** (HH:MM:SS), with ticks aligned to round clock seconds and a spacing chosen from the window width. The whole series is drawn once into a very wide strip; each frame crops a window from that strip and pastes the fixed y-axis back on the left. `WINDOW_S` sets the visible width and `cursor_frac` (default 0.75) sets how much of the panel is past versus future.

**`VideoPanel`** letterboxes the frame to preserve aspect ratio and draws a label. It shows a grey placeholder reading "no video" when the master clock falls outside the recording, which is normal near the ends.

**`WireframePanel`** and **`Points3DPanel`** draw the point cloud as 2D projections (front x-z, side y-z, bottom x-y) and as a 3D scatter. Axis limits come from `LIMS`.

**`MetricsPanel`** prints the `LIVE_VALUES` list in the extended layout, including steering.

**Choosing what the signal panels show.** Each signal panel carries a drop-down in its corner listing the channels from `SIGNAL_CHANNELS`; clicking one rebuilds the plot, which takes about 20 ms. When a channel holds several curves, a small colour key is drawn in the corner — it cannot live inside the plot, because the plot scrolls and would carry it away.

**Axis limits come from the body of the data, not its extremes.** A single spike, typically at take-off, would otherwise compress the whole flight into a thin band: on this flight the angle of attack reaches 65° for a few seconds while the flight sits between −3° and 12°. KAT uses the usual quartile fence over the whole series, so brief extremes run off the chart instead of dictating the scale, and adds a margin so the curves do not touch the frame. The limits are computed once, so the scale stays fixed across the flight and two moments can be compared by eye. A channel can override this with `"ylim"`.

**The strip is built around the current moment, not for the whole flight.** A large panel needs about 25 px per second, which for a 30-minute flight would be a 45 000 px wide figure — beyond what the renderer produces, and it came out short without saying so, leaving the panel blank after a few minutes. KAT now draws three windows either side of the cursor and rebuilds when the time leaves that span, about a dozen times over a flight.

**Angles are interpolated the short way round.** The telemetry is 10 Hz and the panel draws 30 frames a second, so values in between are interpolated. Interpolating yaw linearly across the ±180° boundary sweeps almost a full turn — the kite flipped over for a frame whenever it crossed due south. The series is unwrapped first, which keeps the step at two degrees.

**The tether is drawn**, from the ground station at the origin to the kite, with a marker at the station. It makes the scene readable: the kite visibly hangs on its line, which is also the reference the mounting check uses. The axes then have to include the origin, so the box grows and the kite looks smaller; `SHOW_TETHER = False` restores the tighter view.

**Video panels** show the whole frame by default, with dark bars where the aspect ratio does not match the cell, and a small caption in the corner of the picture. `VIDEO_FIT = "cover"` fills the cell instead by cropping the edges evenly, which makes a row of cameras look uniform at the cost of losing the edges. Thin separators are drawn between panels that touch.

### The interactive viewer

`python viewer.py` opens a matplotlib window with a time slider, a play/pause button, radio buttons for the airborne camera, and a checkbox that toggles the deformation panels — switching the grid between 2×2 and 3×4 live. The slider spans the whole telemetry window from `t = 0`, so the ground phase before take-off is reachable.

| Control | Action |
| --- | --- |
| Space | play / pause |
| ← → | step 1 s back / forward |
| Shift + ← → | step 0.1 s back / forward |
| Double-click a panel | enlarge it to the whole view; double-click again or Esc to return |
| 1 … 9 | enlarge panel N directly, counted row by row; the same key returns |
| F | full-screen window (matplotlib's own shortcut) |
| Drag, in an enlarged 3D panel | rotate the view |
| Scroll wheel, in an enlarged 3D panel | zoom |
| R, in an enlarged 3D panel | reset the view |

An enlarged panel is redrawn at its new size, so text and lines stay sharp even full screen; it is not a magnified picture. Playback continues while a panel is enlarged, and a rotated 3D view is kept when returning to the grid. One limitation: matplotlib's 3D axes do not clip, so when zoomed in far, anything outside the box is still drawn outside it.

**How the frame reaches the screen.** The graphs are composed at the window's own pixel size; only the videos come from the 720p proxies. The composed frame is then written straight into the window's pixel buffer and only that region is pushed to the screen. Matplotlib still draws the buttons and the slider, but no longer processes the frame itself. The earlier approach passed the frame through `imshow`, whose rendering pipeline cost grew with window size — 54 ms per frame at 1600×900 — which forced a choice between sharp graphs and smooth playback. Writing to the buffer takes about 6–8 ms, which removes that trade-off.

`viewer_tk.py` is an alternative window built on Tk and Pillow: equally sharp and fast, with plainer controls.

Two implementation notes matter. Panel figures are created with `Figure` plus `FigureCanvasAgg` rather than `plt.figure()`; with `plt.figure()` the interactive backend registers them and opens each one as its own window. On Windows the viewer declares itself DPI-aware before any window exists; otherwise, with display scaling at 125–150 %, Windows stretches the window as a bitmap and everything blurs.

## 9. Performance

The first working version took 420 ms per frame, which made a 100-second render take about 21 minutes and ruled out an interactive viewer entirely. Three changes brought it down by a factor of twenty to fifty.

### Build once, update what changes

The naive approach rebuilds the whole matplotlib figure every frame: axes, grid, tick labels, title, and all the data. But between two frames only the data moves. So each panel constructs its figure once and afterwards calls `set_data()` or `set_data_3d()` on the existing artists.

### Blitting

Even with a reused figure, matplotlib still redraws every artist. Blitting goes one step further: after the static content is drawn once, the canvas is captured with `copy_from_bbox`. Each frame then restores that snapshot and draws only the moving artists on top of it with `draw_artist`, finishing with `blit`.

The analogy: instead of rebuilding the stage set for every scene, you leave the set standing and move only the actors.

One consequence is that the body-axis triad could no longer use `quiver`, which cannot be updated and has to be recreated each frame. Three plain `Line3D` objects with a marker at the tip carry the same information and update in place. The arrowheads are gone; the colour coding remains.

### Video decoding and seeking

Once the panels were fast, the viewer's remaining delay was jumping on the time slider: nearly a second per jump. The cause was the proxy encoding, not the resolution — see section 4. Short keyframe intervals and no B-frames bring a jump down to about 20 ms per video.

`VideoSource` also steps forward frame by frame for short jumps and seeks only for longer ones; with the recommended proxies a seek costs about as much as decoding a dozen frames, so the threshold is 12 frames. A small frame cache avoids a seek when scrubbing backwards.

### Getting the frame onto the screen

The composed frame was first handed to matplotlib with `imshow`. Its antialiased resampling alone cost about 54 ms per frame, and even with `interpolation="nearest"` the cost grew with the window's pixel count — 35 ms at 1900×1000. Composing a smaller frame and scaling it up made playback smooth but the graphs blurred. The viewer now writes the frame straight into the window's pixel buffer (section 8), which takes 6–8 ms at any size.

### Measured results

| Configuration | First version | Now |
| --- | --- | --- |
| Render frame, default 4 panels @ 2560×1440 | 420 ms | 8–15 ms |
| Render frame, extended 10 panels @ 2560×1440 | 420 ms | 18–23 ms |
| Viewer playback at full window resolution (1600×900) | — | \~40 ms, about 25 fps |
| Viewer jump on the time slider | \~930 ms | \~85 ms |
| `Path3DPanel` alone, with kite model | 150+ ms | \~10 ms |
| `ScrollingTimeSeriesPanel` | \~150 ms | 0.1–2.5 ms |

A 60-second render takes well under a minute. For the viewer, set `KAT_PROFILE=1` to print the composed resolution, the compose and paint times, and the achieved frame rate while playing:

```powershell
$env:KAT_PROFILE=1; python viewer.py
```

The figures above were measured on Linux; a Windows machine can differ.

### Why not port to Julia

The question came up. Almost all of the time is spent in native libraries — FFmpeg for decoding, matplotlib's C++ renderer, NumPy — not in Python itself, so changing the language would change little. Julia's GLMakie would be faster for the 3D view, but because it draws on the GPU, not because of the language. The Python equivalent is pyqtgraph. Porting would mean rewriting the tool and losing direct use of the group's Python code. Julia would make sense for integration with the group's Julia packages, not for speed.

## 10. Data validation

```powershell
python kat_validate.py "flightdata\log_2025-10-09_58-33-00.csv" 2417 3244
```

The tool runs five independent cross-checks. Independent means each one uses a different part of the data to test the same thing, so agreement is meaningful.

### 1. Position

The distance from the ground station to the kite, computed from `kite_pos_east/north/height`, should match `ground_tether_length`.

| Quantity | Value |
| --- | --- |
| Radius, mean | 270.3 m |
| Tether length, mean | 252.7 m |
| Difference | +17.6 m, std 2.1 m |

The standard deviation of only 2.1 m is the important number: the offset is constant, not noise. The radius is *larger* than the tether reading because `ground_tether_length` measures the line paid out from the drum, while the bridle and the KCU add a fixed length below the kite. The radius also matches `kite_distance` exactly, confirming the position columns are internally consistent. **Position data is trustworthy.**

### 2. Attitude

Comparing `kite_0_yaw` with `kite_heading` gave a circular consistency of R = 0.34 and a bimodal distribution — no constant offset could fix it. None of the candidate corrections (±90°, 180°) improved matters.

Comparing `kite_0_yaw` with the direction of the velocity vector instead, using `kite_0_vx/vy`, gave a completely different picture:

| Axis order | R | Circular mean | Within ±30° |
| --- | --- | --- | --- |
| `atan2(vy, vx)` — x = north | **0.94** | −6.4° | 93% |
| `atan2(vx, vy)` — x = east | 0.52 | −74.2° | 31% |

So the logged heading in `kite_0_yaw` is correct. The residual −6.4° is plausibly the kite's actual sideslip, so no correction is applied. The conclusion is that **`kite_heading` measures something other than the NED course** — worth confirming with whoever produced the log. The first row also establishes that `kite_0_vx` is the north component and `kite_0_vy` the east component.

Note what this check does *not* establish: it tests the data, not the drawing. An earlier version of this document concluded from it that the arrows in the 3D panel were right. They were not — see check 5.

### 3. Physics

Fitting `log(F)` against `log(v_a)` gave a slope of 2.26 with a correlation of 0.96 between force and squared apparent wind speed. Tether force scales with the square of apparent wind speed, as expected.

### 4. Force versus elevation

A question that came up while watching the panel: the tether force seems to rise while the kite descends. The check quantifies it:

| Elevation | Mean force |
| --- | --- |
| Low (< 23°) | 316.8 kg |
| High (> 32°) | 125.9 kg |

Correlation between elevation angle and force: −0.59.

This is correct physics, not a synchronisation error. In a pumping cycle the kite flies figure-eights, and it is fastest at the bottom of the eight, where the elevation angle is low and it crosses the centre of the wind window. Apparent wind speed is highest there, and force scales with its square. The kite is not falling; it is accelerating.

### 5. The drawn axes against physics

This check takes the body axes exactly as `Path3DPanel` draws them and compares them with two references that do not come from the attitude sensor at all:

- the **direction of travel**, from the time derivative of the position — the nose (red) should point along it;
- the **direction of the tether**, from the kite position to the ground station — the down axis (blue) should point along it, because the kite hangs on its line.

It uncovered two separate problems.

**A drawing bug.** Converting the body axes from NED to ENU swapped the rows of the rotation matrix instead of its columns. North and south came out right, east and west were mirrored, so a kite flying east was drawn with its nose pointing west. Along a figure-eight the nose deviated from the direction of travel by a median of 132° before the fix and 0.4° after it, on a physically consistent test flight.

**A sensor mounting offset.** With the drawing fixed, the real flight gives:

| Quantity | As logged | After the correction |
| --- | --- | --- |
| Nose vs direction of travel | 18.1° | 17.0° |
| Right wing vs its expected direction | 91.9° | 18.7° |
| Down axis vs the tether | 88.5° | 12.4° |

The nose is right, but the other two axes sit about a quarter turn away. The check finds the single rotation that best maps the sensor frame onto the expected one, in all three angles at once: **(87.1°, 7.2°, 5.3°)** in roll, pitch and yaw. Per 30 s window the roll estimate varies by only 2.8°, so it is a mounting or convention offset, not noise.

Two details matter for reading the result. First, prefer the median of the windows over the single global fit: the global fit is pulled by the few windows where the kite really was at a different attitude, which on this flight are the reel-in phases — the pitch estimate jumps to 20–33° exactly at 2627 s, 2837 s, 3017 s and 3197 s, matching the logged reel-in segments. Second, what remains after the correction (12–19°) is physical, not error: a kite flies at an angle of attack and with some sideslip, so it is neither perpendicular to its tether nor pointing exactly along its velocity. `KITE_MOUNT_RPY` in the flight file applies the correction to the drawing only.

The per-window table is also a diagnostic in its own right. An axis that stays steady is a mounting angle and belongs in the correction. An axis that drifts smoothly is most likely sensor drift, which a constant cannot fix. An axis that scatters without a trend is the kite's own changing attitude — the very thing the 3D panel exists to show, so correcting it away would erase the measurement. Fitting per window and *applying* the result would do exactly that: the kite would always appear perfectly aligned with its tether, and the panel would be drawing the assumption rather than the data.

**This matters beyond KAT.** Heading is unaffected, because the nose axis is shared, which is why check 2b passed. But any other use of `kite_0_roll` and `kite_0_pitch` — an EKF, the aero-structural model, an angle-of-attack estimate — must apply the same correction or it will be wrong. Whether the mounting is exactly (90°, 0°, 0°) should be confirmed with the team that installed the sensor.

## 11. Critical pitfalls

Every item below produced a wrong result without raising an error. That is what makes them dangerous: the code kept running and the output looked plausible.

### Time and frame rate

**29.97 is not 30.** `video_generator.py` computes `7182/30`. The KCU camera runs at exactly 30000/1001 = 29.97. For a single frame the error is 0.24 s; by the end of a 32-minute recording it accumulates to 1.9 seconds. Always read the real rate from the file.

**Variable frame rate breaks the index-to-time relation.** The ground station file has no real `r_frame_rate` and an average of 19.797 fps. In such a file `frame = time × fps` is simply invalid, because frames are not evenly spaced. Re-encode to constant rate before anything else.

**Duplicate timestamps mean two different things.** In the flight log, roughly ten rows share each whole-second `time` value; these are genuine successive samples and must be spread across the second. In the complete dataset, 61 rows share each frame's timestamp because each row is one marker point, and their telemetry values are identical; these must be deduplicated. Applying the wrong treatment scrambles the time axis. KAT distinguishes them by checking whether the values within a timestamp group actually differ. Getting this wrong made the 3D panel jump back and forth in time, and it silently corrupted the time axis of the plots and the metrics panel as well.

### Units and column names

**Tether force is in kilograms.** Mean 96.7, maximum 672.5 for this flight. If a panel shows peaks above about 700 while labelled kg, a conversion to newtons has crept in somewhere.

**Column names differ between the log and the panels.** The log has `airspeed_apparent_windspeed`; the panels expect `kite_measured_va`. KAT renames it. Critically, a missing column does **not** raise an error — it is filled with NaN and the panel simply shows a dash. Always read the list of columns the tool reports as missing.

**`kite_heading` does not agree with the NED course.** Use the velocity vector (`kite_0_vx/vy`) as the reference for attitude instead.

**Time zones.** The status bar shows take-off at about 15:35 UTC, computed from the flight log's Unix time. The Flightdata README lists the flight as 17:35–18:05 "UTC" — exactly two hours later, which is the offset of Dutch summer time (CEST, UTC+2). Pim's own `video_generator.py` output labels a frame early in the flight `UTC: 15:36:15.60`, independently agreeing with the log. So the README times are most likely CEST mislabelled as UTC. Local time at the site in Ireland (IST, UTC+1) would read 16:35. When quoting a time, always name the zone.

### Flight phase and geometry

**`unknown` in `flight_phase` does not mean "on the ground".** For this flight the `pp-*` phases span 827 s while the kite was airborne for 1722 s. Deriving take-off from `flight_phase` gives the wrong answer; use `kite_height`.

**Latitude and longitude are not spherical angles about the ground station.** Converting them with the tether length as a radius produces a meaningless position. Use `kite_pos_east/north/height`.

**The KCU camera is mounted upside down.** The raw frames show grass at the top. `rotate_180=True` is required for that source and must not be applied to the ground station.

### Method

**Cross-correlation is unreliable on periodic flights.** A pumping cycle repeats roughly every 190 s, so shifting by one cycle reproduces the pattern and the correlation curve carries several near-equal peaks. Always inspect the second-highest peak; if it is close to the first, the result carries no information. Two of our attempts failed exactly this way.

**A result that lands on the search boundary is not a result.** It means the optimum lies outside the window, or there is no optimum at all.

**Anchors derived from filenames need their own verification.** The labelled static frames placed all seven anchors in the correct phase for three different offsets. Combined with `kite_height` the three were mutually inconsistent, which indicates the frame numbering of `KCU_1.MP4` does not match Pim's photogrammetry indexing.

**Validating the data does not validate the drawing.** Check 2b proved that `kite_0_yaw` matches the direction of travel, and that was taken as proof that the 3D arrows were right. They were not: a separate indexing bug mirrored east and west in the drawing. Test the code path that produces what the user sees against an independent reference, not only its input.

**The sensor frame is not the kite frame.** The logged attitude sits about (87°, 7°, 5°) from the kite's aerodynamic frame in roll, pitch and yaw. Heading-only checks cannot reveal this, because the nose axis is shared. The tether direction can. A one-axis correction is not enough: the first fix assumed roll alone and left the other two axes wrong.

**B-frames make seeking erratic.** With B-frames in the proxy, OpenCV's frame-accurate seek usually took 25 ms but occasionally 300–450 ms. Averages hide this; always look at the worst case.

**Interpolating an angle across ±180° sweeps a full turn.** Between two telemetry samples the kite appeared to flip over for a single frame, whenever its heading crossed the wrap. Unwrap an angle series before interpolating it.

**A figure can come out narrower than asked for.** The scrolling plot is drawn into one very wide strip. Enlarged to full screen the strip exceeded what the renderer produces, and it returned a shorter image rather than an error, so the panel went blank part way through the flight. Draw only around the current moment, and be suspicious of any dimension in the tens of thousands of pixels.

**Default arguments are frozen at import.** `build(dataset_csv=DATASET_CSV)` captures the value when the module loads, so a setting changed afterwards was silently ignored. Read settings inside the function instead.

**A trailing comment in `.gitignore` is part of the pattern.** `!data/demo.csv   # keep this one` matches nothing, and the file stays excluded without any warning. Comments need their own line. Likewise `output/` excludes the directory itself, and nothing inside an excluded directory can be brought back with a negation — use `output/*` with `!output/.gitkeep`.

**A timer outlives its window.** Closing the viewer during playback left one more frame scheduled, which then painted into a canvas that no longer existed. Stop the clock on close, and check the window still exists before drawing.

### Environment

**Matplotlib.** `fig.canvas.tostring_rgb()` was removed in recent versions; use `np.asarray(fig.canvas.buffer_rgba())[..., :3]`. Panel figures must be created with `Figure` + `FigureCanvasAgg`, not `plt.figure()`, or an interactive backend opens each one as a separate window.

**NumPy 2.x.** `ndarray.ptp()` was removed; use `np.ptp(arr)`.

**ffmpeg.** `-vsync cfr` was replaced by `-fps_mode cfr`.

**Python version.** Several dependencies had no wheels for Python 3.14 at the time of writing. A 3.12 virtual environment avoids the problem. `open3d`, `pyvista` and `pymesh` are optional in the original repository and are not needed for KAT.

## 12. Files, workflow and open items

### The files

| File | Role |
| --- | --- |
| `kat.py` | Engine: sources, panels, grid, frame composition, video writer. Rarely edited. |
| `config.py` | All settings and the `build()` function that assembles the panels. This is the file to edit. |
| `render.py` | Entry point producing the mp4. |
| `viewer.py` | Entry point opening the interactive window. |
| `kat_inspect.py` | Survey a flight log: columns, sampling rate, phases, force units. |
| `kat_sync.py` | `meta` reads camera frame rates; `anchor` scans offset candidates from labelled static frames. |
| `kat_validate.py` | Four independent cross-checks on position, attitude and physics. |
| `kat_dataset.py` | Build a synthetic time series from the real static point cloud, for the deformation panels. |
| `viewer_tk.py` | Alternative viewer built on Tk and Pillow: same features, plainer controls. |
| `flights/` | One file per flight: the measured numbers and the file paths. Selected with --flight. |
| `scripts/` | Preprocessing: make\_proxies.py builds the proxies, find\_takeoff.py locates take-off in the log and in each video. |
| `models/` | V3 wing geometry, from awegroup/TUDELFT\_V3\_KITE, used for the 3D kite model. |
| `data/` | The demonstration point cloud only. Not a measurement; see the repository section. |

### Workflow for a new flight

1. `python kat_inspect.py <flight_log.csv>` — learn the columns, sampling rate, phases and units.
2. Derive take-off and landing from `kite_height`.
3. `ffprobe` each camera file; check whether `r_frame_rate` and `avg_frame_rate` agree.
4. Generate a proxy per camera at constant rate and at the panel width.
5. `python kat_sync.py meta <proxies>` — confirm the proxies came out as intended.
6. Find take-off in each video by extracting frames and bisecting.
7. Copy `flights/_template.py` to `flights/fYYYY_MM_DD.py` and fill in the take-off and landing seconds, the proxy paths with their take-off seconds, and the mounting correction. The clock origin and every video offset follow from them. Run it with `--flight YYYY-MM-DD`; no other file changes.
8. `python kat_validate.py <flight_log.csv>` — confirm the data behaves as expected before trusting the panels.
9. `python viewer.py` to explore, `python render.py` to export.

### Open items

**From Pim.** The photogrammetry time series (the combined dataset from `combine_all_results.py`) for at least one manoeuvre. Also which merge of the KCU footage the photogrammetry frame indices refer to, since `frame_7182` does not line up with `KCU_1.MP4` under any offset consistent with `kite_height`.

**From Roland.** Which sensor streams belong in the time-signal panels — tether force and the flow angles are placeholders. And where the EKF-processed data for this flight should come from, for the raw-versus-filtered comparison.

**From the group.** What `kite_heading` actually measures, given that it disagrees with the NED course derived from the velocity vector.

**Also open.** Whether a 3D model of the V3 kite exists as an `.obj` — the repository's `.gitignore` excludes `*.obj`, so any model Pim used is not in it; until one is found the built-in canopy is shown. And confirmation that the times in the Flightdata README are CEST rather than UTC.

**To report.** The mounting offset between the attitude sensor's frame and the kite's aerodynamic frame — about (87°, 7°, 5°) in roll, pitch and yaw (section 10, check 5) — and whether it is exactly (90°, 0°, 0°). It affects every analysis that uses `kite_0_roll` or `kite_0_pitch`, not only KAT.

## 13. The repository

KAT is published as `awegroup/KAT-Kite-Analysis-Terminal` under the MIT licence, with the copyright notice carrying both this work and P. J. Haanen's original, as required when adapting MIT-licensed code. `CITATION.cff` records the authors and the three works it builds on: the photogrammetry thesis, the public flight log and the V3 geometry.

### What is in it, and what is not

The repository holds code, documentation, the V3 wing geometry and one small data file. It does **not** hold flight logs, camera footage or rendered output: the log is public elsewhere, and the footage cannot be published at all.

`data/demo_deformation.csv` is the exception, and it is a demonstration rather than a measurement. The photogrammetry time series is not public, so the deformation panels would otherwise have nothing to show. The file was built by sampling marker positions from the V3 CAD model — a leading-edge row and six markers on each of eight struts, 61 in total — and moving them with a synthetic rotation and twist, 20 s at 30 Hz. It demonstrates what the panels do and must not be analysed or quoted.

### Confidentiality of the camera footage

The flight videos, in particular the one from the kite control unit, cannot be made public: the kite carries the logo of a car manufacturer involved in an earlier funded project under which it was flown. This extends to anything derived from them — extracted frames, rendered composites, and screenshots of the viewer, whose camera panels show the footage. `.gitignore` therefore excludes video and image files by default, so committing one takes a deliberate `git add -f`.

### Reproducing the environment

Python 3.12 with the pinned versions in `requirements.txt` is the tested combination; 3.14 also works. `ffmpeg` and `ffprobe` are separate programs, not Python packages, so `pip` cannot bring them — `opencv-python` bundles ffmpeg's libraries, which is why KAT reads video without them, but `scripts/make_proxies.py` calls the programs. They are needed once per campaign, to build the proxies; the viewer and the renderer do not use them.

**Remaining development.** The wing-mounted camera channel is wired up but has no file. Selector buttons exist in the viewer; a zoom capability on the plots does not. The offset carries about ±1 s of uncertainty, which a synchronisation marker at the start of the next campaign would eliminate.

### Recommendations for the next campaign

- Record a visible synchronisation marker at the start — a flash, a horn or a stopwatch in front of both cameras.
- Make sure every camera records at a constant frame rate; check `r_frame_rate` against `avg_frame_rate` on site if possible.
- A higher frame rate helps: at 60 fps the temporal resolution is 17 ms instead of 33 ms, and motion blur is lower for photogrammetry.
- Set the camera clocks from GPS so the metadata timestamps are usable as a cross-check.
