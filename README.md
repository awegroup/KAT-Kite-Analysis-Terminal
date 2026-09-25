# KAT — Kite Analysis Terminal

A viewer for airborne wind energy flight campaigns. It puts camera video, the
kite's position and attitude in 3D, and sensor time histories side by side on
a single time axis, so a moment in the flight can be looked at from every
source at once.

Built for the TU Delft AWE group, using the 9 October 2025 Kitepower campaign
(25 m² V3 kite) as the reference flight.

---

## What it shows

**Default layout — four panels**, and it works for any flight, because it needs
nothing but a flight log and the videos:

| | |
|---|---|
| airborne camera (KCU, or a wing-mounted one) | 3D position and attitude, with trail, tether and kite model |
| ground station camera | sensor time history, selectable from a drop-down |

A status bar along the bottom carries the UTC date and time plus live values.

**Extended layout** adds three projection views and a 3D view of the
photogrammetry point cloud. It needs photogrammetry data, so it is optional.

## Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

### ffmpeg

`ffmpeg` and `ffprobe` are separate programs, not Python packages, so `pip`
cannot bring them. (`opencv-python` bundles ffmpeg's *libraries*, which is why
KAT can read video without them — but the proxy tools call the programs.)

```powershell
winget install Gyan.FFmpeg          # Windows
choco install ffmpeg                # Windows, if you use Chocolatey
brew install ffmpeg                 # macOS
sudo apt install ffmpeg             # Debian / Ubuntu
```

Open a new terminal afterwards, so the changed PATH is picked up, and check:

```bash
ffmpeg -version
ffprobe -version
```

Without a package manager on Windows: download a build from
[gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (the "essentials" zip), unpack
it somewhere permanent such as `C:\ffmpeg`, then add `C:\ffmpeg\bin` to PATH —
*Settings → System → About → Advanced system settings → Environment
Variables → Path → New*.

Alternatively `pip install static-ffmpeg` puts both programs inside the
virtual environment; run `static_ffmpeg_paths` once to register them.

ffmpeg is only needed to build the proxies, i.e. once per campaign. The viewer
and the renderer do not use it.

## Confidentiality of the camera footage

The flight videos, in particular the one from the kite control unit, **must not
be published**. The kite carries the logo of a car manufacturer involved in an
earlier funded project under which it was flown, and that footage cannot be
made public.

In practice:

- no video file, no extracted frame and no rendered composite goes into this
  repository, or into a paper, slide deck or issue thread;
- screenshots of the viewer are risky, because the camera panels show the
  footage. If one is needed, check the whole frame, not just the middle, and
  crop or cover anything that shows the logo;
- `.gitignore` excludes video and image files by default, so adding one takes
  a deliberate `git add -f`. Think before you do.

The flight log itself is public
([awegroup/Flightdata09102025](https://github.com/awegroup/Flightdata09102025)),
as are the kite geometry and everything in this repository.

## Data

No flight data is stored in this repository; it is large and belongs to the
campaign. For the reference flight:

- **Flight log** — `awegroup/Flightdata09102025`, unpack the log CSV into
  `flightdata/`.
- **Videos** — the raw camera recordings, from the campaign. Turn them into
  proxies first (see below); the proxies land in `output/`.

### Demonstration data

`data/demo_deformation.csv` is the one data file in this repository. It exists
so the deformation panels can be shown without the photogrammetry dataset,
which is not public: marker positions sampled from the V3 CAD model and moved
by a synthetic rotation and twist, 20 s at 30 Hz.

**It is a demonstration, not a measurement.** Nothing about it should be
analysed or quoted. Point `DATASET_CSV` in the flight file at the real dataset
when it becomes available. The other panels — cameras, 3D position, signals —
always use the real flight data.

## Quick start

```bash
python kat_inspect.py flightdata/<log>.csv                    # know the log
python scripts/find_takeoff.py log flightdata/<log>.csv       # take-off, landing
python scripts/make_proxies.py KCU_1.MP4 Ground_Station.MP4   # once per campaign
python kat_validate.py flightdata/<log>.csv --windows 30      # check the data
python viewer.py                                              # explore
python render.py                                              # write a video
```

### Viewer controls

| | |
|---|---|
| Space | play / pause |
| ← → · Shift + ← → | step 1 s · 0.1 s |
| `[` `]` | slower / faster playback (0.1× … 4×) |
| double-click, or 1 … 9 | enlarge a panel; Esc returns |
| drag · scroll · R | in an enlarged 3D panel: rotate · zoom · reset |
| F | full-screen window |

`viewer_tk.py` is an alternative window with the same features and plainer
controls, in case the matplotlib one misbehaves on a machine.

## Using it on another flight

Everything specific to a flight lives in one file under `flights/`. Copy
`flights/_template.py`, fill it in, and run:

```bash
python viewer.py --flight 2026-03-15
python viewer.py --list-flights          # what is defined here
```

`--flight` works on every entry point (`viewer.py`, `viewer_tk.py`,
`render.py`); `KAT_FLIGHT` does the same through the environment. If a needed
file is missing, KAT says which one and where the path came from instead of
raising a traceback.

No other file changes. The steps for filling it in, with the reasoning behind
each, are in [docs/pipeline.md](docs/pipeline.md):

1. `python kat_inspect.py <flight log>` — columns, sampling rate, phases, units.
2. `python scripts/find_takeoff.py log <flight log>` — take-off and landing.
3. `python scripts/make_proxies.py <videos>` — constant-rate proxies.
4. `python scripts/find_takeoff.py video <proxy>` — the lift-off second per camera.
5. `python kat_validate.py <log> --windows 30` — the sensor mounting correction.

## Layout

```
kat.py            engine: sources, panels, grid, frame composition
config.py         tool-wide settings; loads the selected flight
flights/          one file per flight - paths, take-off instants, corrections
viewer.py         interactive window (matplotlib)
viewer_tk.py      alternative interactive window (Tk)
render.py         writes an mp4
kat_inspect.py    survey a flight log
kat_sync.py       camera frame rates, offset candidates
kat_validate.py   five independent cross-checks on the data
kat_dataset.py    synthetic point-cloud time series for the deformation panels
scripts/          preprocessing (proxy generation)
models/           kite geometry (V3 CAD surface, exported from SpaceClaim)
docs/             pipeline, pitfalls and validation findings
```

## Findings worth knowing

These came out of building the tool and matter to anyone using the same data:

- **The attitude sensor is not aligned with the kite.** For the reference
  flight the logged frame sits about (87°, 7°, 5°) in roll, pitch and yaw from
  the kite's own axes — steady across the flight (R = 0.98), so it is a
  mounting offset, and `KITE_MOUNT_RPY` applies it to the drawing.
  Heading is unaffected, which is why a yaw-only check does not reveal it, but
  anything using `kite_0_roll` or `kite_0_pitch` must apply the same correction.
- **`kite_heading` does not agree with the NED course.** The velocity vector
  (`kite_0_vx/vy`) is the reliable reference.
- **Tether force is logged in kilograms**, not newtons.
- **Times in the Flightdata README appear to be CEST**, not UTC as labelled.

`kat_validate.py` reproduces all of these from the data.

## Acknowledgements

KAT grew out of `video_generator.py` in
[pimjhaanen/photogrammetry_thesis](https://github.com/pimjhaanen/photogrammetry_thesis)
by **P.J. Haanen** (MSc thesis, TU Delft), used and adapted under the MIT
licence. The engine here was rewritten to support several cameras, a
configurable grid and scrolling time histories, but the following came from
that work and shaped this one:

- the idea of a single composite frame carrying every view at one instant;
- the front / side / bottom projection views of the photogrammetry point
  cloud and the live metrics block, which KAT keeps as panels;
- the static photogrammetry frames and stereo calibration used to build and
  test the deformation panels;
- the column conventions of the "complete dataset" format.

The reference flight data (9 October 2025) is from the Kitepower campaign,
published as [awegroup/Flightdata09102025](https://github.com/awegroup/Flightdata09102025).
The V3 wing geometry in `models/` comes from
[awegroup/TUDELFT_V3_KITE](https://github.com/awegroup/TUDELFT_V3_KITE),
developed by G. Lebesque (2020) and adjusted by J.A.W. Poland (2025).

Work carried out in the Airborne Wind Energy group at TU Delft, supervised by
**Roland Schmehl**.

## Licence

MIT, see [`LICENSE`](LICENSE). The copyright notice carries both this work and
P.J. Haanen's original, as required when adapting MIT-licensed code:

```
Copyright (c) 2026 Abdullah Çantav, TU Delft
Copyright (c) 2026 PJ Haanen
```
