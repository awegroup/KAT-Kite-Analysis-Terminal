"""
Template for a new flight.

    copy to flights/fYYYY_MM_DD.py, fill in, then:
    python viewer.py --flight YYYY-MM-DD

Nothing outside this file changes. Every value below is measured, and the
command that produces it is written next to it. You do not have to get them
all at once: fill in the log and the videos first, leave the corrections at
zero, look at the result, then refine. docs/pipeline.md is the long version.
"""

# ---------------------------------------------------------------- 1. the log
# Any flight log with the usual columns. KAT tells you which ones it could
# not find rather than failing, so run kat_inspect.py first:
#     python kat_inspect.py flightdata/<log>.csv
FLIGHT_CSV = "flightdata/<log>.csv"

# ------------------------------------------------- 2. take-off and landing
# Seconds from the START OF THE LOG, not from take-off.
#     python scripts/find_takeoff.py log flightdata/<log>.csv
# It prints the first and last moment the kite is above 30 m. Do not take
# these from flight_phase: "unknown" there means "not pumping", not "on the
# ground" - on the reference flight that mistake cost 900 s.
TELEMETRY_TAKEOFF_S = 0.0
TELEMETRY_LANDING_S = 0.0

# How much ground time to keep on either side. Master clock t = 0 sits
# PRE_TAKEOFF_S before take-off, so t never goes negative. Keep it below the
# shortest camera lead (see takeoff_s below) or the view starts blank; KAT
# warns and names the right value if it is too large.
PRE_TAKEOFF_S = 150.0
POST_LANDING_S = 60.0

# ------------------------------------------------------------- 3. the videos
# One entry per camera, any names you like. GS_KEY says which one is the
# ground station; the rest appear in the viewer's camera selector.
#
#   path        the PROXY, not the original recording:
#                   python scripts/make_proxies.py raw/<file>.MP4 --name kcu
#               (a proxy has a constant frame rate and short keyframe
#               intervals, without which seeking takes ~0.5 s per jump)
#
#   takeoff_s   the second OF THAT VIDEO at which the kite leaves the ground:
#                   python scripts/find_takeoff.py video output/kcu_proxy.mp4 --from 0 --to 600
#               look at the frames, narrow the range, repeat until one second
#               is left. This is the only synchronisation input; everything
#               else is derived from it.
#
#   rotate_180  True if the camera is mounted upside down (the KCU one is:
#               its raw frames show the grass at the top)
VIDEOS = {
    "kcu": {"path": "output/kcu_proxy.mp4", "takeoff_s": None, "rotate_180": False},
    "gs":  {"path": "output/gs_proxy.mp4",  "takeoff_s": None, "rotate_180": False},
}
GS_KEY = "gs"

# --------------------------------------------- 4. sensor mounting correction
# The attitude sensor is rarely aligned with the kite's own axes. On the
# reference flight the logged frame sits about a quarter turn away, which
# drew the kite lying on its side. Find the correction with:
#
#     python kat_validate.py flightdata/<log>.csv --windows 30
#
# Read check 5 in the output:
#
#   "as logged"        nose / right wing / down, each against physics
#                      (direction of travel, and the tether direction, which
#                      come from the position data, not from the sensor)
#   "best fit applied" the same three after the correction it suggests
#   the per-window table, and at the end two ready-to-paste lines:
#
#       KITE_MOUNT_RPY = (87.3, 10.4, 5.7)   # single global fit
#       KITE_MOUNT_RPY = (87.1,  7.2, 5.3)   # robust: median of the windows
#
# Take the robust one when both are printed: the global fit is pulled by the
# few windows where the kite really was at a different attitude, typically
# during reel-in.
#
# Then check the per-window table before believing any of it:
#   steady axis    -> a mounting angle, belongs here
#   steady drift   -> sensor drift, a constant will not fix it
#   scatter        -> the kite's own changing attitude; correcting it away
#                     would erase exactly what the 3D panel exists to show
#
# A residual of 10-20 deg after the correction is normal and physical: a kite
# flies at an angle of attack and with some sideslip.
#
# Leave this at zero on the first run - the kite will be drawn in the sensor's
# frame, which is enough to check that everything else lines up.
KITE_MOUNT_RPY = (0.0, 0.0, 0.0)     # roll, pitch, yaw [deg]

# -------------------------------------------- 5. photogrammetry (optional)
# Only for flights that have a point-cloud time series. Without these the
# default four-panel layout works; the deformation panels stay empty.
#   DATASET_START_S  the second of the log the dataset's own t = 0 refers to
#   LIMS             x, y, z axis limits; kat_dataset.py prints suggestions
# DATASET_CSV = "output/complete_dataset.csv"
# DATASET_START_S = 0.0
# LIMS = [(-5, 5), (-2, 2), (3, 8)]
