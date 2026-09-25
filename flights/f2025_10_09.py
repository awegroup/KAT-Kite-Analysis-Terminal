"""
Flight of 9 October 2025 - Kitepower test site, Bangor Erris, Ireland.
25 m2 V3 kite. Log covers 3997 s; the kite is airborne from 2208 s to 3930 s.

Every value here was measured, not guessed:
  take-off / landing   kat_inspect.py plus the first kite_height above 30 m
  video take-off       the frame in each video where the kite leaves the ground
  mounting correction  kat_validate.py check 5 (median over 30 s windows)
"""

FLIGHT_CSV = "flightdata/log.csv"

TELEMETRY_TAKEOFF_S = 2206.3
TELEMETRY_LANDING_S = 3930.3
PRE_TAKEOFF_S = 150.0
POST_LANDING_S = 60.0

VIDEOS = {
    "kcu":  {"path": "output/kcu_1_proxy.mp4", "takeoff_s": 168.5, "rotate_180": True},
    "wing": {"path": "output/wing_proxy.mp4", "takeoff_s": None, "rotate_180": False},
    "gs":   {"path": "output/gs_proxy.mp4", "takeoff_s": 399.5, "rotate_180": False},
}
GS_KEY = "gs"

# sensor-to-kite mounting correction (roll, pitch, yaw in degrees)
KITE_MOUNT_RPY = (87.1, 7.2, 5.3)

# Photogrammetry. The real time series for this flight is not public, so the
# repository ships a DEMONSTRATION file instead: marker positions sampled from
# the V3 CAD model, moved by a synthetic rotation and twist. It shows what the
# deformation panels do; it is not measured data and must not be analysed.
# Replace DATASET_CSV with the real dataset when it is available.
DATASET_CSV = "data/demo_deformation.csv"
DATASET_START_S = 2417.0                           # the second of the log its t = 0 means
LIMS = [(-5.0, 5.0), (-1.9, 1.9), (5.5, 8.0)]      # x, y, z of the point cloud
