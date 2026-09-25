#!/usr/bin/env python3
"""
KAT - turn raw camera recordings into proxies KAT can use.

    python scripts/make_proxies.py KCU_1.MP4 Ground_Station.MP4
    python scripts/make_proxies.py *.MP4 --width 1280 --out output

Why a proxy is needed at all
----------------------------
KAT finds a frame with one calculation: frame = time x frame rate. That is
only valid when the frame rate is constant and known. Cameras do not always
deliver that: one recording in the reference campaign reported no real
r_frame_rate and an average of 19.797 fps, and forcing it to a constant rate
required duplicating 480 frames and dropping 34.

The proxy solves three more things at the same time:
  * decoding cost  - a 5K H.265 source is far slower to decode than a
                     1280 px H.264 proxy, and the panel is ~1280 px wide
  * seeking cost   - short keyframe intervals and no B-frames turn a jump on
                     the time slider from ~900 ms into ~20 ms
  * the originals stay untouched; the proxy is a working copy

The encoder flags, and why
--------------------------
  -r / -fps_mode cfr   force a constant frame rate; the source rate is kept
                       when it is already constant, so frame numbers survive
  -g 10                a keyframe every 10 frames, so a jump decodes at most 9
  -bf 0                no B-frames: with them, frames are stored out of display
                       order and OpenCV's seek overshoots, steps back and
                       retries - usually 25 ms, occasionally 450 ms
  -tune fastdecode     switch off the tools that are expensive to decode
  -crf 20              visually near-lossless
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def probe(path):
    """Frame rate, frame count and resolution, straight from the container."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True).stdout
    st = (json.loads(out).get("streams") or [{}])[0]

    def frac(value):
        try:
            a, b = value.split("/")
            return float(a) / float(b) if float(b) else 0.0
        except Exception:
            return 0.0

    r, avg = frac(st.get("r_frame_rate", "0/0")), frac(st.get("avg_frame_rate", "0/0"))
    return {
        "r_rate": r, "avg_rate": avg,
        "width": int(st.get("width", 0)), "height": int(st.get("height", 0)),
        "frames": int(st.get("nb_frames", 0) or 0),
        "duration": float(st.get("duration", 0) or 0),
        # a real constant-rate file reports the same sensible value twice
        "constant": r > 1 and avg > 1 and abs(r - avg) / max(r, 1e-9) < 0.001,
    }


def target_rate(info):
    """Keep the source rate when it is constant; otherwise round the average.

    Matching a constant source exactly means ffmpeg copies and drops nothing,
    so frame numbers stay comparable with anything derived from the original.
    """
    if info["constant"]:
        return f"{round(info['r_rate'] * 1000)}/1000", info["r_rate"]
    rate = max(1.0, round(info["avg_rate"]))
    return str(int(rate)), float(rate)


def build(path, out_dir, width, crf, gop, dry_run=False):
    info = probe(path)
    if not info["width"]:
        print(f"  !! could not read {path}")
        return None
    rate_str, rate = target_rate(info)
    name = os.path.splitext(os.path.basename(path))[0].lower() + "_proxy.mp4"
    dst = os.path.join(out_dir, name)

    print(f"\n{path}")
    print(f"  source   : {info['width']}x{info['height']}, "
          f"r_frame_rate {info['r_rate']:.4f}, avg {info['avg_rate']:.4f}")
    print(f"  verdict  : {'constant rate' if info['constant'] else 'VARIABLE rate'}")
    print(f"  proxy    : {width} px wide at {rate} fps -> {dst}")

    cmd = ["ffmpeg", "-v", "error", "-stats", "-y", "-i", path,
           "-vf", f"scale={width}:-2", "-r", rate_str, "-fps_mode", "cfr",
           "-c:v", "libx264", "-crf", str(crf), "-g", str(gop), "-bf", "0",
           "-tune", "fastdecode", "-an", dst]
    if dry_run:
        print("  command  : " + " ".join(cmd))
        return dst
    subprocess.run(cmd, check=True)

    after = probe(dst)
    print(f"  result   : {after['width']}x{after['height']}, {after['r_rate']:.4f} fps")
    if info["constant"] and info["frames"] and after["frames"]:
        same = info["frames"] == after["frames"]
        print(f"  frames   : {info['frames']} -> {after['frames']}"
              f"{'  (unchanged, frame numbers still line up)' if same else '  (CHANGED)'}")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="raw camera files")
    ap.add_argument("--out", default="output", help="where to write the proxies")
    ap.add_argument("--width", type=int, default=1280,
                    help="proxy width; match the video panel, OUT_W / n_cols * colspan")
    ap.add_argument("--crf", type=int, default=20, help="quality, lower is better")
    ap.add_argument("--gop", type=int, default=10, help="keyframe interval in frames")
    ap.add_argument("--dry-run", action="store_true", help="only print the commands")
    args = ap.parse_args()

    absent = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if absent:
        sys.exit(
            f"{' and '.join(absent)} not found on PATH.\n\n"
            "These are separate programs, not Python packages, so pip does not\n"
            "bring them with the rest. Install them one of these ways:\n\n"
            "    winget install Gyan.FFmpeg      # Windows\n"
            "    brew install ffmpeg             # macOS\n"
            "    sudo apt install ffmpeg         # Debian / Ubuntu\n"
            "    pip install static-ffmpeg       # into this virtual environment,\n"
            "                                    # then run: static_ffmpeg_paths\n\n"
            "Open a NEW terminal afterwards so the changed PATH is picked up,\n"
            "then check with:  ffmpeg -version")
    os.makedirs(args.out, exist_ok=True)

    made = [build(v, args.out, args.width, args.crf, args.gop, args.dry_run)
            for v in args.videos]
    made = [m for m in made if m]
    if made and not args.dry_run:
        print("\nNext: find the frame in each proxy where the kite leaves the ground,")
        print("and put that second into VIDEOS[...]['takeoff_s'] in your flights/ file.")
        print("To step through candidates:")
        print('  ffmpeg -v error -y -ss <seconds> -i "<proxy>" -vframes 1 out.png')


if __name__ == "__main__":
    main()
