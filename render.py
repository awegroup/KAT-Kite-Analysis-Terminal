#!/usr/bin/env python3
"""
KAT - Kite Analysis Terminal
Writes the video output. Settings live in config.py.

    python render.py
    $env:DURATION_S=100; python render.py     # override the duration once
"""

import config as cfg
from kat import render_video

if __name__ == "__main__":
    cfg.require_inputs()
    slots, (n_rows, n_cols, size), telemetry, sources = cfg.build()
    render_video(
        slots, n_rows=n_rows, n_cols=n_cols,
        t_start=telemetry.t_start + cfg.WARMUP_S,
        t_end=telemetry.t_start + cfg.WARMUP_S + cfg.DURATION_S,
        fps=cfg.FPS, out_path=cfg.OUT, out_size=size,
        footer=cfg.make_footer(telemetry),
    )
    for s in sources:
        s.release()
