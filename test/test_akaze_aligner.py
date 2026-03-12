#!/usr/bin/env python3
"""
Benchmark test for AkazeAligner: CPU/memory usage and computation time.
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import cv2
import psutil

import akaze
from akaze import AKAZEOptions, AKAZE, Matcher

_SCRIPT_DIR = Path(__file__).resolve().parent


def _monitor_loop(stop_event, cpu_peak, mem_peak, process):
    """Background thread to track peak CPU and memory usage."""
    while not stop_event.is_set():
        if process is not None:
            cpu_peak[0] = max(cpu_peak[0], process.cpu_percent())
            mem_info = process.memory_info()
            mem_peak[0] = max(mem_peak[0], mem_info.rss)
        stop_event.wait(0.2)


def _profile_one(template, image, matcher, nndr=0.80, akaze_cache=None):
    """Profile a single pair, returning per-stage timings (ms) and the result."""
    timings = {}
    perf = time.perf_counter
    if akaze_cache is None:
        akaze_cache = {}

    h, w = template.shape
    t0 = perf()
    key_t = (w, h)
    if key_t in akaze_cache:
        evolution_t = akaze_cache[key_t]
    else:
        options_t = AKAZEOptions()
        options_t.setWidth(w)
        options_t.setHeight(h)
        evolution_t = AKAZE(options_t)
        akaze_cache[key_t] = evolution_t
    timings["akaze_init_tpl"] = (perf() - t0) * 1000

    h2, w2 = image.shape
    t0 = perf()
    key_i = (w2, h2)
    if key_i in akaze_cache:
        evolution_i = akaze_cache[key_i]
    else:
        options_i = AKAZEOptions()
        options_i.setWidth(w2)
        options_i.setHeight(h2)
        evolution_i = AKAZE(options_i)
        akaze_cache[key_i] = evolution_i
    timings["akaze_init_img"] = (perf() - t0) * 1000

    t0 = perf()
    evolution_t.Create_Nonlinear_Scale_Space(template)
    timings["scale_space_tpl"] = (perf() - t0) * 1000

    t0 = perf()
    desc_t, kpts_t = evolution_t.Compute_Descriptors()
    timings["descriptors_tpl"] = (perf() - t0) * 1000

    t0 = perf()
    evolution_i.Create_Nonlinear_Scale_Space(image)
    timings["scale_space_img"] = (perf() - t0) * 1000

    t0 = perf()
    desc_i, kpts_i = evolution_i.Compute_Descriptors()
    timings["descriptors_img"] = (perf() - t0) * 1000

    t0 = perf()
    dmatches = matcher.BFMatch(desc_t, desc_i)
    timings["bf_match"] = (perf() - t0) * 1000

    t0 = perf()
    dmatches = np.asarray(dmatches)
    if dmatches.ndim == 1:
        dmatches = dmatches.reshape(-1, 8)
    dist0 = dmatches[:, 3]
    dist1 = dmatches[:, 7]
    mask = (dist1 > 1e-10) & (dist0 < nndr * dist1)
    valid = np.where(mask)[0]
    pts_t = kpts_t[:, :2].astype(np.float32)
    pts_i = kpts_i[:, :2].astype(np.float32)
    qidx = dmatches[valid, 0].astype(np.int32)
    tidx = dmatches[valid, 1].astype(np.int32)
    pts0 = pts_t[qidx]
    pts1 = pts_i[tidx]
    timings["nndr_filter"] = (perf() - t0) * 1000

    t0 = perf()
    H, status = cv2.findHomography(pts0, pts1, cv2.RANSAC, 2.5,
                                   maxIters=2000, confidence=0.995)
    timings["ransac"] = (perf() - t0) * 1000

    if H is None:
        H = np.eye(3, dtype=np.float32)
    return timings, H.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Benchmark AkazeAligner (CPU)")
    parser.add_argument(
        "--image",
        type=str,
        default=str(_SCRIPT_DIR / "image.png"),
        help="Input image path (default: <script_dir>/image.png)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=str(_SCRIPT_DIR / "template.png"),
        help="Template image path (default: <script_dir>/template.png)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    template_path = Path(args.template).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

    batch = args.batch
    template_batch = np.tile(tpl[np.newaxis, :, :], (batch, 1, 1))
    image_batch = np.tile(img[np.newaxis, :, :], (batch, 1, 1))

    aligner = akaze.AkazeAligner()

    # Warmup
    for _ in range(3):
        _ = aligner.find_transform(template_batch, image_batch)

    process = psutil.Process()
    cpu_peak = [0.0]
    mem_peak = [0]

    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_loop,
        args=(stop_event, cpu_peak, mem_peak, process),
    )
    monitor.start()

    t0 = time.perf_counter()
    result = aligner.find_transform(template_batch, image_batch)
    t1 = time.perf_counter()
    stop_event.set()
    monitor.join()

    elapsed_ms = (t1 - t0) * 1000
    mem_mb = mem_peak[0] / (1024 * 1024)

    warp_matrix = result["warp_matrix"]
    if hasattr(warp_matrix, "numpy"):
        warp_matrix = warp_matrix.numpy()

    print("--- AkazeAligner benchmark (CPU) ---")
    print(f"  Batch size:        {batch}")
    print(f"  Image shape:       {img.shape}")
    print(f"  Template shape:    {tpl.shape}")
    print(f"  CPU peak (%):      {cpu_peak[0]:.1f}")
    print(f"  Memory peak (MB):  {mem_mb:.2f}")
    print(f"  Time (ms):         {elapsed_ms:.2f}")
    print(f"  Time per sample:   {elapsed_ms / batch:.2f} ms")
    print("  Warp matrix (3x3 per sample):")
    np.set_printoptions(precision=4, suppress=True)
    for i in range(batch):
        print(f"    [{i}]:\n{warp_matrix[i]}")

    # --- Per-stage timing breakdown (single pair, averaged over batch) ---
    from akaze.akaze import _to_gray_float32
    t_all = _to_gray_float32(template_batch)
    i_all = _to_gray_float32(image_batch)
    matcher = Matcher()

    all_timings = []
    akaze_cache = {}
    for b in range(batch):
        timings, _ = _profile_one(t_all[b], i_all[b], matcher, akaze_cache=akaze_cache)
        all_timings.append(timings)

    keys = list(all_timings[0].keys())
    print("\n--- Per-stage timing breakdown (avg over batch, ms) ---")
    total = 0.0
    for k in keys:
        avg = np.mean([t[k] for t in all_timings])
        total += avg
        print(f"  {k:24s}: {avg:8.2f} ms")
    print(f"  {'TOTAL':24s}: {total:8.2f} ms")


if __name__ == "__main__":
    main()
