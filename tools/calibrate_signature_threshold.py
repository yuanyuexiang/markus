"""Calibrate SigNet distance threshold on your own signature pairs.

Usage:
  /path/to/python tools/calibrate_signature_threshold.py --pairs pairs.csv

pairs.csv format (UTF-8):
  label,template,query
  1,test_images/signature_template.png,test_images/signature_real.png
  0,test_images/signature_template.png,test_images/signature_fake.png

Where:
  - label: 1 for genuine (same writer), 0 for impostor (different writer)
  - template/query: paths to images

The script prints basic stats, ROC-like sweep results, and a recommended distance
threshold for a target FAR.

Note: This uses the same preprocessing pipeline as the backend (robust+clean
when available) by calling `backend/main.py` helpers.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def _load_pairs(csv_path: Path) -> List[Tuple[int, Path, Path]]:
    rows: List[Tuple[int, Path, Path]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
                template = Path(row["template"]).expanduser()
                query = Path(row["query"]).expanduser()
            except Exception as e:
                raise ValueError(f"Invalid row at line {i}: {row}") from e
            if label not in (0, 1):
                raise ValueError(f"label must be 0/1 at line {i}")
            rows.append((label, template, query))
    if not rows:
        raise ValueError("No rows found in pairs csv")
    return rows


def _sweep_thresholds(labels: np.ndarray, distances: np.ndarray, thresholds: np.ndarray):
    # FAR: impostor accepted
    # FRR: genuine rejected
    results = []
    for t in thresholds:
        pred_accept = distances <= t
        impostor = labels == 0
        genuine = labels == 1
        far = (pred_accept & impostor).sum() / max(1, impostor.sum())
        frr = ((~pred_accept) & genuine).sum() / max(1, genuine.sum())
        results.append((t, far, frr))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="CSV file with label,template,query")
    ap.add_argument("--target-far", type=float, default=0.01, help="Target FAR, e.g. 0.01")
    ap.add_argument("--clean-mode", default="conservative", choices=["conservative", "aggressive"])
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    pairs = _load_pairs(pairs_path)

    # Import backend pipeline (adds backend to sys.path)
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "backend"))
    import main as backend_main  # type: ignore

    labels = []
    dists = []
    for label, template_path, query_path in pairs:
        t_img = Image.open(template_path).convert("L")
        q_img = Image.open(query_path).convert("L")
        res = backend_main.compute_signet_similarity(
            t_img,
            q_img,
            enable_clean=True,
            clean_mode=args.clean_mode,
        )
        if not res:
            raise RuntimeError(f"SigNet pipeline failed for {template_path} vs {query_path}")
        labels.append(label)
        dists.append(float(res["distance"]))

    labels_np = np.asarray(labels, dtype=np.int64)
    dists_np = np.asarray(dists, dtype=np.float64)

    g = dists_np[labels_np == 1]
    i = dists_np[labels_np == 0]

    print(f"pairs: {len(labels_np)} (genuine={len(g)}, impostor={len(i)})")
    if len(g):
        print(f"genuine distance: mean={g.mean():.6f} p50={np.median(g):.6f} p95={np.percentile(g,95):.6f}")
    if len(i):
        print(f"impostor distance: mean={i.mean():.6f} p05={np.percentile(i,5):.6f} p50={np.median(i):.6f}")

    # Sweep thresholds across observed range
    lo = float(dists_np.min())
    hi = float(dists_np.max())
    thresholds = np.linspace(lo, hi, num=200)
    sweep = _sweep_thresholds(labels_np, dists_np, thresholds)

    # Find threshold that meets target FAR with minimal FRR
    target_far = float(args.target_far)
    feasible = [(t, far, frr) for (t, far, frr) in sweep if far <= target_far]
    if feasible:
        best = min(feasible, key=lambda x: x[2])
        t, far, frr = best
        print(f"recommended distance threshold @FAR<={target_far:.4f}: t={t:.6f} FAR={far:.4f} FRR={frr:.4f}")
    else:
        # fallback: choose minimal FAR point
        t, far, frr = min(sweep, key=lambda x: x[1])
        print(f"no threshold meets FAR<={target_far:.4f}; best FAR point: t={t:.6f} FAR={far:.4f} FRR={frr:.4f}")

    # Print a small table around operating points
    print("\nthreshold  FAR     FRR")
    for t, far, frr in sweep[::20]:
        print(f"{t:9.6f}  {far:6.4f}  {frr:6.4f}")


if __name__ == "__main__":
    main()
