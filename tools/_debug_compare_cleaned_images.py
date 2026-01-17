from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def to_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def ink_ratio(arr: np.ndarray, thr: int = 200) -> float:
    # background ~255 (white), ink darker
    return float((arr < thr).mean())


def compare_pair(t_path: Path, q_path: Path) -> None:
    print(f"\nPAIR {t_path.name} {q_path.name}")
    print(" exists", t_path.exists(), q_path.exists())
    if not (t_path.exists() and q_path.exists()):
        return

    t_arr = to_gray(t_path)
    q_arr = to_gray(q_path)

    print(" md5", md5(t_path), md5(q_path))
    print(" shape", t_arr.shape, q_arr.shape)
    print(" t mean/std/ink", float(t_arr.mean()), float(t_arr.std()), ink_ratio(t_arr))
    print(" q mean/std/ink", float(q_arr.mean()), float(q_arr.std()), ink_ratio(q_arr))

    if t_arr.shape != q_arr.shape:
        return

    diff = np.abs(t_arr.astype(np.int16) - q_arr.astype(np.int16))
    print(" equal", bool(np.array_equal(t_arr, q_arr)))
    print(" diff mean/max", float(diff.mean()), int(diff.max()))
    print(" diff pct>5", float((diff > 5).mean()))
    print(" diff pct>20", float((diff > 20).mean()))


def main() -> None:
    debug_dir = Path("/Users/yuanyuexiang/Desktop/workspace/markus/uploaded_samples/debug")

    # Compare matching timestamps if present
    prefix = "template_cleaned_"
    timestamps = []
    for p in debug_dir.glob("template_cleaned_*.png"):
        name = p.name
        if not (name.startswith(prefix) and name.endswith(".png")):
            continue
        timestamps.append(name[len(prefix) : -len(".png")])
    timestamps = sorted(set(timestamps))
    if not timestamps:
        print("No cleaned images found in", debug_dir)
        return

    for ts in timestamps:
        t = debug_dir / f"template_cleaned_{ts}.png"
        q = debug_dir / f"query_cleaned_{ts}.png"
        compare_pair(t, q)


if __name__ == "__main__":
    main()
