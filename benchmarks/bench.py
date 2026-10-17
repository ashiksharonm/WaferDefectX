"""Measured latency of the classical-CV localisation pipeline (Python/OpenCV vs the C++ port).

Nothing here is hardcoded: every number printed comes from timing the code on the machine
that runs this script, and the hardware is printed with it. Run it in CI or locally:

    python benchmarks/bench.py --cpp cpp/build/WaferDefectX_Run --out benchmarks/results/bench.json

Both pipelines time preprocessing (blur, CLAHE) plus localisation (Canny, morphology,
contours) on the same committed 800x800 images, excluding image decoding. The C++ figure is
the binary's own "Processing Time" print, so it also excludes process start-up.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def hardware() -> Dict[str, object]:
    return {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "ci_runner": os.environ.get("RUNNER_NAME")
        or ("GitHub Actions" if os.environ.get("GITHUB_ACTIONS") else None),
        "runner_image": os.environ.get("ImageOS"),
    }


def summarize(ms: List[float]) -> Dict[str, float]:
    ordered = sorted(ms)
    return {
        "n": len(ms),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
    }


def bench_python(paths: List[str], repeats: int) -> List[float]:
    from defect_localization import DefectLocalizer
    from preprocessing import Preprocessor

    pre, loc = Preprocessor(), DefectLocalizer()
    warm = cv2.imread(paths[0])
    for _ in range(3):
        loc.localize(pre.process_pipeline(warm)[0])
    out: List[float] = []
    for path in paths:
        image = cv2.imread(path)
        for _ in range(repeats):
            t0 = time.perf_counter()
            enhanced, _ = pre.process_pipeline(image)
            loc.localize(enhanced)
            out.append((time.perf_counter() - t0) * 1000)
    return out


def bench_cpp(paths: List[str], exe: str, repeats: int) -> List[float]:
    out: List[float] = []
    for path in paths:
        for _ in range(repeats):
            res = subprocess.run([exe, path], capture_output=True, text=True, cwd=ROOT, check=False)
            m = re.search(r"Processing Time:\s*([0-9.eE+-]+)\s*ms", res.stdout)
            if m:
                out.append(float(m.group(1)))
    return out


def bench_service(model_path: str, n: int = 50) -> Optional[Dict[str, float]]:
    """Feature extraction + calibrated classifier for one 512x512 wafer (the /predict hot path)."""
    if not os.path.exists(model_path):
        return None
    sys.path.insert(0, ROOT)
    from wafer_inspect.model import WaferModel
    from wafer_inspect.synth import make_dataset

    model = WaferModel.load(model_path)
    images, _, _ = make_dataset(n, seed=5)
    model.predict_image(images[0])
    times = []
    for im in images:
        t0 = time.perf_counter()
        model.predict_image(im)
        times.append((time.perf_counter() - t0) * 1000)
    return summarize(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "synthetic"))
    ap.add_argument("--cpp", default=os.path.join(ROOT, "cpp", "build", "WaferDefectX_Run"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "wafer_model.joblib"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "*.png")))
    if not paths:
        sys.exit(f"no images in {args.data}")
    result: Dict[str, object] = {"hardware": hardware(), "images": len(paths), "repeats": args.repeats}
    result["python_opencv_pipeline"] = summarize(bench_python(paths, args.repeats))
    if os.path.exists(args.cpp):
        cpp = bench_cpp(paths, args.cpp, args.repeats)
        result["cpp_pipeline"] = summarize(cpp) if cpp else None
    else:
        result["cpp_pipeline"] = None
    result["service_predict_path"] = bench_service(args.model)

    py, cpp_s = result["python_opencv_pipeline"], result["cpp_pipeline"]
    if isinstance(py, dict) and isinstance(cpp_s, dict):
        result["cpp_vs_python_median_ratio"] = round(py["median_ms"] / cpp_s["median_ms"], 2)

    print(f"Hardware: {result['hardware']}")
    for key in ("python_opencv_pipeline", "cpp_pipeline", "service_predict_path"):
        print(f"{key}: {result[key]}")
    if "cpp_vs_python_median_ratio" in result:
        print(f"Python/C++ median ratio on this machine: {result['cpp_vs_python_median_ratio']}x")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
