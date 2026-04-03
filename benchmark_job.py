"""
benchmark_job.py - FramePack Studio automated benchmark

Launches studio.py in --benchmark mode with an incremented seed,
streams output to the console, parses [TIMING] phase lines, and
reports a per-phase timing summary.

Usage:
    python benchmark_job.py 260402_130944_829_8715.json
    python benchmark_job.py 260402_130944_829_8715.json --seed 99999

Every run increments the seed by 1 (or uses --seed).
Results are appended to benchmark_results.jsonl for comparison.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time


SEP = "=" * 60


def find_python():
    """Return the Python executable to use (same venv as caller, or .venv)."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        sys.executable,
        os.path.join(base, ".venv", "Scripts", "python.exe"),
        os.path.join(base, "venv_py312", "Scripts", "python.exe"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return sys.executable  # fallback


def run_benchmark(studio_dir, json_path, seed, results_file):
    python = find_python()
    studio  = os.path.join(studio_dir, "studio.py")

    cmd = [python, studio,
           "--benchmark", json_path,
           "--benchmark-seed", str(seed)]

    print(f"\n{SEP}")
    print(f"  Launching: {' '.join(cmd)}")
    print(f"{SEP}\n")

    timing_lines = {}   # label -> elapsed_s (float)
    total_s      = None
    t_wall_start = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=studio_dir,
        bufsize=1,
    )

    timing_re = re.compile(r"\[TIMING\]\s+(.+?):\s+\+([0-9.]+)s")
    total_re  = re.compile(r"\[BENCHMARK\]\s+TOTAL TIME:\s+([0-9.]+)s")

    for line in proc.stdout:
        print(line, end="", flush=True)

        m = timing_re.search(line)
        if m:
            timing_lines[m.group(1).strip()] = float(m.group(2))

        m = total_re.search(line)
        if m:
            total_s = float(m.group(1))

    proc.wait()
    wall_s = time.time() - t_wall_start

    return proc.returncode, timing_lines, total_s, wall_s


def print_summary(job, seed, returncode, timing_lines, total_s, wall_s, results_file):
    print(f"\n{SEP}")
    print(f"  BENCHMARK RESULTS  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{SEP}")
    print(f"  Exit code   : {returncode}")
    print(f"  Seed used   : {seed}")

    if timing_lines:
        print(f"\n  Phase timing (from worker [TIMING] lines):")
        prev = 0.0
        for label, elapsed in sorted(timing_lines.items(), key=lambda kv: kv[1]):
            delta = elapsed - prev
            print(f"    +{elapsed:6.1f}s  ({delta:+.1f}s)  {label}")
            prev = elapsed

    if total_s is not None:
        print(f"\n  Worker total  : {total_s:.1f}s   (excludes model startup)")
    print(f"  Wall clock    : {wall_s:.1f}s   (includes model loading from disk)")
    print(SEP)

    record = {
        "timestamp"   : time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference"   : os.path.basename(args_global.json_file),
        "seed"        : seed,
        "model"       : job.get("model_type"),
        "length_s"    : job.get("total_second_length"),
        "steps"       : job.get("steps"),
        "window"      : job.get("latent_window_size"),
        "loras"       : list(job.get("loras", {}).keys()),
        "worker_total_s": total_s,
        "wall_clock_s": round(wall_s, 1),
        "phases"      : timing_lines,
        "exit_code"   : returncode,
    }
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n  Results saved → {results_file}")
    print(f"  Run again to benchmark with seed {seed + 1}.")


# keep a module-level reference so print_summary can access it
args_global = None


def main():
    global args_global

    ap = argparse.ArgumentParser(
        description="Automated FramePack Studio benchmark — runs studio.py --benchmark"
    )
    ap.add_argument("json_file", help="Reference job JSON (e.g. 260402_130944_829_8715.json)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed override (default: original seed + 1)")
    ap.add_argument("--results-file", default="benchmark_results.jsonl",
                    help="Append results here (default: benchmark_results.jsonl)")
    args = ap.parse_args()
    args_global = args

    json_path = os.path.abspath(args.json_file)
    if not os.path.isfile(json_path):
        print(f"ERROR: {json_path} not found.")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        job = json.load(f)

    new_seed    = args.seed if args.seed is not None else job["seed"] + 1
    studio_dir  = os.path.dirname(json_path)
    results_file = os.path.join(studio_dir, args.results_file)

    print(f"\n{SEP}")
    print(f"  FramePack Studio  —  Benchmark")
    print(f"{SEP}")
    print(f"  Reference  : {os.path.basename(json_path)}")
    print(f"  Model      : {job.get('model_type','?')}  |  {job.get('total_second_length','?')}s  |  {job.get('steps','?')} steps")
    print(f"  Resolution : {job.get('resolutionW','?')}x{job.get('resolutionH','?')}")
    print(f"  Seed       : {job['seed']}  →  {new_seed}  (only change)")
    for name, w in job.get("loras", {}).items():
        print(f"  LoRA       : {name} @ {w}")
    print(SEP)

    rc, timing_lines, total_s, wall_s = run_benchmark(
        studio_dir, json_path, new_seed, results_file
    )

    print_summary(job, new_seed, rc, timing_lines, total_s, wall_s, results_file)


if __name__ == "__main__":
    main()

