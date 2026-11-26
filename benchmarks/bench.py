"""Compile and time the same fixed arithmetic loop in four languages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nexir.cli import _build  # noqa: E402


def find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    msys = Path(r"C:\msys64\ucrt64\bin") / (name + ".exe")
    if msys.is_file():
        return str(msys)
    raise RuntimeError(f"{name} missing; install it and rerun the benchmark")


def run_checked(command: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, errors="replace", timeout=120)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    build = ROOT / "build" / "benchmark"
    build.mkdir(parents=True, exist_ok=True)
    ext = ".exe" if os.name == "nt" else ""
    cpp = find_tool("g++")
    rust = find_tool("rustc")
    env = os.environ.copy()
    if os.name == "nt":
        env["PATH"] = os.pathsep.join((str(Path(cpp).parent), str(Path(rust).parent), env.get("PATH", "")))

    nx_bin = build / ("work_nx" + ext)
    cpp_bin = build / ("work_cpp" + ext)
    rust_bin = build / ("work_rust" + ext)
    _build(ROOT / "benchmarks" / "work.nx", nx_bin)
    run_checked([cpp, "-std=c++17", "-O2", str(ROOT / "benchmarks" / "work.cpp"), "-o", str(cpp_bin)], env)
    run_checked([rust, "-C", "opt-level=2", "-C", "overflow-checks=off", str(ROOT / "benchmarks" / "work.rs"), "-o", str(rust_bin)], env)

    commands = {
        "NEXIR": [str(nx_bin)],
        "Python": [sys.executable, str(ROOT / "benchmarks" / "work.py")],
        "Rust": [str(rust_bin)],
        "C++": [str(cpp_bin)],
    }
    expected = None
    for name, command in commands.items():
        value = run_checked(command, env)  # warmup, excluded from measurements
        if expected is None:
            expected = value
        elif value != expected:
            raise RuntimeError(f"wrong result from {name}: {value} != {expected}")

    timings = {name: [] for name in commands}
    names = list(commands)
    for round_number in range(args.repeats):
        for name in names[round_number % len(names):] + names[:round_number % len(names)]:
            start = time.perf_counter_ns()
            value = run_checked(commands[name], env)
            elapsed = (time.perf_counter_ns() - start) / 1e9
            if value != expected:
                raise RuntimeError(f"wrong result from {name} during measurement")
            timings[name].append(elapsed)

    versions = {
        "Python": run_checked([sys.executable, "--version"], env),
        "Rust": run_checked([rust, "--version"], env),
        "C++": run_checked([cpp, "--version"], env).splitlines()[0],
    }
    medians = {name: statistics.median(values) for name, values in timings.items()}
    rows = [
        "# Benchmark NEXIR, Python, Rust, C++",
        "",
        f"- Thời điểm UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Hệ thống: {platform.platform()}; CPU: {os.environ.get('PROCESSOR_IDENTIFIER', platform.processor())}",
        f"- Phiên bản: {versions['Python']}; {versions['Rust']}; {versions['C++']}",
        f"- Tác vụ: 20.000.000 lần cập nhật `x = (x * 1664525 + 1013904223) % 2147483647` bằng số nguyên 64 bit.",
        f"- Kết quả của cả bốn: `{expected}`. Mỗi chương trình chạy nóng 1 lần rồi đo {args.repeats} lần bằng `perf_counter_ns` ngoài tiến trình.",
        "- Thời gian dưới đây chỉ tính chạy chương trình, gồm thời gian khởi động tiến trình; không tính biên dịch. NEXIR/C++ dùng `-O2`; Rust dùng `-C opt-level=2 -C overflow-checks=off`.",
        "",
        "| Ngôn ngữ | Trung vị (giây) | Các lần đo (giây) | Tốc độ so với Python |",
        "|---|---:|---|---:|",
    ]
    for name in commands:
        values = ", ".join(f"{v:.4f}" for v in timings[name])
        rows.append(f"| {name} | {medians[name]:.4f} | {values} | {medians['Python'] / medians[name]:.2f}× |")
    rows += [
        "",
        "Đây là phép đo một vòng lặp số học cụ thể trên một máy, không đại diện hiệu năng chung. NEXIR hiện sinh C++17 rồi dùng G++, nên kết quả chịu ảnh hưởng trực tiếp từ backend C++.",
        "",
    ]
    report = ROOT / "benchmarks" / "results.md"
    report.write_text("\n".join(rows), encoding="utf-8")
    sys.stdout.buffer.write(report.read_bytes())


if __name__ == "__main__":
    main()
