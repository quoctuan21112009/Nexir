"""Bundle the built wheel, install helper, docs and examples for distribution."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
wheel = ROOT / "release" / f"nexir-{VERSION}-py3-none-any.whl"
if not wheel.is_file():
    raise SystemExit(f"Build the wheel first: {wheel}")
output = ROOT / "release" / f"nexir-starter-{VERSION}.zip"
files = [
    wheel,
    ROOT / "README.md",
    ROOT / "scripts" / "install.ps1",
    ROOT / "nexir" / "include" / "nx_value_wire.h",
    ROOT / "nexir" / "assets" / "nx-icon.png",
    ROOT / "nexir" / "assets" / "nx-icon.ico",
    ROOT / "nexir" / "assets" / "nx-logo.svg",
]
files += [ROOT / "examples" / name for name in (
    "hello.nx", "fibonacci.nx", "arrays.nx", "dynamic.nx",
    "ring.nx", "ring_stale.nx", "native_math.nx", "native_math.cpp",
)]
files += [ROOT / "benchmarks" / name for name in (
    "bench.py", "work.nx", "work.py", "work.cpp", "work.rs", "results.md",
)]
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for file in files:
        archive.write(file, file.relative_to(ROOT))
print(output)
