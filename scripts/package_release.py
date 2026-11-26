"""Create a Windows starter archive from the built standalone CLI and VSIX."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
VERSION = "1.2.0"
exe = RELEASE / "nx.exe"
if not exe.is_file():
    raise SystemExit(f"Build nx.exe first: {exe}")
output = RELEASE / f"nexir-windows-{VERSION}.zip"
files = [
    exe,
    ROOT / "README.md",
    ROOT / "scripts" / "install-windows.ps1",
    ROOT / "scripts" / "build-windows.ps1",
    ROOT / "docs" / "huong-dan-day-du.md",
    ROOT / "docs" / "features-1.1.md",
    ROOT / "docs" / "features-1.2.md",
    ROOT / "docs" / "syntax.md",
    ROOT / "docs" / "python-c-api.md",
    ROOT / "nexir" / "include" / "nx_value_wire.h",
]
files += list((ROOT / "examples").glob("*.nx"))
files += [ROOT / "examples" / "native_math.cpp", ROOT / "examples" / "nx_worker.py", ROOT / "examples" / "nx_py_worker.py"]
vsix = RELEASE / "nexir-language.vsix"
if vsix.is_file():
    files.append(vsix)
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for file in files:
        archive.write(file, file.relative_to(ROOT))
print(output)
