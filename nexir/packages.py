"""Explicit package-manager adapters for external Python and C++ libraries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .compiler import NexirError


CPP_PACKAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*\Z")
PY_PACKAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*(?:\[[A-Za-z0-9_,.\-]+\])?(?:[<>=!~]=?[A-Za-z0-9_.+\-]+)?\Z")
TRIPLET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*\Z")


def python_command() -> list[str]:
    explicit = os.environ.get("NEXIR_PYTHON")
    if explicit:
        if not Path(explicit).is_file():
            raise NexirError(f"NEXIR_PYTHON does not point to a Python executable: {explicit}")
        return [explicit]
    if not getattr(sys, "frozen", False):
        return [sys.executable]
    launcher = shutil.which("py")
    if launcher:
        return [launcher, "-3"]
    python = shutil.which("python")
    if python:
        return [python]
    raise NexirError("Python 3 is required for Python packages/--python; set NEXIR_PYTHON")


def python_info() -> dict[str, str]:
    """Query the actual external interpreter, never PyInstaller's private runtime."""
    code = (
        "import json,os,sys,sysconfig; "
        "print(json.dumps({'version':sys.version.split()[0],"
        "'include':sysconfig.get_path('include'),"
        "'library_dir':os.path.join(sys.base_prefix,'libs') if os.name=='nt' else sysconfig.get_config_var('LIBDIR'),"
        "'library':('python'+str(sys.version_info.major)+str(sys.version_info.minor)) if os.name=='nt' else sysconfig.get_config_var('LDLIBRARY'),"
        "'dll_dir':sys.base_prefix if os.name=='nt' else sysconfig.get_config_var('LIBDIR')}))"
    )
    proc = subprocess.run([*python_command(), "-c", code], capture_output=True, text=True, errors="replace")
    if proc.returncode:
        raise NexirError("cannot inspect external Python: " + proc.stderr.strip())
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise NexirError("external Python returned invalid sysconfig data") from exc


def vcpkg_root() -> Path:
    return Path(os.environ.get("VCPKG_ROOT") or Path.home() / ".nexir" / "vcpkg").expanduser().resolve()


def default_triplet() -> str:
    if os.name == "nt":
        return "x64-mingw-static"
    return "x64-linux" if sys.platform.startswith("linux") else "x64-osx"


def vcpkg_executable() -> Path:
    root = vcpkg_root()
    exe = root / ("vcpkg.exe" if os.name == "nt" else "vcpkg")
    if not exe.is_file():
        raise NexirError(f"vcpkg is not installed at {root}; run: nx pkg cpp bootstrap")
    return exe


def cpp_prefix(triplet: str) -> Path:
    if not TRIPLET.fullmatch(triplet):
        raise NexirError(f"invalid vcpkg triplet: {triplet}")
    prefix = vcpkg_root() / "installed" / triplet
    if not (prefix / "include").is_dir() or not (prefix / "lib").is_dir():
        raise NexirError(f"no vcpkg libraries installed for {triplet}: {prefix}")
    return prefix


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    try:
        result = subprocess.run(command, env=env)
    except OSError as exc:
        raise NexirError(str(exc)) from exc
    if result.returncode:
        raise NexirError(f"package command failed (exit {result.returncode})")


def package_command(args) -> None:
    if args.kind == "python":
        if not PY_PACKAGE.fullmatch(args.package):
            raise NexirError(f"invalid Python package specifier: {args.package}")
        command = [*python_command(), "-m", "pip", args.action, args.package]
        if args.action == "install" and args.upgrade:
            command.insert(-1, "--upgrade")
        _run(command)
        return

    if args.kind == "rust":
        cargo = shutil.which("cargo")
        if not cargo:
            raise NexirError("Cargo is required to build Rust libraries")
        manifest = args.manifest.resolve()
        if manifest.name != "Cargo.toml" or not manifest.is_file():
            raise NexirError(f"Cargo manifest not found: {manifest}")
        command = [cargo, "build", "--release", "--locked", "--manifest-path", str(manifest)]
        if args.target:
            if not TRIPLET.fullmatch(args.target):
                raise NexirError("invalid Rust target triple")
            command.extend(["--target", args.target])
        if args.offline:
            command.append("--offline")
        _run(command)
        return

    root = vcpkg_root()
    if args.action == "bootstrap":
        if root.exists():
            raise NexirError(f"vcpkg directory already exists: {root}; set VCPKG_ROOT to another directory")
        git = shutil.which("git")
        if not git:
            raise NexirError("Git is required to download vcpkg")
        root.parent.mkdir(parents=True, exist_ok=True)
        _run([git, "clone", "https://github.com/microsoft/vcpkg.git", str(root)])
        script = root / ("bootstrap-vcpkg.bat" if os.name == "nt" else "bootstrap-vcpkg.sh")
        try:
            _run([str(script)])
        except NexirError:
            raise NexirError(f"vcpkg bootstrap failed; inspect or remove the partial directory: {root}")
        print(root)
        return
    triplet = args.triplet or default_triplet()
    if not TRIPLET.fullmatch(triplet):
        raise NexirError(f"invalid vcpkg triplet: {triplet}")
    if args.action == "path":
        print(cpp_prefix(triplet))
        return
    package = args.package
    if not CPP_PACKAGE.fullmatch(package):
        raise NexirError(f"invalid C++ package name: {package}")
    env = os.environ.copy()
    env["VCPKG_DEFAULT_TRIPLET"] = triplet
    env["VCPKG_DEFAULT_HOST_TRIPLET"] = triplet
    compiler = Path(r"C:\msys64\ucrt64\bin")
    if os.name == "nt" and compiler.is_dir():
        env["PATH"] = str(compiler) + os.pathsep + env.get("PATH", "")
    _run([str(vcpkg_executable()), args.action, f"{package}:{triplet}"], env=env)
