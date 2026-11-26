"""Command line interface for the NEXIR bootstrap compiler."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from . import __version__
from .compiler import NexirError, compile_file


def _compiler() -> str:
    explicit = os.environ.get("NEXIR_CXX")
    if explicit:
        return explicit
    for name in ("clang++", "g++", "c++"):
        found = shutil.which(name)
        if found:
            return found
    msys_gcc = Path(r"C:\msys64\ucrt64\bin\g++.exe")
    if msys_gcc.is_file():
        return str(msys_gcc)
    raise NexirError("C++17 compiler missing; install Clang++ or G++, or set NEXIR_CXX")


def _output_path(source: Path, requested: Path | None, *, shared: bool = False) -> Path:
    suffix = (".dll" if os.name == "nt" else ".so") if shared else (".exe" if os.name == "nt" else "")
    output = requested or source.with_suffix(suffix)
    return output.resolve()


def _build(source: Path, output: Path, *, emit_c: Path | None = None, shared: bool = False,
           cpp_sources: list[Path] | None = None, include_dirs: list[Path] | None = None,
           library_dirs: list[Path] | None = None, libraries: list[str] | None = None,
           objects: list[Path] | None = None, static_runtime: bool = True,
           icon: Path | None = None) -> None:
    code = compile_file(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if emit_c:
        emit_c.parent.mkdir(parents=True, exist_ok=True)
        emit_c.write_text(code, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nexir-", dir=output.parent) as directory:
        c_file = Path(directory) / "program.cpp"
        staged_output = Path(directory) / output.name
        c_file.write_text(code, encoding="utf-8")
        compiler = _compiler()
        env = os.environ.copy()
        if os.name == "nt":
            env["PATH"] = str(Path(compiler).resolve().parent) + os.pathsep + env.get("PATH", "")
        command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wno-unused-function"]
        if os.name == "nt" and static_runtime:
            command.append("-static")
        if shared:
            command.extend(["-shared", "-fPIC"])
        for include_dir in include_dirs or []:
            command.extend(["-I", str(include_dir.resolve())])
        command.append(str(c_file))
        command.extend(str(path.resolve()) for path in cpp_sources or [])
        command.extend(str(path.resolve()) for path in objects or [])
        if os.name == "nt" and not shared:
            icon_file = icon or Path(__file__).parent / "assets" / "nx-icon.ico"
            windres = shutil.which("windres") or str(Path(compiler).resolve().parent / "windres.exe")
            if icon_file.is_file() and Path(windres).is_file():
                rc_file = Path(directory) / "icon.rc"
                icon_object = Path(directory) / "icon.o"
                rc_file.write_text(f'1 ICON "{icon_file.resolve().as_posix()}"\n', encoding="utf-8")
                icon_result = subprocess.run([windres, "-i", str(rc_file), "-o", str(icon_object)],
                                             capture_output=True, text=True, errors="replace", env=env)
                if icon_result.returncode:
                    raise NexirError("icon resource failed: " + icon_result.stderr.strip())
                command.append(str(icon_object))
        for library_dir in library_dirs or []:
            command.extend(["-L", str(library_dir.resolve())])
        command.extend("-l" + name for name in libraries or [])
        if os.name == "nt":
            command.append("-lws2_32")
        command.extend(["-o", str(staged_output)])
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", env=env)
        if result.returncode:
            raise NexirError(f"C++ backend failed (exit {result.returncode}):\n" + (result.stderr or result.stdout).strip())
        os.replace(staged_output, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexir", description="NEXIR native compiler")
    parser.add_argument("--version", action="version", version=f"NEXIR {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="compile .nx to native executable")
    build.add_argument("source", type=Path)
    build.add_argument("-o", "--output", type=Path)
    build.add_argument("--emit-cpp", "--emit-c", dest="emit_cpp", type=Path, help="also write generated C++ for inspection")
    build.add_argument("--shared", action="store_true", help="build a shared library for @export functions")
    build.add_argument("--cpp", action="append", type=Path, default=[], help="link an additional C++ source file")
    build.add_argument("--include-dir", action="append", type=Path, default=[])
    build.add_argument("--library-dir", action="append", type=Path, default=[])
    build.add_argument("--library", action="append", default=[])
    build.add_argument("--object", action="append", type=Path, default=[], help="link a precompiled object, including CUDA objects")
    build.add_argument("--dynamic-runtime", action="store_true", help="do not statically link the C++ runtime on Windows")
    build.add_argument("--icon", type=Path, help="Windows executable icon (.ico)")
    check = sub.add_parser("check", help="parse and type-check .nx source")
    check.add_argument("source", type=Path)
    emit = sub.add_parser("emit", help="generate C++17 source without linking")
    emit.add_argument("source", type=Path)
    emit.add_argument("-o", "--output", type=Path, required=True)
    run = sub.add_parser("run", help="compile and run .nx source")
    run.add_argument("source", type=Path)
    run.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to executable")
    init = sub.add_parser("init", help="create a starter .nx project")
    init.add_argument("directory", type=Path)
    sub.add_parser("doctor", help="show compiler availability")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            print(f"NEXIR {__version__}")
            print(f"Python: {sys.version.split()[0]}")
            print(f"C++17 compiler: {_compiler()}")
            return 0
        if args.command == "init":
            directory = args.directory.resolve()
            directory.mkdir(parents=True, exist_ok=True)
            starter = directory / "main.nx"
            if starter.exists():
                raise NexirError(f"refusing to overwrite {starter}")
            starter.write_text('def main():\n    print("Hello from NEXIR")\n', encoding="utf-8")
            print(starter)
            return 0
        source = args.source.resolve()
        if args.command == "check":
            compile_file(source)
            print(f"OK: {source}")
            return 0
        if args.command == "emit":
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(compile_file(source), encoding="utf-8")
            print(output)
            return 0
        if args.command == "build":
            output = _output_path(source, args.output, shared=args.shared)
            _build(source, output, emit_c=args.emit_cpp, shared=args.shared,
                   cpp_sources=args.cpp, include_dirs=args.include_dir,
                   library_dirs=args.library_dir, libraries=args.library, objects=args.object,
                   static_runtime=not args.dynamic_runtime, icon=args.icon)
            print(output)
            return 0
        with tempfile.TemporaryDirectory(prefix="nexir-run-") as directory:
            output = Path(directory) / ("program.exe" if os.name == "nt" else "program")
            _build(source, output)
            return subprocess.run([str(output), *args.args]).returncode
    except (NexirError, OSError) as exc:
        print(f"nexir: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
