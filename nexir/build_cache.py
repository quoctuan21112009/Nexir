"""Content-addressed release object cache; never cache link results."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .compiler import NexirError


def _run(command, *, cwd, env):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True)
    if result.returncode:
        raise NexirError("C++ backend failed (exit %s):\n%s" % (
            result.returncode, (result.stderr or result.stdout).decode(errors="replace").strip()))
    return result.stdout


def cached_object(compiler: str, flags: list[str], source: Path, output: Path,
                  env: dict[str, str]) -> Path:
    """Preprocess to fingerprint transitive headers, then compile that exact input.

    Only generated sources are passed here. Keep their relative filenames stable
    across staging directories. Objects are published atomically, and the linker
    always runs against current external libraries. Debug/sanitizer builds bypass
    this cache in the caller. Cache is local trusted-user data, not a sandbox.
    """
    source, output = source.resolve(), output.resolve()
    cwd = source.parent
    cc = Path(compiler).resolve()
    stat = cc.stat()
    version = _run([compiler, "--version"], cwd=cwd, env=env)
    target = _run([compiler, "-dumpmachine"], cwd=cwd, env=env)
    # These variables can affect toolchain/target selection independently of headers.
    selected_env = {k: env.get(k) for k in (
        "PATH", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "COMPILER_PATH",
        "GCC_EXEC_PREFIX", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "SOURCE_DATE_EPOCH",
        "INCLUDE", "LIB", "LIBRARY_PATH", "CLANG_CONFIG_FILE_SYSTEM_DIR",
        "CLANG_CONFIG_FILE_USER_DIR", "GCC_COMPARE_DEBUG")}
    meta = json.dumps(["nx-object-v1", str(cc), stat.st_size, stat.st_mtime_ns,
                       flags, selected_env], sort_keys=True).encode()
    preprocessed = _run([compiler, *flags, "-E", source.name], cwd=cwd, env=env)
    h = hashlib.sha256()
    for part in (meta, version, target, preprocessed):
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    root = Path(env.get("NEXIR_CACHE_DIR") or Path.home() / ".nexir" / "cache")
    folder = root / "objects"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    cached = folder / (h.hexdigest() + ".o")
    try:
        shutil.copyfile(cached, output)
        return output
    except FileNotFoundError:
        pass
    prefile = output.with_suffix(".ii")
    prefile.write_bytes(preprocessed)
    # GCC and Clang both support c++-cpp-output. The exact preprocessed bytes
    # avoid races between hashing and compilation of changing headers.
    _run([compiler, *flags, "-x", "c++-cpp-output", "-c", prefile.name,
          "-o", output.name], cwd=cwd, env=env)
    fd, name = tempfile.mkstemp(prefix="object-", dir=folder)
    os.close(fd)
    try:
        shutil.copyfile(output, name)
        os.replace(name, cached)
    finally:
        Path(name).unlink(missing_ok=True)
    return output
