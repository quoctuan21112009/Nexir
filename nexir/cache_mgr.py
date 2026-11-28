"""Function-level incremental compiler, x86_64 ASM pipeline, and LRU cache manager."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from . import __version__


def default_cache_root() -> Path:
    env = os.environ.get("NEXIR_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".nexir" / "cache").resolve()


class CacheManager:
    """Manages ~/.nexir/cache with LRU eviction, deduplication, ASM units, and incremental objects."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_cache_root()).resolve()
        self.inc_dir = self.root / "inc"
        self.asm_dir = self.root / "asm"
        self.bin_dir = self.root / "bins"
        self.pydeps_dir = self.root / "pydeps"
        self.index_path = self.root / "index.json"
        for d in (self.inc_dir, self.asm_dir, self.bin_dir, self.pydeps_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, Any] = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        if self.index_path.is_file():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("entries", {})
                    data.setdefault("hits", 0)
                    data.setdefault("misses", 0)
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return {"entries": {}, "hits": 0, "misses": 0}

    def save(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2) + "\n", encoding="utf-8")

    def touch(self, key: str, path: Path, kind: str) -> None:
        size = path.stat().st_size if path.is_file() else 0
        self.index["entries"][key] = {
            "path": str(path),
            "kind": kind,
            "size": size,
            "atime": time.time(),
        }
        self.save()

    def record_hit(self) -> None:
        self.index["hits"] = int(self.index.get("hits", 0)) + 1

    def record_miss(self) -> None:
        self.index["misses"] = int(self.index.get("misses", 0)) + 1

    def status(self) -> dict[str, Any]:
        total_bytes = 0
        counts = {"inc": 0, "asm": 0, "bins": 0, "pydeps": 0}
        for key, sub in (("inc", self.inc_dir), ("asm", self.asm_dir), ("bins", self.bin_dir), ("pydeps", self.pydeps_dir)):
            if sub.is_dir():
                for f in sub.rglob("*"):
                    if f.is_file():
                        total_bytes += f.stat().st_size
                        counts[key] += 1
        return {
            "cache_dir": str(self.root),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 3),
            "counts": counts,
            "hits": self.index.get("hits", 0),
            "misses": self.index.get("misses", 0),
        }

    def prune(self, max_mb: float = 256.0) -> dict[str, Any]:
        limit = max(0, int(max_mb * 1024 * 1024))
        files: list[tuple[float, int, Path, str | None]] = []
        known_by_path = {v.get("path"): k for k, v in self.index.get("entries", {}).items() if isinstance(v, dict)}
        total = 0
        for sub in (self.inc_dir, self.asm_dir, self.bin_dir):
            if not sub.is_dir():
                continue
            for f in sub.rglob("*"):
                if f.is_file():
                    st = f.stat()
                    total += st.st_size
                    files.append((st.st_atime, st.st_size, f, known_by_path.get(str(f))))
        files.sort(key=lambda item: item[0])  # oldest first (LRU)
        removed = 0
        freed = 0
        for _, sz, path, key in files:
            if total <= limit:
                break
            try:
                path.unlink()
                total -= sz
                freed += sz
                removed += 1
                if key and key in self.index["entries"]:
                    del self.index["entries"][key]
            except OSError:
                pass
        self.save()
        return {"removed": removed, "freed_bytes": freed, "remaining_bytes": total}

    def tune(self, max_mb: float = 256.0) -> dict[str, Any]:
        """Deduplicate stale entries, remove orphaned index keys, and enforce LRU size budget."""
        stale_keys = [k for k, v in self.index.get("entries", {}).items() if not Path(v.get("path", "")).exists()]
        for k in stale_keys:
            del self.index["entries"][k]
        res = self.prune(max_mb=max_mb)
        res["stale_index_removed"] = len(stale_keys)
        self.save()
        return res

    def clean(self) -> dict[str, Any]:
        before = self.status()["total_bytes"]
        for sub in (self.inc_dir, self.asm_dir, self.bin_dir, self.pydeps_dir):
            if sub.is_dir():
                shutil.rmtree(sub, ignore_errors=True)
                sub.mkdir(parents=True, exist_ok=True)
        self.index = {"entries": {}, "hits": 0, "misses": 0}
        self.save()
        return {"freed_bytes": before}


def function_ast_hashes(source: str) -> dict[str, str]:
    """Compute per-function AST hashes for incremental function-level caching."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"__module__": hashlib.sha256(source.encode("utf-8")).hexdigest()}
    hashes: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            dump = ast.dump(node, include_attributes=False)
            hashes[node.name] = hashlib.sha256(f"{__version__}:{dump}".encode("utf-8")).hexdigest()[:16]
    return hashes


def compile_asm_and_link(
    cxx: str,
    cpp_path: Path,
    output_path: Path,
    *,
    include_dirs: list[Path],
    extra_flags: list[str],
    cache_mgr: CacheManager | None = None,
    emit_asm_path: Path | None = None,
    incremental: bool = True,
    source_text: str = "",
) -> Path:
    """Compile C++17 -> Assembly (.s) -> Object (.o) -> static standalone binary with caching."""
    mgr = cache_mgr or CacheManager()
    fn_hashes = function_ast_hashes(source_text) if source_text else {}
    cpp_bytes = cpp_path.read_bytes()
    stable_flags = [f for f in extra_flags if not f.endswith("icon.o")]
    content_key = hashlib.sha256(cpp_bytes + f"|{cxx}|{' '.join(stable_flags)}|{__version__}".encode("utf-8")).hexdigest()[:24]

    cached_s = mgr.asm_dir / f"{content_key}.s"
    cached_o = mgr.inc_dir / f"{content_key}.o"
    inc_meta = mgr.inc_dir / f"{content_key}.meta.json"

    inc_flags = [item for d in include_dirs for item in ("-I", str(d.resolve()))]
    base_compile = [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wno-unused-function", "-Wno-unknown-pragmas", "-fopenmp-simd", *inc_flags]
    env = os.environ.copy()
    if os.name == "nt":
        env["PATH"] = str(Path(cxx).resolve().parent) + os.pathsep + env.get("PATH", "")

    if incremental and cached_s.is_file() and cached_o.is_file():
        mgr.record_hit()
        mgr.touch(content_key, cached_o, "obj")
        mgr.touch(f"{content_key}:asm", cached_s, "asm")
    else:
        mgr.record_miss()
        subprocess.run([*base_compile, "-S", str(cpp_path), "-o", str(cached_s)], check=True, env=env)
        subprocess.run([*base_compile, "-c", str(cached_s), "-o", str(cached_o)], check=True, env=env)
        mgr.touch(f"{content_key}:asm", cached_s, "asm")
        mgr.touch(content_key, cached_o, "obj")
        inc_meta.write_text(json.dumps({"functions": fn_hashes, "asm": str(cached_s), "obj": str(cached_o)}, indent=2), encoding="utf-8")

    if emit_asm_path is not None:
        emit_asm_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_s, emit_asm_path)

    # Link cached object statically to minimize host DLL/library conflicts
    link_cmd = [cxx, str(cached_o), *extra_flags, "-o", str(output_path)]
    subprocess.run(link_cmd, check=True, env=env)
    return output_path
