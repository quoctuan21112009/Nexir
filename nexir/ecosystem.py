"""pkg.go.dev-style Nexir package ecosystem with lockfile, integrity hashing,
local/registry resolution, multi-language FFI manifest, and security audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from .compiler import Compiler, NexirError
from .docs_gen import generate_docs, serve_docs
from .security import HIGH_RISK_CAPABILITIES, sha256_file

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?\Z")
_DEPENDENCY = re.compile(r"(?:(?:\^|~|>=|==)?(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?|path:.+)\Z")
_MANIFEST = "nx.toml"
_LOCKFILE = "nx.lock"


def _registry_root() -> Path:
    return Path(os.environ.get("NX_REGISTRY") or (Path.home() / ".nexir" / "registry")).expanduser().resolve()


def _install_cache_root(project_root: Path) -> Path:
    return project_root / ".nx" / "packages"


def _project_dir(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise NexirError(f"Nexir project directory does not exist: {root}")
    return root


def _parse_semver(ver: str) -> tuple[int, int, int]:
    core = ver.split("-")[0]
    parts = [int(x) for x in core.split(".")]
    return parts[0], parts[1], parts[2]


def _matches_constraint(version: str, constraint: str) -> bool:
    if constraint.startswith("path:"):
        return True
    v = _parse_semver(version)
    if constraint.startswith("^"):
        req = _parse_semver(constraint[1:])
        return v[0] == req[0] and v >= req
    if constraint.startswith("~"):
        req = _parse_semver(constraint[1:])
        return v[0] == req[0] and v[1] == req[1] and v >= req
    if constraint.startswith(">="):
        return v >= _parse_semver(constraint[2:])
    if constraint.startswith("=="):
        return v == _parse_semver(constraint[2:])
    return v == _parse_semver(constraint)


def _dir_sha256(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        rel_path = p.relative_to(directory)
        if p.is_file() and not any(part in {".git", ".nx", "build"} for part in rel_path.parts):
            rel = rel_path.as_posix().encode("utf-8")
            h.update(rel)
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def _manifest(root: Path) -> dict:
    path = root / _MANIFEST
    if not path.is_file():
        raise NexirError(f"missing {path}; run `nx pkg nx init` to create a Nexir package manifest")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise NexirError(f"invalid {_MANIFEST}: {exc}") from exc
    package = data.get("package")
    if not isinstance(package, dict):
        raise NexirError("nx.toml requires a [package] table")
    name, version = package.get("name"), package.get("version")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise NexirError("[package].name must be lowercase letters, digits, '_' or '-' (max 64 chars)")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise NexirError("[package].version must be a semantic version such as 0.1.0")
    entry = package.get("entry", "src/main.nx")
    if not isinstance(entry, str) or not entry.endswith(".nx"):
        raise NexirError("[package].entry must be a relative .nx file path")
    entry_path = (root / entry).resolve()
    if not entry_path.is_relative_to(root):
        raise NexirError("[package].entry must stay inside the project directory")
    if not entry_path.is_file():
        raise NexirError(f"package entry file does not exist: {entry}")
    dependencies = data.get("dependencies", {})
    if not isinstance(dependencies, dict):
        raise NexirError("[dependencies] must be a table of package names and version constraints")
    for dep, constraint in dependencies.items():
        if not isinstance(dep, str) or not _NAME.fullmatch(dep) or dep == name:
            raise NexirError(f"invalid dependency package name: {dep!r}")
        if not isinstance(constraint, str) or not _DEPENDENCY.fullmatch(constraint):
            raise NexirError(f"invalid version constraint for dependency {dep!r}: {constraint!r}")
    return data


def _write_manifest(root: Path, data: dict) -> None:
    pkg = data["package"]
    lines = [
        "[package]",
        f'name = "{pkg["name"]}"',
        f'version = "{pkg["version"]}"',
        f'entry = "{pkg.get("entry", "src/main.nx")}"',
    ]
    for opt_k in ("module", "description", "license"):
        if opt_k in pkg:
            lines.append(f'{opt_k} = "{pkg[opt_k]}"')
    lines.append("")
    lines.append("[dependencies]")
    for dep, ver in sorted(data.get("dependencies", {}).items()):
        lines.append(f'{dep} = "{ver}"')
    for sec in ("ffi", "sandbox"):
        if sec in data and isinstance(data[sec], dict):
            lines.append("")
            lines.append(f"[{sec}]")
            for k, v in data[sec].items():
                if isinstance(v, list):
                    arr = ", ".join(json.dumps(x) for x in v)
                    lines.append(f"{k} = [{arr}]")
                elif isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                elif isinstance(v, int):
                    lines.append(f"{k} = {v}")
                else:
                    lines.append(f'{k} = "{v}"')
    (root / _MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scan_capabilities(pkg_dir: Path) -> list[str]:
    caps: set[str] = set()
    for nf in sorted(pkg_dir.rglob("*.nx")):
        if any(part in {".nx", "build", ".git"} for part in nf.parts):
            continue
        try:
            comp = Compiler(nf.read_text(encoding="utf-8"), entry_path=nf, require_main=False)
            for fn in comp.entry_unit.functions.values():
                caps.update(fn.caps)
        except Exception:
            pass
    return sorted(caps)


def _resolve_package(dep_name: str, constraint: str, project_root: Path) -> tuple[Path, str, str]:
    if constraint.startswith("path:"):
        raw = constraint[5:].strip()
        cand = (project_root / raw).resolve()
        if not cand.is_dir():
            raise NexirError(f"local dependency path not found: {cand}")
        meta = _manifest(cand)
        return cand, meta["package"]["version"], _dir_sha256(cand)
    reg = _registry_root()
    pkg_reg_dir = reg / dep_name
    if not pkg_reg_dir.is_dir():
        raise NexirError(f"package '{dep_name}' not found in registry ({reg})")
    candidates = []
    for ver_dir in pkg_reg_dir.iterdir():
        if ver_dir.is_dir() and _VERSION.fullmatch(ver_dir.name) and _matches_constraint(ver_dir.name, constraint):
            candidates.append((_parse_semver(ver_dir.name), ver_dir.name, ver_dir))
    if not candidates:
        raise NexirError(f"no version of '{dep_name}' satisfies '{constraint}'")
    candidates.sort(reverse=True)
    _, ver, chosen_dir = candidates[0]
    return chosen_dir, ver, _dir_sha256(chosen_dir)


def _resolve_and_lock(root: Path, *, install_files: bool = False) -> dict:
    data = _manifest(root)
    deps = data.get("dependencies", {})
    locked: dict[str, dict] = {}
    queue = list(deps.items())
    seen: set[str] = set()

    while queue:
        dep_name, constraint = queue.pop(0)
        if dep_name in seen:
            continue
        seen.add(dep_name)
        src_dir, version, digest = _resolve_package(dep_name, constraint, root)
        caps = _scan_capabilities(src_dir)
        dep_manifest = _manifest(src_dir)
        locked[dep_name] = {
            "version": version,
            "constraint": constraint,
            "sha256": digest,
            "capabilities": caps,
            "source": str(src_dir),
        }
        if install_files:
            dst = _install_cache_root(root) / dep_name
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_dir, dst)
        for sub_dep, sub_con in dep_manifest.get("dependencies", {}).items():
            if sub_dep not in seen:
                queue.append((sub_dep, sub_con))

    lock_data = {
        "lockfile_version": 1,
        "package": data["package"]["name"],
        "packages": locked,
    }
    (root / _LOCKFILE).write_text(json.dumps(lock_data, indent=2) + "\n", encoding="utf-8")
    return lock_data


def _verify_lock(root: Path) -> list[str]:
    lock_path = root / _LOCKFILE
    if not lock_path.is_file():
        raise NexirError(f"missing {_LOCKFILE}; run `nx pkg nx resolve` first")
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    errors = []
    for name, info in lock_data.get("packages", {}).items():
        installed_dir = _install_cache_root(root) / name
        if not installed_dir.is_dir():
            errors.append(f"{name}: not installed in .nx/packages")
            continue
        actual_hash = _dir_sha256(installed_dir)
        if actual_hash != info.get("sha256"):
            errors.append(f"{name}: sha256 mismatch (expected {info.get('sha256')}, got {actual_hash})")
    return errors


def ecosystem_command(args: argparse.Namespace) -> None:
    root = _project_dir(args.path)
    manifest_path = root / _MANIFEST
    if args.action == "init":
        if manifest_path.exists():
            raise NexirError(f"refusing to overwrite {manifest_path}")
        package_name = args.name or root.name.lower().replace(" ", "-")
        if not _NAME.fullmatch(package_name):
            raise NexirError("package name must use lowercase letters, digits, '_' or '-' (max 64 chars)")
        source = root / "src" / "main.nx"
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            source.write_text('def main():\n    print("Hello from Nexir")\n', encoding="utf-8")
        _write_manifest(root, {
            "package": {
                "name": package_name,
                "version": "0.1.0",
                "entry": "src/main.nx",
                "module": f"nx.dev/local/{package_name}",
            },
            "dependencies": {},
        })
        print(f"Created {manifest_path}")
        return

    data = _manifest(root)
    package = data["package"]
    if args.action == "validate":
        print(f"Valid Nexir package: {package['name']} {package['version']}")
        return
    if args.action == "list":
        print(json.dumps({"package": package, "dependencies": data.get("dependencies", {})}, indent=2))
        return
    if args.action == "add":
        dep = args.package
        if not _NAME.fullmatch(dep) or dep == package["name"]:
            raise NexirError(f"invalid dependency name: {dep}")
        ver = args.version or "^0.1.0"
        if not _DEPENDENCY.fullmatch(ver):
            raise NexirError(f"invalid version constraint: {ver}")
        data.setdefault("dependencies", {})[dep] = ver
        _write_manifest(root, data)
        print(f"Added {dep} = \"{ver}\" to {manifest_path}")
        return
    if args.action == "remove":
        dep = args.package
        deps = data.get("dependencies", {})
        if dep not in deps:
            raise NexirError(f"dependency not found: {dep}")
        del deps[dep]
        _write_manifest(root, data)
        print(f"Removed {dep} from {manifest_path}")
        return
    if args.action == "publish":
        reg = _registry_root()
        target = reg / package["name"] / package["version"]
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for item in root.iterdir():
            if item.name in {".git", ".nx", "build", ".build-env", _LOCKFILE}:
                continue
            dst = target / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
        digest = _dir_sha256(target)
        print(f"Published {package['name']}@{package['version']} -> {target} (sha256: {digest[:16]}...)")
        return
    if args.action == "search":
        reg = _registry_root()
        query = (args.query or "").lower()
        matches = []
        if reg.is_dir():
            for pkg_dir in sorted(reg.iterdir()):
                if pkg_dir.is_dir() and query in pkg_dir.name.lower():
                    vers = sorted(v.name for v in pkg_dir.iterdir() if v.is_dir())
                    matches.append({"name": pkg_dir.name, "versions": vers})
        print(json.dumps(matches, indent=2))
        return
    if args.action in {"resolve", "install"}:
        lock = _resolve_and_lock(root, install_files=(args.action == "install"))
        print(f"{'Installed' if args.action == 'install' else 'Resolved'} {len(lock['packages'])} package(s) -> {root / _LOCKFILE}")
        return
    if args.action == "verify":
        errs = _verify_lock(root)
        if errs:
            raise NexirError("integrity verification failed:\n  " + "\n  ".join(errs))
        print("OK: all installed packages match nx.lock SHA-256 checksums")
        return
    if args.action == "audit":
        lock_path = root / _LOCKFILE
        lock_data = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else _resolve_and_lock(root, install_files=False)
        report = {"package": package["name"], "own_capabilities": _scan_capabilities(root), "dependencies": {}}
        for name, info in lock_data.get("packages", {}).items():
            caps = info.get("capabilities", [])
            risky = [c for c in caps if c in HIGH_RISK_CAPABILITIES]
            report["dependencies"][name] = {
                "version": info.get("version"),
                "sha256": info.get("sha256"),
                "capabilities": caps,
                "high_risk": risky,
            }
        print(json.dumps(report, indent=2))
        return
    if args.action == "graph":
        lock_path = root / _LOCKFILE
        lock_data = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else _resolve_and_lock(root, install_files=False)
        print(f"{package['name']}@{package['version']}")
        for dep, info in lock_data.get("packages", {}).items():
            print(f"  -> {dep}@{info.get('version')} ({info.get('constraint')})")
        return
    if args.action == "doc":
        out_dir = (root / "build" / "docs")
        out = generate_docs(root, out_dir, format_kind=getattr(args, "format", "html"))
        print(out)
        if getattr(args, "serve", False):
            serve_docs(out_dir, port=getattr(args, "port", 8080))
        return
    raise NexirError(f"unknown Nexir ecosystem action: {args.action}")
