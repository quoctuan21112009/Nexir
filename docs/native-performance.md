# Native performance (experimental)

NX already emits C++17. Use explicit typed `@extern("symbol")` for a C ABI
function and link its object/archive via `--object` (or `--library` and
`--library-dir`). Use fixed-width integers/floats and pointer+length adapters;
never infer a Rust/C++ ABI from a symbol name. Rust `extern "C"` with
`#[unsafe(no_mangle)]` (Rust 2024) / `#[no_mangle]` (older editions), a
`staticlib` or `cdylib`, and compatible target/toolchain are required.
C++ classes, exceptions, Rust panics and owned containers must not cross this
boundary. The existing dynamic import syntax still infers signatures and is
unsafe for mismatched signatures; prefer explicit declarations. Dynamic native
symbol lookup now caches its pointer per call site, with the existing policy
check on every call.

```sh
nx pkg rust build --manifest native/Cargo.toml
nx build main.nx --object native/target/release/libnative.a --profile fast
nx build main.nx --profile size
nx run --profile fast main.nx
nx run --no-cache main.nx
```

Rust build uses Cargo's existing dependency/build cache, --release and --locked;
a Cargo.lock must exist. --offline is available. Link flags and platform
library paths remain explicit.

Release builds cache generated translation units and the Python bridge
separately under ~/.nexir/cache/objects (NEXIR_CACHE_DIR overrides root).
Preprocessing still runs to fingerprint transitive headers; the linker runs
each time. External C++ sources are not cached by this implementation.
Debug and sanitizer builds bypass the cache. Remove the objects directory
when no builds are running to reclaim space. The cache is trusted local data,
not shared with untrusted users. No automatic size eviction is implemented.
The Python import cache now fingerprints generated NX code including NX imports
and the compiler executable metadata; changes in external native headers may
still require manually clearing ~/.nexir/cache/python.

Profiles: balanced (-O2), fast (-O3), size (-Os; executable section GC).
Shared-library exports are retained; stripping unused exported symbols would
break downstream callers. No fast-math or CPU-specific instructions are forced.
Python extension modules using PyObject* still require matching CPython,
even if their implementation is written in C/C++/Rust. Python bridge uses
vectorcall for small positional calls, and still converts Python objects.
An independently exported native C function can be linked without CPython.

--deny-ffi exposes the existing dynamic loader policy in CLI run. It does not
isolate linked native code, Python extensions or arbitrary native libraries.
Use an OS-isolated worker for untrusted code. Existing sandbox is not a
security boundary for arbitrary foreign code. No claim of complete sandboxing,
whole-program tree shaking, automatic wheel conversion, or a separate ASM
backend is made here.
