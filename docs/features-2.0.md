# Tính năng NX 2.0

NX 2.0 nâng cấp toàn diện cú pháp, hệ thống kiểu, quản lý bộ nhớ, FFI đa ngôn ngữ, async/multi-core/GPU runtime, sandbox bảo mật và hệ sinh thái package theo phong cách `pkg.go.dev`.

## 1. Cú pháp tiệm cận Python (`class`, `def`, `enum`, `match`)
- **Đổi `struct` → `class` và xóa `fn`:** Mọi khai báo hàm dùng `def` (hoặc `async def`), mọi kiểu dữ liệu hướng đối tượng dùng `class` (hỗ trợ kế thừa đơn, `super().__init__(...)`, field mặc định, `__init__`, `__del__`, `__str__`, operator overloading, `with ... as ...:`, trait conformance).
- **`enum` bậc cao (ADT / Tagged Union) & `match`/`case`:** Hỗ trợ enum có payload (`Circle(radius: f64)`), phương thức trên enum và kiểm tra vét cạn (`exhaustive match`) ngay tại thời điểm biên dịch.

## 2. Chuyển đổi số học tự do (`i8..i128`, `u8..u128`, `f32`, `f64`)
- Hỗ trợ đầy đủ `i8`, `i16`, `i32`, `i64`, `i128`, `u8`, `u16`, `u32`, `u64`, `u128`, `f32`, `f64`, cùng alias Python `int`/`float`.
- Tự động nâng kiểu trong biểu thức và cho phép chuyển đổi tự do giữa các độ rộng bit; chuyển đổi thu hẹp kiểm tra tràn miền giá trị an toàn (`nx_num_cast`).

## 3. Memory Model Cache-based, Owned String (`NxStr`) & `StringBuilder`
- Loại bỏ hoàn toàn cơ chế giữ con trỏ vào global `nx_owners` gây rò rỉ bộ nhớ/OOM.
- `str` là kiểu **`NxStr` sở hữu bộ nhớ theo RAII**: tối ưu **22-byte SSO (Small String Optimization)** trên stack (0 cấp phát heap), chuỗi dài dùng `NxStrBlock` đếm tham chiếu nguyên tử cấp phát qua **`NxMemCache`** (cache free-list theo size-class + bounded string intern cache).
- `StringBuilder` và tối ưu hóa nối chuỗi `+` / f-string nâng cao (`:.2f`, `:04d`, `:x`, `:>10`, `{obj}`) giúp tránh allocation chain.

## 4. Hybrid Typing & Import đa ngôn ngữ không rắc rối
- **Hybrid Typing:** Kết hợp tốc độ native tĩnh (`i64`, `List[T]`, `class`, `SIMD`) với `dyn` (`NxValue` 16-byte) và hàm tham số không cần chú giải kiểu (`def add(a, b):`).
- **Import đa file `.nx`:** `import mod`, `import pkg.sub`, `from mod import sym as alias`, namespace cô lập theo từng module.
- **Chạy thẳng thư viện Python & C++ không cần file wrapper:**
  - `import py.math`, `import py.statistics`, `import py.numpy as np` qua **NX Native ABI v2** (`nexir/include/nx_native_abi.h`).
  - `import cpp.cmath as cmath` gọi trực tiếp `#include <cmath>`.
  - `import c.<lib>`, `import rust.<lib>`, `import go.<lib>` nạp thư viện chia sẻ qua C ABI.
- **Chạy 2 chiều:** Python có thể import trực tiếp module `.nx` qua `nexir.interop.import_nx("mod.nx")` hoặc `enable_import_hook()`; `nx bind` sinh binding cho **C, C++, Python, Rust và Go**.

## 5. `Result`, `Optional` cho `class`, `match` và toán tử `?`
- `Optional[T]` và `Result[T]` / `Result[T, E]` hỗ trợ mọi `class`, `owned[Class]`, `enum`, `List`, `Dict`.
- Hỗ trợ `is not None` flow narrowing, toán tử truyền lỗi `?` (`val = load()?`), `try ... except Err as e: ... finally:`.

## 6. Hybrid Async (`nx.asyncio`), Multi-Core (`nx.thread`) & GPU (`nx.gpu`)
- `import nx.asyncio`: `async def`, `await`, `asyncio.run`, `asyncio.gather`, `asyncio.sleep`, `asyncio.read_text`, `asyncio.http_get` chạy trên work-stealing executor đa lõi + I/O poller đa nền tảng (`WSAPoll` Windows / `kqueue` macOS / `epoll` Linux).
- `import nx.thread`: `cpu_count()`, `ThreadPool`, `Channel[T]`, `Mutex[T]`, `AtomicI64`, `parallel_range`.
- `import nx.gpu`: Tự dò CUDA/OpenCL runtime và fallback mượt mà sang **multi-core SIMD (`cpu-simd`)** với `to_device`, `add`, `mul`, `reduce_sum`, `matmul`, `matmul_async`.

## 7. Deep File I/O, Networking, Debug Model, Sandbox & Audit Log
- `nx.fs` / `nx.io`: `read_text`, `read_bytes`, `write_text`, `write_bytes`, `append_text`, `copy_file`, `move_file`, `edit_file`, `delete_file`, `stat`, `chmod`, `list_dir`, `with open(...) as f:`.
- `nx.net`: `TcpStream`, `TcpListener`, `dns_resolve`, `http_get`, `http_post`.
- **Debug Model (`nx debug` / `--debug`):** Stack trace ánh xạ file/dòng/hàm `.nx` (`NxTraceGuard`) và báo cáo cache/leak (`nx.debug.mem_report()`).
- **Sandbox & Bảo mật:** Kiểm soát quyền tại thời điểm biên dịch (`@cap`) + FFI policy + sandbox thực thi (`--sandbox`, `--max-memory`, `--timeout`, `--allow-read`, `--allow-write`, `--allow-host`, `--deny-net`) + **Audit Log JSONL (`--audit-log`)** ghi nhận mọi module/FFI được nạp.
- **Package Ecosystem (`nx pkg nx` & `nx doc`):** Lockfile `nx.lock` kiểm tra SHA-256 (`verify`), kiểm toán quyền (`audit`), registry (`publish`, `search`, `add`, `install`), và sinh tài liệu HTML theo phong cách `pkg.go.dev`.
