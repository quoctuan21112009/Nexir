# NX 2.0

![Biểu tượng NX](nexir/assets/nx-logo.svg)

NX 2.0 là ngôn ngữ lập trình biên dịch native qua C++17 với cú pháp tiệm cận Python (`class`, `def`, `enum`, `match`, `async`/`await`), hệ thống kiểu hybrid tĩnh/động, mô hình bộ nhớ cache không rò rỉ (`NxStr` SSO + `NxMemCache`), **NX Native ABI v2** kết nối hai chiều với Python/C/C++/Rust/Go, runtime bất đồng bộ đa nhân/GPU và cơ chế sandbox bảo mật kèm audit log.

Xem:
- [Tính năng NX 2.0](docs/features-2.0.md)
- [Hướng dẫn cú pháp NX](docs/syntax.md)
- [Hệ sinh thái package `pkg.nx.dev`](docs/package-ecosystem.md)
- [Python C API & NX Native ABI](docs/python-c-api.md)

## Bắt đầu

Cần Python 3.10+ và compiler C++17 (`clang++` hoặc `g++`).

```powershell
py -3 -m nexir doctor
py -3 -m nexir check examples/hello.nx
py -3 -m nexir run examples/enum_match.nx
py -3 -m nexir debug examples/hybrid_async_gpu.nx
py -3 -m nexir build examples/integers_128.nx -o build/integers_128.exe
```

## Điểm nổi bật NX 2.0

- **Cú pháp Pythonic (`class` thay `struct`, xóa `fn`):** Mọi hàm dùng `def` / `async def`, mọi đối tượng dùng `class` (kế thừa, default fields, `__init__`, `__del__`, `__str__`, `with`), `enum` bậc cao với payload & `match`/`case` kiểm tra vét cạn lúc biên dịch.
- **Số học đa độ rộng & chuyển đổi tự do:** `i8`, `i16`, `i32`, `i64`, `i128`, `u8`, `u16`, `u32`, `u64`, `u128`, `f32`, `f64`, alias `int`/`float` với kiểm tra tràn thu hẹp an toàn.
- **Memory Model không rò rỉ & tối ưu theo Cache:** Loại bỏ `nx_owners`; `str` dùng `NxStr` (22-byte SSO + `NxStrBlock` refcount qua slab cache `NxMemCache` + bounded intern cache); `StringBuilder` và tối ưu chuỗi `+` / f-string (`:.2f`, `:04d`, `:x`) tránh allocation chain.
- **Hybrid Typing & Import trực tiếp đa ngôn ngữ:** `import mod`, `from mod import sym`, `import py.math`, `import cpp.cmath` không cần viết file wrapper riêng; Python gọi ngược `.nx` qua `nexir.interop.import_nx()` và `nx bind --lang c|cpp|python|rust|go`.
- **Hybrid Async (`nx.asyncio`), Multi-Core (`nx.thread`) & GPU (`nx.gpu`):** Work-stealing executor đa lõi + I/O poller (`WSAPoll` Windows / `kqueue` macOS / `epoll` Linux), `Channel[T]`, `Mutex[T]`, `AtomicI64`, và GPU dispatch (`cuda`/`opencl`/`cpu-simd` fallback).
- **Deep I/O & Networking (`nx.fs`, `nx.io`, `nx.net`):** Đọc, ghi, sửa (`edit_file`), sao chép, xóa, phân quyền, `with open(...) as f:`, `TcpStream`, `TcpListener`, `dns_resolve`, `http_get`/`http_post`.
- **An toàn & Sandbox & Audit Log:** Kiểm tra move/borrow/null/exhaustiveness/FFI policy lúc biên dịch; thực thi sandbox `--sandbox --max-memory --timeout --allow-read --allow-host --deny-net --audit-log audit.jsonl`.
- **Hệ sinh thái `pkg.go.dev`-style:** `nx pkg nx init|add|resolve|install|publish|search|verify|audit|graph|doc` với `nx.lock` SHA-256 và `nx doc` sinh tài liệu web HTML.

## Kiểm thử

```powershell
py -3 -m unittest discover -s tests -v
```
