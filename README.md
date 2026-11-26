# NX 1.0

![Biểu tượng NX](nexir/assets/nx-logo.svg)

NX là ngôn ngữ lập trình mục đích chung, biên dịch tệp `.nx` thành chương trình native qua C++17. Trình biên dịch kiểm tra cú pháp, kiểu và quyền `@cap` trước khi gọi Clang++ hoặc G++. Cú pháp hiện dùng Python AST để phân tích thụt lề; chương trình đã biên dịch không cần Python khi chạy.

## Bắt đầu

Cần Python 3.10+ và compiler C++17 (`clang++` hoặc `g++`). Trên Windows có thể dùng `py -3 -m nexir` ngay trong thư mục này:

```powershell
py -3 -m nexir doctor
py -3 -m nexir check examples/hello.nx
py -3 -m nexir run examples/hello.nx
py -3 -m nexir build examples/hello.nx -o build/hello.exe
.\build\hello.exe
```

Cài CLI bằng `py -3 -m pip install -e .` để dùng lệnh `nx` và `nexir`. `nx init myproject` tạo `main.nx`. `nx emit file.nx -o file.cpp` xuất mã C++ để xem. `NEXIR_CXX` chọn compiler C++ khác. Lệnh `build --icon path.ico` đặt icon riêng cho executable Windows; mặc định dùng biểu tượng NX.

## Ngôn ngữ hiện có

- Kiểu: `i32`, `i64`, `f32`, `f64`, `bool`, `str`, `dyn`, `void`.
- Mảng số: `Span[T]` chỉ đọc và `MutSpan[T]` đọc/ghi; phép truy cập `data[i]` kiểm tra biên.
- Hàm, suy luận kiểu biến cục bộ, số học, so sánh, `if`, `while`, `for ... in range(...)`, `break`, `continue`, `return`, `print`.
- `Ring[f64]` với kiểm tra chỉ số và thế hệ khi đọc dữ liệu cũ.
- `@export` tạo ký hiệu C ABI trong thư viện chia sẻ. `@extern("symbol")` khai báo hàm C ABI ngoài. CLI hỗ trợ `--cpp`, `--include-dir`, `--library-dir`, `--library` và `--object` để liên kết mã native.

Ví dụ mảng:

```nx
def sum_values(data: Span[f32]) -> f32:
    total: f32 = f32(0)
    for i in range(len(data)):
        total += data[i]
    return total

def main():
    values = [f32(1), f32(2), f32(3)]
    values[1] = f32(4)
    print(sum_values(values))
```

`zeros_f32(n)` và `zeros_f64(n)` cấp phát vùng nhớ và cần `@cap("memory.alloc")`. Mảng cấp phát được giữ sống đến khi chương trình thoát để con trỏ `Span` ổn định.

## Kiểm tra quyền

`@cap` hiện nhận `memory.alloc`, `filesystem.read` và `network.connect`. `read_text(path)` cần cả quyền đọc tệp và cấp phát; `tcp_connect(host, port)` cần quyền kết nối. Hàm gọi phải khai báo đủ quyền của hàm được gọi. Đây là kiểm tra lúc biên dịch, không phải sandbox hệ điều hành.

```nx
@cap("filesystem.read", "memory.alloc")
def main():
    print(read_text("config.txt"))
```

## Kiểm thử và đo tốc độ

```powershell
py -3 -m unittest discover -s tests -v
py -3 benchmarks/bench.py
```

[Phép đo mẫu](benchmarks/results.md) so sánh một vòng lặp số học với Python, Rust và C++. Đây chỉ là kết quả trên một máy và một tác vụ, không chứng minh ưu thế tốc độ tổng quát.

## Giới hạn

NX 1.0 là compiler đang phát triển, chưa có đầy đủ ownership/borrow checking, sandbox runtime, generic functions, backend GPU hoặc cấu hình freestanding cho PLC. Kiểm tra biên áp dụng cho phép truy cập mảng trong mã NX; mã ngoài qua FFI phải tự bảo đảm con trỏ hợp lệ. `dyn` là tagged value 16 byte, nhưng điều đó không bảo đảm một lệnh SIMD hay độ an toàn tương đương Rust. C++17 và Python là công cụ cần có lúc biên dịch. Không chạy mã NX không tin cậy trên máy sản xuất.

Logo SVG do người dùng cung cấp được lưu tại [nx-logo.svg](nexir/assets/nx-logo.svg); bản PNG và ICO được tạo từ cùng SVG.
