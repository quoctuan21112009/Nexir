# Hướng dẫn NX 1.2 trên Windows

Xem thêm [NX 1.1](features-1.1.md) và [NX 1.2](features-1.2.md) cho `struct`, Buffer có ownership, bộ sưu tập, module, `@cpp_fn`, `@py_fn`, SIMD, song song và xử lý lỗi.

Tài liệu này mô tả **những chức năng đang chạy trong repo**, cách đóng gói compiler thành một `nx.exe`, cài thư viện ngoài, liên kết C++, nhúng Python và quản lý bộ nhớ. NX là compiler tạo C++17 rồi gọi G++/Clang++; `nx.exe` chứa frontend Python, nhưng vẫn cần compiler C++ khi `build`/`run`. Chương trình NX đã build không cần frontend Python, trừ khi dùng `--python`.

## 1. Tạo `nx.exe` và cài vào PATH

Mở PowerShell tại thư mục repo, chạy:

```powershell
.\scripts\build-windows.ps1
.\scripts\install-windows.ps1
```

Lệnh build tạo `release\nx.exe` (một file thực thi CLI), `release\nexir-language.vsix` và `release\nexir-windows-1.2.0.zip` chứa các file phát hành. Script cài đặt chép `nx.exe` vào `%LOCALAPPDATA%\Programs\Nexir\bin`, thêm thư mục này vào **User PATH**, nhận diện G++ ở `C:\msys64\ucrt64\bin\g++.exe` và cài VSIX vào VS Code/Antigravity nếu tìm được lệnh editor. Mở terminal và editor mới để nhận PATH mới, rồi chạy `nx doctor`.

Nếu chưa có G++, dùng `.\scripts\install-windows.ps1 -InstallCompiler` để cài MSYS2 qua `winget`, sau đó cài GCC UCRT64 qua `pacman`. Cài đặt MSYS2 ở vị trí khác `C:\msys64` thì tự đặt `NEXIR_CXX` thành đường dẫn tuyệt đối đến `g++.exe`. Có thể dùng G++/Clang++ sẵn có; lệnh `nx doctor` phải hiển thị đường dẫn compiler. Máy xây dựng `nx.exe` cần Python 3.10+ và Internet để cài PyInstaller; máy sử dụng chỉ cần `nx.exe` và C++ compiler.

Tham khảo tài liệu chính thức: [MSYS2 Getting Started](https://www.msys2.org/), [PyInstaller one-file](https://pyinstaller.org/en/stable/usage.html).

Nếu editor không có lệnh CLI trong PATH, dùng **Extensions → Install from VSIX...** và chọn `release\nexir-language.vsix`. Sau đó cấu hình `nexir.runCommand` thành đường dẫn tuyệt đối đến `nx.exe`, hoặc để `nx` nếu PATH đã hoạt động. Chạy lệnh **Nexir: Enable Nexir File Icons** trong Command Palette để bật biểu tượng `.nx` ở Explorer. Antigravity chỉ chạy extension khi phiên bản đang dùng hỗ trợ VS Code extension host/API.

## 2. Chương trình đầu tiên

Tạo `hello.nx` UTF-8:

```nx
def add(a: i32, b: i32) -> i32:
    return a + b

def main() -> i32:
    print("Xin chào NX", add(2, 3))
    return 0
```

```powershell
nx doctor
nx check .\hello.nx
nx run .\hello.nx
nx build .\hello.nx -o .\hello.exe
.\hello.exe
nx emit .\hello.nx -o .\hello.cpp
```

`check` chỉ phân tích cú pháp và kiểu, không cần G++. `run` tạo executable tạm rồi chạy. `build` tạo executable tại `-o`; `--emit-cpp` ghi kèm mã C++ để xem. `init <directory>` tạo project mẫu. `--shared` tạo DLL (hoặc `.so` trên Linux) cho hàm `@export`.

## 3. Cú pháp và chức năng

NX dùng thụt lề như Python, nhưng chỉ chấp nhận tập con nêu trong [hướng dẫn cú pháp](syntax.md), [NX 1.1](features-1.1.md) và [NX 1.2](features-1.2.md). Ở cấp file có import `.nx` cùng thư mục, hằng/biến primitive có chú giải kiểu, hàm, struct và trait. `main()` không có tham số; kiểu trả về là `void` hoặc `i32` để làm exit code.

| Nhóm | Hiện hỗ trợ |
| --- | --- |
| Kiểu | `i32`, `i64`, `f32`, `f64`, `bool`, `str`, `dyn`, `void`, `Span[T]`, `MutSpan[T]`, `Buffer[T]`, `List[T]`, `Dict[K,V]`, `Set[T]`, `Tuple[...]`, `Optional[T]`, `Ring[f64]` |
| Luồng | `if`/`else`, `while`, `for ... in range(...)`, `for item in name`, `break`, `continue`, `return`, `pass` |
| Hàm dựng sẵn | `print`, `len`, `zeros_f32`, `zeros_f64`, `read_text`, `tcp_connect`, `ring_f64`, `ring_write`, `ring_valid`, `ring_read`, `type_name`, `as_i64`, `as_f64` |
| Decorator | `@cap(...)`, `@extern("symbol")`, `@export`, `@native`, `@cpp_header`, `@cpp_fn`, `@py_fn` |
| Native | `--cpp`, `--object`, `--include-dir`, `--library-dir`, `--library`, `--shared`, `--icon` |

`dyn` hỗ trợ giá trị số, chuỗi, cộng số/nối chuỗi và kiểm tra kiểu khi lấy lại. `@cap` kiểm tra tĩnh các quyền `memory.alloc`, `filesystem.read`, `network.connect`, `ffi.cpp`, `python.interop`; nó không phải sandbox hệ điều hành. NX 1.1 có generic và ownership ở phạm vi giới hạn; NX 1.2 thêm import nội bộ, chưa có kế thừa `class`, borrow checker đầy đủ, GPU backend hay freestanding PLC runtime.

## 4. Thư viện C++ trên máy

Với mã nguồn C++ tự viết, khai báo C ABI trong NX:

```nx
@extern("nx_external_square")
def square(x: f64) -> f64:
    pass

def main():
    print(square(3.0))
```

```cpp
extern "C" double nx_external_square(double x) { return x * x; }
```

```powershell
nx build .\examples\native_math.nx --cpp .\examples\native_math.cpp -o .\build\native_math.exe
.\build\native_math.exe
```

Để tải thư viện C++ bằng vcpkg:

```powershell
nx pkg cpp bootstrap
nx pkg cpp install zlib
nx pkg cpp path
```

Mặc định Windows dùng triplet `x64-mingw-static`. Đặt `VCPKG_ROOT` nếu muốn thư mục vcpkg khác mặc định `%USERPROFILE%\.nexir\vcpkg`; dùng `--triplet` cho `install`/`remove`/`path` khi cần. vcpkg cần Git, G++ và mạng. MinGW triplets là community triplets; một số port có thể không build được. Để liên kết thư viện đã cài, dùng `--vcpkg` cộng `--library` (tên thư viện linker, thường có trong tài liệu của package):

Xem [hướng dẫn MinGW của vcpkg](https://learn.microsoft.com/en-us/vcpkg/users/platforms/mingw) để chọn triplet và kiểm tra hỗ trợ từng port.

```powershell
nx build .\app.nx --cpp .\bridge.cpp --vcpkg x64-mingw-static --library z -o .\app.exe
```

`--vcpkg` thêm `<VCPKG_ROOT>\installed\<triplet>\include` và `lib` vào lệnh compiler; nó **không tự biết** danh sách thư viện con. Với hàm C++ có chữ ký primitive, `@cpp_header` + `@cpp_fn` tạo wrapper trực tiếp không cần C ABI; xem [ví dụ chung](../examples/cpp_python_ffi.nx). Có thể thêm `--include-dir`, `--library-dir`, `--library`, `--object` nhiều lần. Không trộn thư viện MSVC `.lib` với linker MinGW; dùng triplet/toolchain cùng ABI.

## 5. Thư viện Python và cầu nối CPython

`nx.exe` không chứa pip hoặc các package Python. Cài một Python 3 tương thích và tải package vào **chính interpreter dùng lúc build/chạy**:

```powershell
nx pkg python install numpy
nx pkg python show numpy
```

`NEXIR_PYTHON` trỏ đến `python.exe` cụ thể khi có nhiều bản Python; nếu không đặt, bản exe chọn `py -3` rồi `python` trong PATH. Với `--python`, NX tìm header/lib bằng `sysconfig` của interpreter đó. Ví dụ cầu nối dùng `Span[f64]`:

```nx
@extern("nx_py_call_f64_span")
def mean(module: str, function: str, values: Span[f64]) -> f64:
    pass

def main():
    values = [1.0, 2.0, 3.0]
    print(mean("worker", "calculate", values))
```

```python
# worker.py, đặt ở thư mục làm việc khi chạy app.exe
def calculate(values):
    return sum(values) / len(values)
```

```powershell
nx build .\app.nx --python --python-preload worker -o .\app.exe
$pyHome = & py -3 -c "import sys; print(sys.base_prefix)"
$env:Path = "$pyHome;$env:Path"
.\app.exe
```

Bridge nhận một `Span[f64]` hoặc `MutSpan[f64]`. Lời gọi `@extern` cũ trả số thực; [`@py_call`/`@py_fn` của NX 1.2](features-1.2.md) còn hỗ trợ `str` và `owned[Buffer[f64]]`. `@py_fn` tự bật bridge và preload module; `nx run` tự thêm Python DLL vào PATH của tiến trình con trên Windows. `MutSpan` cho phép Python sửa buffer; không giữ ctypes view sau lời gọi, không sửa đồng thời ở hai phía. Với NumPy, `np.ctypeslib.as_array(values)` tạo view đầu vào; thư viện downstream có thể copy. Ứng dụng Python vẫn cần Python DLL, thư viện chuẩn, `worker.py` và package đã cài ở máy chạy. Khi chạy `.exe` từ `nx build` bên ngoài CLI, thư mục chứa `python3xx.dll` vẫn phải có trong `PATH`. Xem [chi tiết Python C API](python-c-api.md).

## 6. Kiểm soát bộ nhớ và giới hạn an toàn

- `Span[T]` chỉ đọc, `MutSpan[T]` đọc/ghi. Phép `data[i]` trong mã NX kiểm tra biên lúc chạy; `len(data)` lấy số phần tử.
- `[1, 2, 3]` và `zeros_f32/zeros_f64` cấp phát vùng nhớ native. `zeros_*`, `ring_f64` và `read_text` yêu cầu `@cap("memory.alloc")`; `read_text` còn cần `filesystem.read`.
- Mảng cũ từ literal/`zeros_*` và chuỗi do runtime giữ đến khi tiến trình kết thúc để con trỏ ổn định. `owned[Struct]` và [`owned[Buffer[T]]`](features-1.2.md) được giải phóng theo scope. **Hiện chưa có `free` cho Span cũ, bộ cấp phát tùy chỉnh hay giới hạn heap**. Tránh cấp phát lặp vô hạn; dùng Buffer cho dữ liệu lớn sống theo scope, hoặc `Ring[f64]` có sức chứa cố định cho dữ liệu luân phiên. `Ring` kiểm tra thế hệ để báo dữ liệu đã bị ghi đè.
- `dyn` dùng bộ đếm tham chiếu cho chuỗi/object, còn `Ring` dùng `shared_ptr`; bản thân các đối tượng này được giải phóng khi hết tham chiếu. Việc đó không áp dụng cho vùng lưu Span được giữ đến hết tiến trình.
- Không trả `Span` từ hàm NX; compiler chặn thao tác đó vì chưa có kiểm tra lifetime. Với `@extern`, C++/Python phải tự bảo đảm lifetime, layout C ABI, chỉ số và đồng bộ. `@cap` không hạn chế mã C++ được liên kết.

Ví dụ tái sử dụng mảng:

```nx
@cap("memory.alloc")
def main():
    samples = zeros_f64(1024)
    for i in range(1024):
        samples[i] = f64(i)
    print(samples[0], samples[1023])
```

## 7. Lệnh tra cứu nhanh

```text
nx --version
nx doctor
nx init DIRECTORY
nx check FILE.nx
nx run FILE.nx
nx emit FILE.nx -o FILE.cpp
nx build FILE.nx -o FILE.exe [--cpp FILE.cpp] [--vcpkg TRIPLET] [--library NAME]
nx pkg cpp bootstrap|install NAME|remove NAME|path
nx pkg python install NAME|show NAME
```

Chạy `nx <lệnh> --help` để xem toàn bộ cờ. Kết quả `doctor` chỉ chứng minh compiler được tìm thấy; thử `nx build examples\hello.nx -o build\hello.exe` để kiểm tra biên dịch và linker.
