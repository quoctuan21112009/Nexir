# NX 1.2: Buffer sở hữu bộ nhớ, `@py_call` và SIMD x86

## Buffer có vòng đời RAII

`Buffer[T](length)` cấp phát một `owned[Buffer[T]]` với `T` là `i32`, `i64`, `f32` hoặc `f64`. Bộ nhớ nằm trong `std::vector<T>` và được giải phóng khi owner ra khỏi scope. Lệnh tạo cần `@cap("memory.alloc")`. Có thể trả owner từ hàm; compiler sinh `std::move` cho giá trị trả về. Dùng `move(owner)` khi chuyển quyền sở hữu giữa các biến hoặc vào hàm khác.

```nx
@cap("memory.alloc")
fn make_data(n: i64) -> owned[Buffer[f64]]:
    data = Buffer[f64](n)
    for i in range(n):
        data[i] = f64(i)
    return data

fn first(data: borrowed[Buffer[f64]]) -> f64:
    return data[0]
```

`len(data)` và `data[i]` kiểm tra biên. `buffer_span(data)` tạo `Span[T]` và `buffer_mutspan(data)` tạo `MutSpan[T]` **chỉ để truyền trực tiếp vào một lời gọi đồng bộ**; compiler không cho lưu view vào biến hoặc trả view từ hàm. Mã `@extern` phải tự bảo đảm không giữ con trỏ sau lời gọi. Các mảng cũ từ `[1,2]` và `zeros_f64` vẫn giữ sống đến hết tiến trình; dùng Buffer cho vòng lặp cấp phát dài hạn. Xem [ví dụ](../examples/owned_buffer.nx).

## Gọi Python bằng `@py_call`

```nx
@py_call("nx_py_worker", "mean")
def python_mean(values: Span[f64]) -> f64:
    pass
```

Compiler tự sinh wrapper và bật cầu nối CPython khi build/run tệp có `@py_call`, không cần tự khai báo `@extern` hay thêm `--python`. Decorator nhận tên module và hàm Python cố định; hàm NX có đúng một tham số `Span[f64]` hoặc `MutSpan[f64]`, thân `pass`. Kết quả hỗ trợ `f64`, `str` và `owned[Buffer[f64]]` (Python trả một dãy số, được sao chép vào Buffer NX). Trả `MutSpan[f64]` trần từ Python không được hỗ trợ vì không có owner để bảo vệ lifetime. Kết quả `str` được sao chép vào vùng chuỗi mà runtime hiện giữ đến khi tiến trình kết thúc; dùng `owned[Buffer[f64]]` cho dữ liệu mảng lớn hoặc gọi lặp nhiều lần. `str` và Buffer trả về cần quyền `memory.alloc` trên hàm gọi.

```powershell
$env:PYTHONPATH = (Resolve-Path .\examples).Path
nx build .\examples\python_decorator.nx -o .\build\python_decorator.exe
$pyHome = & py -3 -c "import sys; print(sys.base_prefix)"
$env:Path = "$pyHome;$env:Path"
.\build\python_decorator.exe
```

Máy build và chạy vẫn cần CPython tương thích, DLL, thư viện chuẩn và module Python. `@py_call` không đóng gói các package Python vào executable. Lỗi Python được in ra stderr; đường trả `f64` cũ trả NaN khi lỗi, còn đường trả `str`/Buffer dừng chương trình nếu không tạo được giá trị hợp lệ. Xem [ví dụ NX](../examples/python_decorator.nx) và [worker Python](../examples/nx_py_worker.py).

## SIMD x86

`SIMD[T,N]` hiện căn hàng dữ liệu theo 32 byte. Trên x86 với G++/Clang++, phép `+`, `-`, `*` cho `f32`/`f64` và một số phép số nguyên có nhánh AVX2 dùng intrinsics thực; `f32`/`f64` có nhánh SSE2 khi không có AVX2. Chương trình kiểm tra CPU lúc chạy và dùng vòng lặp scalar với `#pragma omp simd` cho kiểu, phép toán hoặc chiều rộng không có đường vector. G++ được gọi với `-fopenmp-simd`.

`alignas(32)` và pragma **không tự bảo đảm** vector hóa. Intrinsics bảo đảm có lệnh SIMD trong nhánh được chọn, nhưng `N` nhỏ hơn một vector, CPU không hỗ trợ, hoặc `i64` nhân vẫn có thể chạy scalar. Backend khác x86 dùng đường scalar/pragma. Không bật `-mavx2` cho toàn bộ chương trình để tránh lỗi illegal instruction trên CPU cũ.

## Bộ sưu tập native có kiểu tĩnh

`List[T]` ánh xạ sang `std::vector<T>` và tự giải phóng theo scope. `T` hiện là `i32`, `i64`, `f32`, `f64` hoặc `str`. Tạo danh sách bằng `List[i32]()` hoặc `List[i32]([3, 1, 2])`, rồi dùng `.append(value)`/`.push(value)`, `.pop()`, `.sort()`, `len(list)` và `list[index]`. Danh sách từ cú pháp `[1, 2]` cũ vẫn là `MutSpan[i32]` để giữ tương thích; dùng `List[T](...)` khi cần kích thước thay đổi. Tạo List và `.append()`/`.push()` cần `@cap("memory.alloc")`.

`Dict[K,V]` dùng `std::unordered_map`, `Set[T]` dùng `std::unordered_set`, với phần tử primitive đồng nhất. Có thể viết `{"a": 1}`, `{1, 2}`, `Dict[str,i32]()`, `Set[i32]()`, truy cập và gán `dict[key]`, kiểm tra `key in dict`, `item in set`, và dùng `set.add/remove/discard`. Truy cập khóa thiếu dừng chương trình; gán khóa mới tạo phần tử. Thứ tự duyệt Dict/Set không cố định. `Tuple[T1,T2,...]` hỗ trợ 2–8 giá trị primitive, ví dụ `fn pair() -> Tuple[i32,str]: return 1, "one"`; chỉ số tuple phải là hằng số không âm.

`Optional[T]` hỗ trợ số, bool và str: `value: Optional[i32] = None`, `value is None`, `value is not None`, `value_or(value, 0)`. Sử dụng `None` cần chú giải kiểu. `List`, `Dict`, `Set`, `Tuple` và `Optional` là kiểu C++17 native, không gọi Python.

## Cú pháp gọn

```nx
import utils

@cap("memory.alloc")
def main() -> i32:
    values = List[i32]([3, 1, 2])
    values.sort()
    doubled = [x * 2 for x in values if x > 0]
    for value in doubled[1:]:
        print(f"value={value}")
    print(utils.twice(21))
    return 0
```

`import utils` nạp `utils.nx` nằm cùng thư mục với file nhập. Compiler gộp module ở lúc build; `utils.twice(...)` gọi hàm NX native. Module không có `main`, không dùng alias, và các tên hàm/struct toàn dự án phải duy nhất. Chu trình import bị từ chối.

Vòng `for item in name` hỗ trợ `List`, `Span`, `MutSpan`, `Buffer`, `Set`, `Dict` (duyệt khóa) và `str` (từng ký tự dạng str). Cắt lát `name[start:stop:step]` hỗ trợ `List`, `Span`, `Buffer`, `str`; kết quả mảng là `List[T]` mới. List/Dict comprehension hỗ trợ một mệnh đề `for` trên tên List/Span/Buffer, có thể thêm `if` lọc. `a if condition else b`, so sánh nối chuỗi đơn giản, `in`/`not in`, và `is None`/`is not None` cũng được kiểm tra kiểu tĩnh.

Hàm có thể dùng giá trị mặc định là hằng số và được gọi bằng đối số tên, ví dụ `fn greet(name: str = "NX") -> str:` rồi `greet(name="Lan")`. Chuỗi hỗ trợ nối `+`, f-string cho số/bool/str, `.lower()`, `.upper()`, `.strip()`, `.replace(old,new)`, `.split(separator)`; mặc định `.split()` dùng dấu cách đơn. Các thao tác tạo chuỗi cần `@cap("memory.alloc")`. `str` kết quả hiện được runtime giữ đến hết tiến trình; với luồng dữ liệu dài hạn, ưu tiên Buffer/List cho dữ liệu lớn.

`struct` có thể định nghĩa `__add__`, `__sub__`, `__mul__`, `__eq__`, `__getitem__`; biểu thức tương ứng gọi phương thức đã kiểm tra kiểu. Biến cấp file khai báo dạng `NAME: i32 = 1` với hằng khởi tạo primitive; tên viết HOA được phát sinh thành hằng và không thể gán lại. Có thể dùng `global NAME` trong hàm. Xem [ví dụ bộ sưu tập](../examples/collections.nx) và [ví dụ module](../examples/modules_main.nx).

## Phạm vi chưa hỗ trợ

NX giữ kiểu tĩnh và mã native nên chưa nhận `*args`, `**kwargs`, `lambda`/closure, `yield`/generator, `enumerate`/`zip`, walrus `:=`, `for...else`, `try...finally` hoặc bắt lỗi theo lớp Python, `class` kế thừa và `with`. Comprehension lồng nhiều vòng, format spec phức tạp của f-string, module Python tùy ý và `import` có alias chưa hỗ trợ. `try/except` hiện chỉ làm việc với `Result[T]` theo cú pháp giới hạn NX. `str` hiện giữ bộ nhớ đến khi tiến trình thoát; `List`, `Dict`, `Set` và `Buffer` có vòng đời RAII.

## Gọi trực tiếp C++ và Python

```nx
@cpp_header("cmath")
@cap("ffi.cpp")
@cpp_fn("std::hypot")
fn cpp_hypot(x: f64, y: f64) -> f64:
    pass

@cap("python.interop")
@py_fn("numpy", "std")
fn py_std_dev(values: Span[f64]) -> f64:
    pass
```

`@cpp_header` là decorator trên khai báo hàm (có thể lặp nhiều lần), được phát sinh thành `#include <cmath>`. Để dùng header dự án dạng dấu nháy, truyền chuỗi chứa dấu nháy: `@cpp_header('"raylib.h"')`. Header và tên ký hiệu C++ chỉ nhận đường dẫn/tên đơn giản để tránh chèn mã vào file sinh. `@cpp_fn` tạo wrapper `static inline` gọi ký hiệu như `std::hypot`; chữ ký hiện chỉ hỗ trợ số, bool, str và kết quả void. Khai báo phải có `@cap("ffi.cpp")` và thân `pass`; hàm gọi cũng cần quyền này. Compiler C++ có thể tối ưu wrapper, nhưng `inline` không cam kết triệt tiêu chi phí ở mọi cấu hình. Với thư viện ngoài STL, vẫn phải cài header/library và chỉ định đường dẫn hoặc dùng `nx pkg cpp`/`--vcpkg` theo thư viện.

`@py_fn(module, function)` tự bật bridge CPython, tự preload module khi chương trình khởi động và yêu cầu `@cap("python.interop")` trên khai báo lẫn hàm gọi. Kiểu hiện hỗ trợ một `Span[f64]`/`MutSpan[f64]` và kết quả `f64`, `str` hoặc `owned[Buffer[f64]]`; tên module/hàm là hằng. Khi dùng `nx run app.nx`, CLI thêm thư mục Python DLL vào `PATH` của tiến trình con trên Windows và thư mục chứa file `.nx` vào `PYTHONPATH`. Khi dùng `nx build`, máy chạy vẫn cần CPython tương thích và module đã cài. Cài NumPy bằng `nx pkg python install numpy`, rồi chạy [ví dụ C++ và Python](../examples/cpp_python_ffi.nx) bằng `nx run examples/cpp_python_ffi.nx`.
