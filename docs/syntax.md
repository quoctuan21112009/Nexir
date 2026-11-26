# Hướng dẫn cú pháp NX 1.0

Tài liệu này mô tả **cú pháp compiler hiện hỗ trợ**. Tệp nguồn có đuôi `.nx`, mã hóa UTF-8 và dùng thụt lề để xác định khối lệnh. Cú pháp nhìn gần với Python, nhưng NX chỉ nhận một tập con được kiểm tra kiểu và biên dịch sang C++17; NX không chạy mã Python.

## 1. Viết và chạy chương trình

Mỗi tệp cần có `main()` không tham số. `main` có thể không khai báo kiểu trả về (ngầm là `void`) hoặc trả `i32` làm mã thoát tiến trình.

```nx
def add(a: i64, b: i64) -> i64:
    return a + b

def main() -> i32:
    print("2 + 3 =", add(2, 3))
    return 0
```

```powershell
nx check hello.nx
nx run hello.nx
nx build hello.nx -o hello.exe
.\hello.exe
```

Trên Windows, nếu lệnh `nx` chưa có trong `PATH`, dùng `py -3 -m nexir` thay cho `nx`. `nx doctor` kiểm tra compiler C++17. `nx emit hello.nx -o hello.cpp` ghi mã C++ được sinh ra.

Chỉ **định nghĩa hàm** được đặt ở cấp tệp. Có thể viết chú thích bằng `#`. Không đặt `import`, lớp hay biến toàn cục ở cấp tệp.

## 2. Kiểu dữ liệu

| Kiểu | Ý nghĩa | Ví dụ |
| --- | --- | --- |
| `i32`, `i64` | Số nguyên có dấu 32/64 bit | `count: i64 = 10` |
| `f32`, `f64` | Số thực 32/64 bit | `rate: f32 = f32(0.5)` |
| `bool` | `True` hoặc `False` | `ready: bool = True` |
| `str` | Chuỗi | `name: str = "NX"` |
| `dyn` | Giá trị động có nhãn kiểu | `value: dyn = 42` |
| `Span[T]` | Vùng dữ liệu số chỉ đọc | `Span[f32]` |
| `MutSpan[T]` | Vùng dữ liệu số đọc/ghi | `MutSpan[i64]` |
| `Ring[f64]` | Bộ đệm vòng số thực | `Ring[f64]` |
| `void` | Hàm không trả giá trị | `def log() -> void:` |

`T` của Span và MutSpan chỉ nhận `i32`, `i64`, `f32`, `f64`. Hằng số nguyên vừa trong 32 bit được suy ra là `i32`; số nguyên lớn hơn là `i64` nếu vẫn nằm trong 64 bit. Hằng số thực như `1.5` là `f64`. Dùng `i32(x)`, `i64(x)`, `f32(x)`, `f64(x)` để chuyển giữa các kiểu số. Khi truyền/gán, compiler tự mở rộng `i32 → i64`, `f32 → f64` và `MutSpan[T] → Span[T]`; các chuyển đổi khác cần viết rõ hoặc không được hỗ trợ.

## 3. Hàm và biến

Tham số phải có kiểu, là tham số vị trí bắt buộc. Kiểu trả về bỏ trống nghĩa là `void`. Hàm trả kiểu khác `void` phải `return` trên mọi nhánh có thể đi đến cuối hàm.

```nx
def twice(value: i32) -> i32:
    return value * 2

def main():
    first = 21              # suy ra i32
    second: i32 = twice(first)
    first = 22              # gán lại cùng kiểu
    print(second, first)    # 42 22
```

Biến khai báo trong `if`, `while` hoặc `for` chỉ có hiệu lực trong khối đó. Không khai báo tên trùng tham số hoặc biến đã khai báo trong phạm vi đang có hiệu lực. Tên bắt đầu bằng `nx_` và tên hàm dựng sẵn được dành riêng.

## 4. Biểu thức và điều khiển luồng

- Số học: `+`, `-`, `*`, `/`, `%` (`%` chỉ áp dụng cho số nguyên). Chia số nguyên làm tròn về 0.
- So sánh: `==`, `!=`, `<`, `<=`, `>`, `>=`. Chuỗi chỉ hỗ trợ `==` và `!=`. Chưa hỗ trợ so sánh nối như `a < b < c`.
- Logic: `and`, `or`, `not` với toán hạng `bool`.
- Gán tăng/giảm: `+=`, `-=`, `*=`, `/=` trên biến số hoặc phần tử MutSpan.
- `if`/`else`, `while`, `for name in range(stop)`, `range(start, stop)`, `range(start, stop, step)`, `break`, `continue`, `return`, `pass`.

```nx
def main():
    total: i64 = 0
    for i in range(10):
        if i == 5:
            break
        if i % 2 == 0:
            continue
        total += i
    print(total)  # 4
```

`range` nhận 1–3 đối số số nguyên, đầu cuối không bao gồm `stop`; bước `0` gây lỗi khi chạy. Điều kiện `if`/`while` phải là `bool`. `for ... else` và `while ... else` chưa được hỗ trợ.

## 5. Mảng và Span

Danh sách dạng `[1, 2, 3]` chỉ chứa số, được suy ra thành `MutSpan[T]`. `zeros_f32(n)` và `zeros_f64(n)` tạo mảng số 0; lời gọi này cần quyền `memory.alloc`. `len(span)` trả độ dài. Chỉ số bắt đầu từ 0 và được kiểm tra biên khi chạy.

```nx
def sum_values(data: Span[f32]) -> f32:
    total: f32 = f32(0)
    for i in range(len(data)):
        total += data[i]
    return total

@cap("memory.alloc")
def main():
    values: MutSpan[f32] = [f32(1), f32(2), f32(3)]
    values[1] = f32(4)
    print(sum_values(values))   # 8
    scratch = zeros_f32(2)
    scratch[0] = f32(7)
    print(scratch[0], len(scratch))
```

`Span[T]` không thể ghi. `MutSpan[T]` có thể truyền cho tham số `Span[T]`. Không thể trả Span từ hàm vì compiler chưa kiểm tra được thời gian sống của vùng nhớ. Danh sách rỗng `[]` chưa có kiểu suy luận; dùng `zeros_f32(0)` hoặc `zeros_f64(0)`.

## 6. Giá trị động

`dyn` lưu giá trị kèm nhãn kiểu. Hiện hỗ trợ cộng số hoặc nối hai chuỗi bằng `+`; `type_name(x)` đọc tên kiểu, `as_i64(x)` và `as_f64(x)` lấy giá trị số. Lấy sai kiểu gây lỗi khi chạy.

```nx
def main():
    value: dyn = 41
    value = value + 1
    print(type_name(value), as_i64(value))  # i64 42
    label: dyn = "Ne"
    label = label + "xir"
    print(label)
```

## 7. Bộ đệm vòng

`ring_f64(capacity)` tạo `Ring[f64]` và cần `memory.alloc`. `ring_write(ring, index, value)` ghi theo chỉ số tuyệt đối; `ring_valid(ring, index)` cho biết dữ liệu còn hợp lệ; `ring_read(ring, index)` dừng chương trình nếu vị trí chưa ghi hoặc đã bị ghi đè.

```nx
@cap("memory.alloc")
def main():
    buffer = ring_f64(2)
    ring_write(buffer, 0, 1.5)
    print(ring_valid(buffer, 0), ring_read(buffer, 0))
    ring_write(buffer, 2, 2.5)
    print(ring_valid(buffer, 0))  # false
```

## 8. Quyền `@cap` và I/O

| Thao tác | Quyền cần khai báo trên hàm gọi |
| --- | --- |
| `zeros_f32`, `zeros_f64`, `ring_f64` | `memory.alloc` |
| `read_text(path)` | `filesystem.read`, `memory.alloc` |
| `tcp_connect(host, port)` | `network.connect` |
| Gọi hàm NX có `@cap` | Toàn bộ quyền của hàm được gọi |

```nx
@cap("filesystem.read", "memory.alloc")
def show_file(path: str):
    print(read_text(path))

@cap("filesystem.read", "memory.alloc")
def main():
    show_file("notes.txt")
```

`print(a, b, ...)` in các giá trị cách nhau bằng dấu cách và kết thúc bằng dòng mới. `@cap` là kiểm tra tĩnh khi biên dịch, không cô lập chương trình khỏi hệ điều hành.

## 9. Liên kết mã native

`@extern("symbol")` khai báo một ký hiệu C ABI bên ngoài; thân hàm phải chỉ có `pass`. Dùng `nx build ... --cpp file.cpp` hoặc các tùy chọn liên kết `--object`, `--include-dir`, `--library-dir`, `--library`. Xem [ví dụ native](../examples/native_math.nx).

`@export` xuất hàm NX sang thư viện chia sẻ khi build với `--shared`. Chữ ký xuất hiện hỗ trợ tham số số, bool, Span/MutSpan và kết quả số, bool hoặc void. Tệp vẫn cần `main()`. `@native` là decorator được nhận diện; mọi hàm NX hiện đều đi qua cùng backend C++17.

## 10. Các giới hạn cú pháp dễ gặp

- Không dùng Python `import`, `class`, `dict`, `tuple`, `lambda`, `try`, `with`, thuộc tính đối tượng hoặc gọi hàm với đối số đặt tên.
- Không dùng tham số mặc định, `*args`, `**kwargs` hay kiểu generic tùy ý.
- `str` chưa có các phương thức Python; chuỗi thường không cộng trực tiếp, chỉ hỗ trợ nối khi là `dyn`.
- Compiler dùng cú pháp thụt lề giống Python nhưng **không thực thi thư viện Python** từ tệp `.nx`.
- Mã qua FFI phải tự bảo đảm con trỏ và ABI hợp lệ.

Khi gặp lỗi, chạy `nx check file.nx` trước; thông báo thường chỉ dòng và cột. Xem thêm [README](../README.md) để biết yêu cầu cài đặt và giới hạn an toàn.
