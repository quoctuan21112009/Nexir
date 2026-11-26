# Tính năng NX 1.1

NX 1.1 bổ sung một tập cú pháp có thể biên dịch và chạy trên backend C++17. Đây là tập con của các ý tưởng trong bảng so sánh với Mojo; các tính năng không có cùng mức đầy đủ hoặc bảo đảm an toàn như Mojo. Chạy `nx check FILE.nx` trước, rồi `nx build FILE.nx -o FILE.exe`.

## `struct`, phương thức và RAII

```nx
struct Counter:
    value: i32

    def __init__(self, value: i32):
        self.value = value

    fn increment(self) -> void:
        self.value += 1

    def __del__(self):
        print("released")

fn main() -> i32:
    item = Counter(4)
    item.increment()
    print(item.value)
    return 0
```

`struct` sinh `struct` C++17 và biến cục bộ có vòng đời theo scope. Trường khai báo `name: Type` (không có giá trị mặc định); nếu thiếu `__init__`, constructor nhận các trường theo thứ tự khai báo. `__del__` sinh destructor C++ và chạy khi giá trị ra khỏi scope. Truyền struct vào hàm hiện là truyền tham trị; bản sao cũng có destructor riêng. Chưa có kế thừa, method generic, overload hoặc kiểm tra an toàn khi destructor quản lý tài nguyên thô. Xem [ví dụ](../examples/structs.nx).

## `owned`, `borrowed`, `mut`, `move`

```nx
fn read(item: borrowed[Counter]) -> i32:
    return item.value

fn change(item: mut[Counter]) -> void:
    item.value += 1

@cap("memory.alloc")
fn main() -> i32:
    item: owned[Counter] = owned(Counter(10))
    print(read(item))
    change(item)
    next: owned[Counter] = move(item)
    print(read(next))
    return 0
```

`owned[Struct]` ánh xạ `std::unique_ptr`, `move(x)` ánh xạ `std::move`; frontend báo lỗi nếu dùng lại tên biến đã move. `borrowed[Struct]` là `const T&` chỉ dùng làm tham số, `mut[Struct]` là `T&` chỉ dùng làm tham số. Không lưu/trả borrow. `owned(...)` cần `@cap("memory.alloc")`. Đây **chưa phải borrow checker đầy đủ**: chưa chứng minh tính độc quyền của mọi `mut`, chưa phân tích alias qua FFI, trường lồng nhau và mọi nhánh điều khiển. Xem [ví dụ](../examples/ownership.nx).

## `fn`, generic và trait

`fn` yêu cầu kiểu trả về rõ ràng và không nhận/trả `dyn`; `def` giữ cách khai báo hiện có của NX. Cả hai vẫn được kiểm tra kiểu; `def` không chạy Python động. `@cap` tiếp tục được kiểm tra tĩnh.

```nx
fn add[T: Number](a: T, b: T) -> T:
    return a + b

trait HasValue:
    fn value(self) -> i32:
        pass

fn twice[T: HasValue](item: T) -> i32:
    return item.value() + item.value()
```

Generic một hoặc nhiều **tham số kiểu** sinh `template <typename T>` C++17. Bound dựng sẵn: `Number` (kiểu số), `Integer` (số nguyên). Tham số hằng số `size: Int` sinh non-type template parameter và hiện được suy ra từ `SIMD[T,size]`, ví dụ [cộng vector generic](../examples/generic_simd.nx). Cùng một `T` trong các tham số phải suy ra cùng một kiểu. `trait` khai báo phương thức với các tham số số/bool/str hoặc `Self`, kết quả số/bool hoặc `Self`, thân `pass`; compiler kiểm tra struct ở điểm gọi và sinh phép kiểm tra `std::void_t`/`static_assert`. Chưa có cú pháp truyền đối số generic tường minh, associated type hoặc trait inheritance. Xem [generic](../examples/generics.nx), [trait](../examples/traits.nx), [trait có tham số](../examples/trait_methods.nx) và [trait dùng `Self`](../examples/trait_self.nx).

## SIMD và đa luồng

```nx
@cap("memory.alloc")
fn main() -> i32:
    data = zeros_f64(16)
    for i in parallel_range(16):
        data[i] = f64(i) * 2.0
    a: SIMD[f64, 4] = SIMD[f64, 4](1.0, 2.0, 3.0, 4.0)
    b: SIMD[f64, 4] = SIMD[f64, 4](2.0, 2.0, 2.0, 2.0)
    c = a * b
    print(data[7], c[2])
    return 0
```

`SIMD[T,N]` nhận `T` là `i32`, `i64`, `f32`, `f64` và `N` là hằng số 1–1024. NX 1.1 dùng `std::array<T,N>` và phụ thuộc vào auto-vectorization; [NX 1.2](features-1.2.md) bổ sung nhánh AVX2/SSE2 bằng intrinsics cho các kiểu/chiều rộng hỗ trợ. `parallel_range(stop)` hoặc `parallel_range(start,stop)` chia các vòng lặp cho `std::thread`; không hỗ trợ bước khác 1, `break`, `continue`, `return` hoặc gán trực tiếp biến/field dùng chung trong thân vòng. Ghi vào mảng qua chỉ số vẫn có thể gây data race nếu nhiều vòng lặp cùng truy cập một ô; người viết mã phải chia vùng dữ liệu độc lập. Xem [ví dụ](../examples/simd_parallel.nx).

## `Result[T]`, `raises` và `try/except`

```nx
fn checked(value: i32) -> i32 raises:
    if value == 0:
        raise "zero"
    return 10 / value

fn main() -> i32:
    try:
        result = checked(2)
        print(result)
    except:
        print("failed")
    return 0
```

`raises` chuyển kiểu trả về thành `Result[T]` và `return value` thành giá trị thành công; `raise "message"` trả lỗi. `try` phải bắt đầu bằng một phép gán từ `Result[T]`, có đúng một `except:` trống, không có `else`/`finally`. Ngoài `try`, dùng `is_ok(result)`, `unwrap(result)`, `error(result)`; `error` yêu cầu biến Result có tên để giữ lifetime của chuỗi. Backend dùng `std::variant<T,std::string>` và không phát sinh `throw/catch` cho lỗi dự kiến. Đây không phải lời hứa “zero cost” tuyệt đối: lỗi chuỗi có cấp phát, và lỗi runtime khác vẫn có thể kết thúc tiến trình. `Result[T]` hiện nhận số, bool hoặc str; chưa có error type tùy biến hay tự động truyền lỗi qua nhiều lời gọi. Xem [ví dụ](../examples/result_errors.nx).

## Giới hạn chung

NX 1.1 vẫn dùng Python AST để bootstrap frontend và cần compiler C++17 khi build. `@extern` là ranh giới mã native đáng tin cậy, không được borrow checker bảo vệ. Các mảng `Span`/`MutSpan` của runtime cũ vẫn giữ bộ nhớ đến khi thoát chương trình; `owned[Struct]` không thay đổi quy tắc đó. Cần kiểm tra mã C++ được sinh (`nx emit`) và test native khi dùng destructor, đồng thời hoặc FFI.
