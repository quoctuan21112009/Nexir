# Hướng dẫn cú pháp NX 2.0

Xem thêm [Tính năng NX 2.0](features-2.0.md). Tệp nguồn có đuôi `.nx`, mã hóa UTF-8 và dùng thụt lề như Python. NX biên dịch sang C++17 native, kiểm tra kiểu, ownership và quyền `@cap` tại thời điểm biên dịch.

## 1. Viết và chạy chương trình

Mỗi tệp thực thi cần có `def main()` không tham số (`void` hoặc `-> i32`). Module thư viện import không cần `main()`.

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
nx debug hello.nx
nx build hello.nx -o hello.exe
.\hello.exe
```

Ở cấp tệp có thể đặt `import`, hằng/biến cấp tệp, `def`, `async def`, `class`, `enum` và `trait`. Từ khóa `struct` và `fn` ở NX 1.x đã được thay thế hoàn toàn bằng `class` và `def`.

## 2. Kiểu dữ liệu & Chuyển đổi số tự do

| Kiểu | Ý nghĩa |
| --- | --- |
| `i8`, `i16`, `i32`, `i64`, `i128` | Số nguyên có dấu 8..128 bit (`int` tương đương `i64`) |
| `u8`, `u16`, `u32`, `u64`, `u128` | Số nguyên không dấu 8..128 bit |
| `f32`, `f64` | Số thực 32/64 bit (`float` tương đương `f64`) |
| `bool`, `str` | Boolean (`True`/`False`) và chuỗi `NxStr` sở hữu bộ nhớ (SSO + cache) |
| `dyn` | Giá trị động 16-byte cho hybrid typing |
| `List[T]`, `Dict[K,V]`, `Set[T]`, `Tuple[...]` | Bộ sưu tập RAII native, cú pháp `[1, 2]`, `{}`, `{1, 2}` |
| `Span[T]`, `MutSpan[T]`, `Buffer[T]` | Vùng nhớ liên tiếp tốc độ cao (`List[T]` truyền trực tiếp vào `Span[T]`) |
| `Optional[T]`, `Result[T]`, `Result[T,E]` | Hỗ trợ mọi kiểu nguyên thủy, `class`, `enum`, `owned[Class]` |
| `Future[T]`, `Channel[T]`, `Mutex[T]`, `GpuBuffer[T]` | Kiểu bất đồng bộ, đa luồng và tính toán GPU/SIMD |

Các kiểu số chuyển đổi tự do qua lại (`i32 <-> i64 <-> i128 <-> f64`, `u8 <-> i16`, v.v.); chuyển đổi thu hẹp tự động kiểm tra tràn miền giá trị lúc chạy.

## 3. `class`, `enum`, `trait` và `match`

```nx
enum Shape:
    Circle(radius: f64)
    Rect(w: f64, h: f64)
    Point

    def area(self) -> f64:
        match self:
            case Shape.Circle(r):
                return 3.14159265358979 * r * r
            case Shape.Rect(w, h):
                return w * h
            case Shape.Point:
                return 0.0

class Box:
    value: i32 = 0
    def __init__(self, value: i32):
        self.value = value
    def __del__(self):
        pass
```

## 4. Import đa file và đa ngôn ngữ

- `import mod` / `from mod import func as f`: nạp `mod.nx` với namespace riêng.
- `import py.math as math` / `import py.statistics`: chạy thẳng module Python không cần file wrapper (`@cap("python.interop")`).
- `import cpp.cmath as cmath`: gọi thẳng hàm C++ STL/header (`@cap("ffi.cpp")`).
- `import nx.asyncio`, `import nx.thread`, `import nx.fs`, `import nx.net`, `import nx.gpu`, `import nx.sandbox`, `import nx.debug`.
