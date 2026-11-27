# Lộ trình phát triển NX

Tài liệu này chuyển đề xuất định vị NX thành lộ trình có phạm vi, trạng thái và tiêu chí nghiệm thu. Đây là kế hoạch phát triển, **không phải tuyên bố rằng mọi thành phần bên dưới đã được triển khai**.

## Định vị

NX hướng tới giảm chi phí chuyển đổi giữa cú pháp dễ viết và native code: người dùng viết phần điều phối bằng NX, giữ vòng lặp/tính toán phù hợp ở native, và gọi hệ sinh thái Python/C/C++ qua các ranh giới có kiểu cùng quy tắc vòng đời rõ ràng. NX không đặt mục tiêu thay PyTorch, Transformers, llama.cpp hay các framework AI hiện có; cũng không hứa nhanh hơn C++ hoặc Python trong mọi tác vụ.

## Trạng thái hiện tại đã kiểm chứng trong mã nguồn

| Khu vực | Đã có | Chưa nên tuyên bố |
| --- | --- | --- |
| Frontend và type system | Bootstrap compiler Python; cú pháp NX được kiểm tra bằng AST; kiểu tĩnh, `fn`, generic/trait trong phạm vi mô tả ở tài liệu tính năng | Frontend độc lập, type system/lifetime hoàn chỉnh |
| Native backend | Sinh C++17 rồi gọi compiler C++ để tạo executable/shared library; `check`, `emit`, `build`, `run` | NX-IR riêng, CFG/SSA, LLVM backend, tối ưu hóa liên thủ tục |
| Bộ nhớ và mảng | `Buffer`, `Span`/`MutSpan`, ownership một phần, kiểm tra chỉ số; Python bridge tạo view `ctypes` trên input `f64` span | Tensor/ndarray API, zero-copy tổng quát, lifetime checker đầy đủ; output từ Python hiện được sao chép vào Buffer NX |
| Python | Embedded CPython; `@py_call`/`@py_fn` cho một `Span[f64]` hoặc `MutSpan[f64]`, kết quả `f64`, `str` hoặc `owned[Buffer[f64]]` | Đối tượng Python tổng quát, chuỗi/đối tượng model làm input, callback tổng quát, tự đóng gói môi trường Python |
| C/C++ | `@extern` cho C ABI; `@cpp_header`/`@cpp_fn` cho một tập chữ ký primitive; tùy chọn linker/nguồn C++ | Tương thích C++ ABI rộng hoặc binding tự động cho mọi thư viện |
| Tối ưu hóa | Một số SIMD x86 có dispatch runtime; `parallel_range` có giới hạn; C++ compiler tối ưu khi build | SIMD portable tổng quát, profiler/benchmark suite bao phủ, tối ưu hóa tự động được bảo đảm |
| GPU/AI serving | Có thể liên kết thư viện bên ngoài bằng đường native hiện có | GPU backend NX, CUDA/ROCm integration chuyên biệt, adapter vLLM hoặc serving native |
| Công cụ/package | CLI, VS Code syntax/run support, manifest `nx.toml` có validate | Dependency resolver/registry/lockfile, debugger và IDE language server đầy đủ |

## Các nguyên tắc thiết kế

1. **Ranh giới có kiểu:** khai báo rõ dữ liệu, quyền, ownership, lỗi và ABI tại mỗi lời gọi ngoại vi.
2. **Batch trước, gọi nhiều lần sau:** gom dữ liệu qua một lần gọi FFI/Python thay vì chuyển từng phần tử.
3. **Zero-copy chỉ khi được chứng minh:** ghi rõ bên nào sở hữu bộ nhớ, thời hạn view, khả năng mutate và nơi nào có copy. Không suy diễn “zero-copy end-to-end” từ một view đầu vào.
4. **Tích hợp theo ABI thích hợp:** C qua C ABI; Python qua CPython bridge; C++ qua adapter theo thư viện/toolchain; GPU trước hết dùng runtime/thư viện vendor.
5. **Tối ưu có đo lường:** benchmark nhiều kích thước dữ liệu, kiểm tra đúng đắn và báo compiler/CPU/runtime; không hứa parity hiệu năng nếu chưa đo.
6. **Không thực thi mã tùy ý trong compiler:** package manager, plugin và build tooling phải tách biệt dữ liệu cấu hình với việc thực thi mã.

## Lộ trình và tiêu chí hoàn thành

### Giai đoạn 1 — Cú pháp, kiểu, module và package (ưu tiên rất cao)

**Nền móng hiện tại:** cú pháp/type checking, module `.nx` cùng thư mục và định dạng manifest metadata đã tồn tại ở mức giới hạn.

**Công việc:** đặc tả ngữ nghĩa/phiên bản cú pháp; cải thiện lỗi có vị trí; kiểm tra import và biểu đồ module; package resolver có lockfile, kiểm tra checksum và cài đặt tái lập; giữ tương thích ngược có kiểm thử.

**Nghiệm thu:** tài liệu cú pháp phản ánh test; lỗi type/import có file-dòng-cột; package resolution tái lập từ lockfile; reject path traversal, checksum sai và manifest không hợp lệ; test không tải/chạy package trong bước validate.

### Giai đoạn 2 — NX-IR, CFG/SSA và tối ưu (ưu tiên rất cao)

**Nền móng hiện tại:** compiler kiểm tra AST rồi phát sinh C++ trực tiếp; chưa có NX-IR/CFG/SSA.

**Công việc:** tạo IR có type và source location; lowering AST → IR; verifier cho block/terminator/type; xây CFG và SSA; đưa constant folding/dead-code elimination vào pass độc lập; giữ C++17 backend làm backend đầu tiên.

**Nghiệm thu:** mọi chương trình trong corpus hiện tại có thể qua frontend → IR verifier → backend; golden tests cho IR; pass có kiểm thử trước/sau và không đổi kết quả; `emit`/diagnostic chỉ rõ phase lỗi. Chỉ cân nhắc LLVM sau khi IR ổn định và có benchmark chứng minh nhu cầu.

### Giai đoạn 3 — Python/C/C++ interoperability (ưu tiên rất cao)

**Nền móng hiện tại:** C ABI và adapter C++/CPython có tập chữ ký hữu hạn.

**Công việc:** công bố ABI versioning; bảng chuyển đổi kiểu và ownership; truyền lỗi có cấu trúc; kiểm tra ABI/linker settings và toolchain; adapter thử nghiệm cho NumPy/PyTorch và thư viện C/C++ phổ biến mà không đưa phụ thuộc bắt buộc vào compiler.

**Nghiệm thu:** integration tests chạy trong CI cho các toolchain được hỗ trợ; kiểm tra ABI/layout; stress tests về lifetime, lỗi Python, nhiều thread và shutdown; benchmark chi phí boundary-call và dữ liệu lớn/nhỏ. Tài liệu phải liệt kê phiên bản Python/compiler đã xác nhận.

### Giai đoạn 4 — Tensor, ndarray và chia sẻ bộ nhớ (ưu tiên cao)

**Công việc:** API view tensor/ndarray có dtype, shape, strides, alignment, mutability/device; kiểm tra overflow và kích thước; chỉ nhận buffer exporter được hỗ trợ; mô tả owner giữ sống trong suốt thời gian mượn; thêm đường copy rõ ràng làm fallback.

**Nghiệm thu:** test shape/stride/dtype sai, buffer rỗng, read-only, lifetime và concurrent access; xác nhận chia sẻ con trỏ cho đúng trường hợp hỗ trợ; benchmark đo cả thời gian copy lẫn call overhead; không gọi là zero-copy cho trường hợp có copy.

### Giai đoạn 5 — SIMD, đa luồng và tối ưu bộ nhớ (ưu tiên cao)

**Nền móng hiện tại:** SIMD x86 và `parallel_range` hỗ trợ các trường hợp giới hạn.

**Công việc:** kiểm thử scalar fallback/dispatch trên CPU không hỗ trợ ISA; chính sách thread pool và tránh oversubscription với BLAS/PyTorch; profiling; kiểm tra overflow, alias và data race; tối ưu allocation/lifetime sau khi có benchmark.

**Nghiệm thu:** kết quả scalar/SIMD/parallel tương đương; test race bằng sanitizer phù hợp; benchmark nhiều kích thước và số thread; fallback chạy được trên CPU baseline; không bật song song hóa tự động nếu chưa chứng minh độc lập dữ liệu.

### Giai đoạn 6 — GPU và tối ưu AI (sau)

**Công việc:** ưu tiên adapter tới CUDA/cuBLAS/cuDNN hoặc ROCm khi có nền tảng CI/toolchain phù hợp; định nghĩa device buffer, đồng bộ và lỗi; khảo sát ONNX Runtime/llama.cpp trước khi xây backend riêng. Serving như vLLM chủ yếu là tích hợp qua API/runtime tương thích, không nhất thiết gọi trực tiếp trong process NX.

**Nghiệm thu:** matrix hỗ trợ phần cứng/driver/runtime; test CPU fallback và lỗi thiết bị; xác định rõ copy/transfer/synchronization; benchmark end-to-end gồm transfer và inference. Chỉ bắt đầu backend riêng khi workload, người bảo trì và lợi ích đã được chứng minh.

### Giai đoạn 7 — Package manager, tooling, IDE, debugger (ưu tiên cao)

**Nền móng hiện tại:** CLI, VS Code syntax/run support và manifest metadata; manifest hiện chưa tải hoặc resolve dependency.

**Công việc:** hoàn thiện resolver/lockfile sau khi chính sách registry và integrity được xác định; formatter/linter; language server cho diagnostics/completion/navigation; debugger có source map từ NX tới native; trải nghiệm build/test thống nhất.

**Nghiệm thu:** cài lại dự án từ manifest + lockfile trong môi trường sạch; không chạy script của dependency trong validate; test VS Code extension; breakpoint/stack trace ánh xạ về dòng NX; CLI báo lỗi ổn định và có test trên Windows/Linux/macOS được hỗ trợ.

## Thứ tự triển khai đề xuất

1. Chốt đặc tả và test corpus cho cú pháp, type, module và ABI đang tồn tại.
2. Tách frontend khỏi backend bằng NX-IR verifier, sau đó mới thêm pass tối ưu.
3. Ổn định hợp đồng ownership/ABI và đo chi phí Python/C/C++ boundary.
4. Thiết kế ndarray/tensor dựa trên lifetime và stride thực tế; không mở rộng Python bridge thành đối tượng tùy ý trước khi có mô hình ownership.
5. Bổ sung kiểm thử hiệu năng, race và portability trước khi mở rộng SIMD/GPU.
6. Hoàn thiện package/tooling dựa trên định dạng và quy trình build đã ổn định.

## Phạm vi AI ban đầu

NX nên điều phối pipeline, tiền/hậu xử lý và phần tính toán native phù hợp; Python có thể gọi PyTorch/Transformers khi API bridge hỗ trợ loại dữ liệu cần thiết. Phiên bản hiện tại có thể chuyển một `f64` span vào callable Python và nhận scalar/string/buffer theo các giới hạn đã nêu. Cú pháp như `model.generate("prompt")` **chưa được bridge hiện tại hỗ trợ**: nó cần truyền string/đối tượng Python và thiết kế ownership tương ứng. Không nên đưa ví dụ đó thành mã NX được quảng bá là chạy được cho tới khi API và test thực sự có mặt.

## Ngoài phạm vi cam kết

- Thay thế các framework AI hoặc hỗ trợ mọi API Python/C++.
- Khẳng định hiệu năng gần C/C++ cho mọi chương trình.
- GPU backend, zero-copy tổng quát hoặc ABI tương thích mọi compiler khi chưa có kiểm thử.
- Đánh dấu toàn bộ lộ trình là hoàn tất chỉ bằng tài liệu hay prototype.
