# Benchmark NEXIR, Python, Rust, C++

- Thời điểm UTC: 2026-11-26T11:50:40+00:00
- Hệ thống: Windows-10-10.0.19045-SP0; CPU: Intel64 Family 6 Model 126 Stepping 5, GenuineIntel
- Phiên bản: Python 3.10.0; rustc 1.94.0 (4a4ef493e 2026-03-02) (Rev2, Built by MSYS2 project); g++.exe (Rev13, Built by MSYS2 project) 15.2.0
- Tác vụ: 20.000.000 lần cập nhật `x = (x * 1664525 + 1013904223) % 2147483647` bằng số nguyên 64 bit.
- Kết quả của cả bốn: `2028551754`. Mỗi chương trình chạy nóng 1 lần rồi đo 5 lần bằng `perf_counter_ns` ngoài tiến trình.
- Thời gian dưới đây chỉ tính chạy chương trình, gồm thời gian khởi động tiến trình; không tính biên dịch. NEXIR/C++ dùng `-O2`; Rust dùng `-C opt-level=2 -C overflow-checks=off`.

| Ngôn ngữ | Trung vị (giây) | Các lần đo (giây) | Tốc độ so với Python |
|---|---:|---|---:|
| NEXIR | 0.1200 | 0.1225, 0.1206, 0.1149, 0.1200, 0.1160 | 30.21× |
| Python | 3.6254 | 4.2842, 3.6254, 3.5994, 3.7053, 3.5268 | 1.00× |
| Rust | 0.1124 | 0.1174, 0.1100, 0.1083, 0.1125, 0.1124 | 32.26× |
| C++ | 0.1220 | 0.1314, 0.1219, 0.1152, 0.1220, 0.1234 | 29.73× |

Đây là phép đo một vòng lặp số học cụ thể trên một máy, không đại diện hiệu năng chung. NEXIR hiện sinh C++17 rồi dùng G++, nên kết quả chịu ảnh hưởng trực tiếp từ backend C++.
