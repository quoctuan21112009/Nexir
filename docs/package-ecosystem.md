# Hệ sinh thái Package NX (`pkg.nx.dev` / `nx.toml` + `nx.lock`)

NEXIR 2.0 cung cấp hệ thống quản lý gói lấy cảm hứng từ `pkg.go.dev` với khóa phiên bản xác định (`nx.lock`), kiểm tra toàn vẹn SHA-256, kiểm toán quyền bảo mật (`audit`), registry cục bộ/cache và trình sinh tài liệu HTML/Markdown (`nx doc`).

## Lệnh CLI

```powershell
nx pkg nx init --path . --name my-lib
nx pkg nx validate --path .
nx pkg nx add nx-math "^0.1.0" --path .
nx pkg nx resolve --path .
nx pkg nx install --path .
nx pkg nx verify --path .
nx pkg nx audit --path .
nx pkg nx graph --path .
nx pkg nx publish --path .
nx pkg nx search math
nx doc --path . -o build/docs --format html [--serve --port 8080]
```

## Cấu trúc `nx.toml` và `nx.lock`

```toml
[package]
name = "my-app"
version = "0.1.0"
entry = "src/main.nx"
module = "nx.dev/local/my-app"

[dependencies]
nx-math = "^0.1.0"
```

- `resolve` và `install` ghi `nx.lock` kèm checksum SHA-256 và danh sách `@cap(...)` của từng dependency.
- `verify` phát hiện mọi thay đổi trái phép hoặc hỏng file trong `.nx/packages`.
- `audit` liệt kê và cảnh báo các dependency sử dụng quyền rủi ro cao (`ffi.*`, `process.exec`, `filesystem.delete`, `network.listen`).
- `validate`/`resolve`/`audit` không bao giờ thực thi mã tùy ý trong gói.
