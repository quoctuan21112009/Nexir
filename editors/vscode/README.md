# Nexir VS Code / Antigravity Language & Icon Extension (2.0.0)

Provides:
- `.nx` file icon recognition in Explorer folder trees (both via language icon metadata on default themes and the dedicated `Nexir File Icons` icon theme for `.nx`, `main.nx`, `nx.toml`, `nx.lock`).
- Syntax highlighting for NX 2.0 (`class`, `def`, `async`/`await`, `enum`, `match`/`case`, `with`, `i8..i128`, `u8..u128`, `@cap`, etc.).
- One-click Run and Debug buttons in the editor title bar, CodeLens, and Explorer folder context menu (`right-click .nx file`).
- Windows Explorer `.nx` folder icon registration via `nx icon register` (or Command Palette `Nexir: Register .nx Icon in Windows Explorer`).

## Quick Install

```powershell
py -3 -m nexir icon install-editor
py -3 -m nexir icon register
```

Or run `Nexir: Enable Nexir File Icons` from the editor Command Palette to activate the `Nexir File Icons` theme.
