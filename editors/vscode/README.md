# Nexir VS Code / Antigravity support

This extension adds `.nx` syntax coloring, a Nexir file icon theme, a Run button in
the editor title bar, and a `Run Nexir` CodeLens above the source. It uses VS
Code's task API and passes the active file as a separate process argument.

## Run from the extension source

1. Open this repository in VS Code or an extension-compatible editor.
2. Open `editors/vscode` as an extension development host (or use **Run Extension**
   from a VS Code extension-development launch configuration).
3. Set `nexir.runCommand` to the Nexir executable available on your PATH, then
   open a `.nx` file and click **Run Nexir**.
4. If you launch Nexir through `py -3 -m nexir`, make a small wrapper script and
   set `nexir.runCommand` to that script; the setting intentionally accepts one
   executable, not shell text.

To package it, run `npx @vscode/vsce package` in this directory and install the
resulting VSIX. The icon theme is named **Nexir File Icons**; select it in
Preferences → File Icon Theme.

Antigravity support depends on its version supporting VS Code extensions and
extension-host APIs. The extension does not edit editor internals or inject
commands into terminals.
