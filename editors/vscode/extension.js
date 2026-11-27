const vscode = require('vscode');
const path = require('path');

function activate(context) {
  const executeNexir = async (subcommand, targetUri) => {
    let document = null;
    let filePath = targetUri && targetUri.fsPath ? targetUri.fsPath : null;
    if (!filePath) {
      document = vscode.window.activeTextEditor?.document;
      if (!document || document.languageId !== 'nexir') {
        vscode.window.showWarningMessage('Open or select a .nx file first.');
        return;
      }
      await document.save();
      filePath = document.uri.fsPath;
    } else if (path.extname(filePath).toLowerCase() !== '.nx') {
      vscode.window.showWarningMessage('Selected file is not a .nx file.');
      return;
    } else {
      const openDoc = vscode.workspace.textDocuments.find(d => d.uri.fsPath === filePath);
      if (openDoc && openDoc.isDirty) await openDoc.save();
    }

    const folder = vscode.workspace.getWorkspaceFolder(vscode.Uri.file(filePath));
    const cwd = folder ? folder.uri.fsPath : path.dirname(filePath);
    const configured = vscode.workspace.getConfiguration('nexir').get('runCommand', 'nx').trim();
    if (!configured) {
      vscode.window.showErrorMessage('Set nexir.runCommand to your Nexir CLI executable.');
      return;
    }

    const execution = new vscode.ProcessExecution(configured, [subcommand, filePath], { cwd });
    const task = new vscode.Task(
      { type: 'nexir', file: filePath, subcommand },
      folder || vscode.TaskScope.Workspace,
      `${subcommand === 'debug' ? 'Debug' : 'Run'} ${path.basename(filePath)}`,
      'Nexir',
      execution,
      '$gcc'
    );
    task.presentationOptions = { reveal: vscode.TaskRevealKind.Always, focus: false, panel: vscode.TaskPanelKind.Dedicated, clear: false };
    await vscode.tasks.executeTask(task);
  };

  context.subscriptions.push(vscode.commands.registerCommand('nexir.runFile', (uri) => executeNexir('run', uri)));
  context.subscriptions.push(vscode.commands.registerCommand('nexir.debugFile', (uri) => executeNexir('debug', uri)));
  context.subscriptions.push(vscode.commands.registerCommand('nexir.enableFileIcons', async () => {
    await vscode.workspace.getConfiguration('workbench').update(
      'iconTheme', 'nexir-file-icons', vscode.ConfigurationTarget.Global
    );
    vscode.window.showInformationMessage('Nexir file icons are enabled in Explorer.');
  }));
  context.subscriptions.push(vscode.commands.registerCommand('nexir.registerWindowsIcon', async () => {
    const configured = vscode.workspace.getConfiguration('nexir').get('runCommand', 'nx').trim() || 'nx';
    const execution = new vscode.ProcessExecution(configured, ['icon', 'register']);
    await vscode.tasks.executeTask(new vscode.Task(
      { type: 'nexir', subcommand: 'icon' },
      vscode.TaskScope.Workspace,
      'Register .nx Windows Explorer Icon',
      'Nexir',
      execution
    ));
  }));
  context.subscriptions.push(vscode.languages.registerCodeLensProvider(
    { language: 'nexir', scheme: 'file' },
    { provideCodeLenses() {
      return [
        new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), { title: '$(play) Run Nexir', command: 'nexir.runFile' }),
        new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), { title: '$(debug-alt) Debug Nexir', command: 'nexir.debugFile' })
      ];
    } }
  ));
}

function deactivate() {}
module.exports = { activate, deactivate };
