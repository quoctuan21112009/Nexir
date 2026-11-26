const vscode = require('vscode');
const path = require('path');

function activate(context) {
  const run = async (document = vscode.window.activeTextEditor?.document) => {
    if (!document || document.languageId !== 'nexir') {
      vscode.window.showWarningMessage('Open a .nx file before running Nexir.');
      return;
    }
    await document.save();
    const folder = vscode.workspace.getWorkspaceFolder(document.uri);
    if (!folder) {
      vscode.window.showErrorMessage('Open the Nexir file inside a workspace to run it.');
      return;
    }

    const configured = vscode.workspace.getConfiguration('nexir').get('runCommand', 'nx').trim();
    if (!configured) {
      vscode.window.showErrorMessage('Set nexir.runCommand to your Nexir CLI executable.');
      return;
    }
    // ProcessExecution keeps the source path as one argument; spaces in paths
    // do not become shell syntax. Configure a wrapper script if the CLI needs
    // multiple launcher arguments (for example py -3 -m nexir).
    const execution = new vscode.ProcessExecution(configured, ['run', document.uri.fsPath], {
      cwd: folder.uri.fsPath
    });
    const task = new vscode.Task(
      { type: 'nexir', file: document.uri.fsPath },
      folder,
      `Run ${path.basename(document.uri.fsPath)}`,
      'Nexir',
      execution,
      '$gcc'
    );
    task.presentationOptions = { reveal: vscode.TaskRevealKind.Always, focus: false, panel: vscode.TaskPanelKind.Dedicated, clear: false };
    await vscode.tasks.executeTask(task);
  };

  context.subscriptions.push(vscode.commands.registerCommand('nexir.runFile', () => run()));
  context.subscriptions.push(vscode.commands.registerCommand('nexir.enableFileIcons', async () => {
    await vscode.workspace.getConfiguration('workbench').update(
      'iconTheme', 'nexir-file-icons', vscode.ConfigurationTarget.Global
    );
    vscode.window.showInformationMessage('Nexir file icons are enabled in Explorer.');
  }));
  context.subscriptions.push(vscode.languages.registerCodeLensProvider(
    { language: 'nexir', scheme: 'file' },
    { provideCodeLenses(document) {
      return [new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), {
        title: '$(play) Run Nexir', command: 'nexir.runFile'
      })];
    } }
  ));
}

function deactivate() {}
module.exports = { activate, deactivate };
