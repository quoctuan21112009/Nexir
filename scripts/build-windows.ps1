param(
    [switch]$SkipExtension
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$venvPython = Join-Path $root '.build-env\Scripts\python.exe'
$release = Join-Path $root 'release'
$work = Join-Path $root 'build\pyinstaller'
New-Item -ItemType Directory -Force -Path $release, $work | Out-Null

if (-not (Test-Path -LiteralPath $venvPython)) {
    & py -3 -m venv (Join-Path $root '.build-env')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python build environment.' }
}
& $venvPython -m pip install --disable-pip-version-check 'pyinstaller>=6.15,<7' $root
if ($LASTEXITCODE -ne 0) { throw 'Could not install build dependencies.' }

& $venvPython -m PyInstaller --noconfirm --clean --onefile --console --name nx `
    --icon (Join-Path $root 'nexir\assets\nx-icon.ico') `
    --collect-data nexir `
    --add-data "$(Join-Path $root 'editors\vscode');editors\vscode" `
    --distpath $release --workpath $work --specpath $work `
    --paths $root (Join-Path $root 'scripts\nx_entry.py')
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
& (Join-Path $release 'nx.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'The generated nx.exe did not start.' }

if (-not $SkipExtension) {
    $extension = Join-Path $root 'editors\vscode'
    Push-Location $extension
    try {
        & npx --yes '@vscode/vsce' package --no-dependencies --out (Join-Path $release 'nexir-language.vsix')
        if ($LASTEXITCODE -ne 0) { throw 'VSIX build failed.' }
    } finally {
        Pop-Location
    }
}
& $venvPython (Join-Path $root 'scripts\package_release.py')
if ($LASTEXITCODE -ne 0) { throw 'Could not create the release archive.' }
Write-Host "Built $(Join-Path $release 'nx.exe')"
if (-not $SkipExtension) { Write-Host "Built $(Join-Path $release 'nexir-language.vsix')" }
