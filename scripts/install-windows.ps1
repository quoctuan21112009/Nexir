param(
    [string]$Exe = (Join-Path $PSScriptRoot '..\release\nx.exe'),
    [string]$Vsix = (Join-Path $PSScriptRoot '..\release\nexir-language.vsix'),
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\Nexir\bin'),
    [switch]$InstallCompiler,
    [switch]$SkipEditors
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Exe)) { throw "nx.exe not found: $Exe. Run scripts/build-windows.ps1 first." }
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$destination = Join-Path $InstallDir 'nx.exe'
Copy-Item -LiteralPath $Exe -Destination $destination -Force

# Add the CLI directory to user PATH without replacing other entries.
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$parts = @($userPath -split ';' | Where-Object { $_ })
if (-not ($parts | Where-Object { [string]::Equals($_.TrimEnd('\'), $InstallDir.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase) })) {
    [Environment]::SetEnvironmentVariable('Path', (($parts + $InstallDir) -join ';'), 'User')
}
$env:Path = "$InstallDir;$env:Path"

$compiler = 'C:\msys64\ucrt64\bin\g++.exe'
if ($InstallCompiler -and -not (Test-Path -LiteralPath $compiler)) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) { throw 'winget is required to install MSYS2 automatically. Install MSYS2 from msys2.org.' }
    & winget install --id MSYS2.MSYS2 --exact --source winget --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw 'MSYS2 installation failed.' }
    $bash = 'C:\msys64\usr\bin\bash.exe'
    if (-not (Test-Path -LiteralPath $bash)) { throw 'MSYS2 was installed outside C:\msys64; install UCRT64 GCC in that location manually.' }
    & $bash -lc 'pacman -Sy --needed --noconfirm mingw-w64-ucrt-x86_64-gcc'
    if ($LASTEXITCODE -ne 0) { throw 'GCC installation failed; update MSYS2 and retry pacman.' }
}
if ((Test-Path -LiteralPath $compiler) -and -not [Environment]::GetEnvironmentVariable('NEXIR_CXX', 'User')) {
    [Environment]::SetEnvironmentVariable('NEXIR_CXX', $compiler, 'User')
    $env:NEXIR_CXX = $compiler
}

if (-not $SkipEditors -and (Test-Path -LiteralPath $Vsix)) {
    foreach ($editor in @('code', 'antigravity')) {
        $command = Get-Command $editor -ErrorAction SilentlyContinue
        $editorPath = if ($command) { $command.Source } else { $null }
        if (-not $editorPath -and $editor -eq 'code') {
            foreach ($candidate in @(
                (Join-Path $env:LOCALAPPDATA 'Programs\Microsoft VS Code\bin\code.cmd'),
                (Join-Path $env:ProgramFiles 'Microsoft VS Code\bin\code.cmd')
            )) {
                if (Test-Path -LiteralPath $candidate) { $editorPath = $candidate; break }
            }
        }
        if (-not $editorPath -and $editor -eq 'antigravity') {
            $alternate = Get-Command antigravity-ide -ErrorAction SilentlyContinue
            if ($alternate) { $editorPath = $alternate.Source }
            if (-not $editorPath) {
                $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Antigravity IDE\bin\antigravity-ide.cmd'
                if (Test-Path -LiteralPath $candidate) { $editorPath = $candidate }
            }
        }
        if ($editorPath) {
            & $editorPath --install-extension (Resolve-Path -LiteralPath $Vsix).Path --force
            if ($LASTEXITCODE -ne 0) { Write-Warning "$editor did not install the VSIX; install it through the editor's Extensions view." }
        } else {
            Write-Host "$editor command not found; install $Vsix through the editor's Extensions view if needed."
        }
    }
}
Write-Host "Installed $destination"
& $destination icon register
& $destination icon install-editor
Write-Host 'Open a new terminal, then run: nx doctor'
