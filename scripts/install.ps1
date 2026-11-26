param(
    [string]$Wheel = (Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot '..\release') -Filter 'nexir-*.whl' | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName)
)
$ErrorActionPreference = 'Stop'
if (-not $Wheel -or -not (Test-Path -LiteralPath $Wheel)) {
    throw 'NEXIR wheel not found. Build it with: py -3 -m pip wheel . --no-deps -w release'
}
& py -3 -m pip install --user --upgrade $Wheel
if ($LASTEXITCODE -ne 0) { throw 'NEXIR installation failed' }
& py -3 -m nexir doctor
