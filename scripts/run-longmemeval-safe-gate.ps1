[CmdletBinding()]
param(
    [string]$CollectionRoot = ".local-tdai/longmemeval-collection",
    [string]$DatasetName = "oracle-v1-n500-positional-lambda0",
    [string]$PolicyPrefix = "oracle-v1-n500-positional-alpha",
    [string]$OutputName = "oracle-v1-n500-positional-safe-gate.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$collection = [IO.Path]::GetFullPath((Join-Path $repoRoot $CollectionRoot))
$dataset = Join-Path $collection "training/$DatasetName"
$output = Join-Path $collection "policies/$OutputName"

python (Join-Path $PSScriptRoot "evaluate-longmemeval-safe-gate.py") `
    --dataset-dir $dataset `
    --policy-group "alpha0=$(Join-Path $collection "policies/$($PolicyPrefix)0-bestdev-seed*")" `
    --policy-group "alpha0p1=$(Join-Path $collection "policies/$($PolicyPrefix)0p1-bestdev-seed*")" `
    --policy-group "alpha1=$(Join-Path $collection "policies/$($PolicyPrefix)1-bestdev-seed*")" `
    --output $output
if ($LASTEXITCODE -ne 0) { throw "Safe gate evaluation failed" }
