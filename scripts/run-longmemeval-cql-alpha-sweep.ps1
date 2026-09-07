[CmdletBinding()]
param(
    [string]$CollectionRoot = ".local-tdai/longmemeval-collection",
    [string]$BatchName = "oracle-v1-n500-injected-cost-lambda0",
    [string]$CqlAlphas = "0,0.1,1.0",
    [string]$Seeds = "7,17,29",
    [int]$CqlEpochs = 100,
    [string]$PolicyPrefix = "oracle-v1-n500-alpha",
    [string]$FeatureVersion = "visible-state-hash-v4-memory-text"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$alphas = $CqlAlphas.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) | ForEach-Object {
    $text = $_.Trim()
    $value = 0.0
    if (-not [double]::TryParse(
        $text,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$value
    ) -or $value -lt 0) {
        throw "Invalid non-negative CQL alpha: $text"
    }
    $value
}
if ($alphas.Count -eq 0) { throw "At least one CQL alpha is required" }

foreach ($alpha in $alphas) {
    $alphaText = $alpha.ToString("0.################", [System.Globalization.CultureInfo]::InvariantCulture)
    $tag = $alphaText.Replace(".", "p")
    & (Join-Path $PSScriptRoot "run-longmemeval-training-smoke.ps1") `
        -CollectionRoot $CollectionRoot `
        -BatchName $BatchName `
        -PolicyTag "$PolicyPrefix$tag-bestdev" `
        -Seeds $Seeds `
        -CqlEpochs $CqlEpochs `
        -CqlAlpha $alpha `
        -SelectBestDev `
        -FeatureVersion $FeatureVersion
    if ($LASTEXITCODE -ne 0) { throw "CQL alpha sweep failed for alpha=$alphaText" }
}
