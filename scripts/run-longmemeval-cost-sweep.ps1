[CmdletBinding()]
param(
    [string]$CollectionRoot = ".local-tdai/longmemeval-collection",
    [string]$SourceBatchName = "oracle-v1",
    [string]$SelectionFile = ".local-tdai/longmemeval-collection/selections/oracle-stratified-v1.json",
    [int]$Limit = 200,
    [string]$CostCoefficients = "0,0.1,0.3,1.0",
    [double]$CostNormalizerTokens = 100.0,
    [string]$Seeds = "7,17,29",
    [int]$CqlEpochs = 100,
    [string]$PolicyPrefix = "oracle-v1-n200-injected-cost"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$collection = [IO.Path]::GetFullPath((Join-Path $repoRoot $CollectionRoot))
$source = Join-Path $collection "runs/$SourceBatchName"
$selection = [IO.Path]::GetFullPath((Join-Path $repoRoot $SelectionFile))
if (-not (Test-Path -LiteralPath (Join-Path $source "dataset-manifest.json"))) {
    throw "Source collection not found: $source"
}
if (-not (Test-Path -LiteralPath $selection)) {
    throw "Selection file not found: $selection"
}
if ($Limit -le 0) { throw "Limit must be positive" }
if ($CostNormalizerTokens -le 0) { throw "CostNormalizerTokens must be positive" }

$coefficients = $CostCoefficients.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) | ForEach-Object {
    $text = $_.Trim()
    $value = 0.0
    if (-not [double]::TryParse(
        $text,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$value
    ) -or $value -lt 0) {
        throw "Invalid non-negative cost coefficient: $text"
    }
    [pscustomobject]@{ Text = $text; Value = $value }
}
if ($coefficients.Count -eq 0) { throw "At least one cost coefficient is required" }

foreach ($coefficient in $coefficients) {
    $coefficientText = $coefficient.Value.ToString("0.################", [System.Globalization.CultureInfo]::InvariantCulture)
    $tag = $coefficientText.Replace(".", "p")
    $datasetName = "$PolicyPrefix-lambda$tag"
    $dataset = Join-Path $collection "training/$datasetName"

    python (Join-Path $PSScriptRoot "export-longmemeval-training.py") `
        --collection-root $source `
        --output-dir $dataset `
        --cost-coefficient $coefficientText `
        --cost-normalizer-tokens $CostNormalizerTokens `
        --cost-measure injected-l1-tokens `
        --split-seed longmemeval-v1 `
        --question-ids-file $selection `
        --limit $Limit
    if ($LASTEXITCODE -ne 0) { throw "Training export failed for lambda=$coefficientText" }

    & (Join-Path $PSScriptRoot "run-longmemeval-training-smoke.ps1") `
        -CollectionRoot $CollectionRoot `
        -BatchName $datasetName `
        -PolicyTag $datasetName `
        -Seeds $Seeds `
        -CqlEpochs $CqlEpochs
    if ($LASTEXITCODE -ne 0) { throw "Policy sweep failed for lambda=$coefficientText" }
}
