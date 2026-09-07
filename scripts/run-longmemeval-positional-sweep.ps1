[CmdletBinding()]
param(
    [string]$CollectionRoot = ".local-tdai/longmemeval-collection",
    [string]$SourceBatchName = "oracle-v1",
    [string]$SelectionFile = ".local-tdai/longmemeval-collection/selections/oracle-stratified-v1.json",
    [int]$Limit = 500,
    [string]$CqlAlphas = "0,0.1,1.0",
    [string]$Seeds = "7,17,29",
    [int]$CqlEpochs = 100,
    [string]$DatasetName = "oracle-v1-n500-positional-lambda0",
    [string]$PolicyPrefix = "oracle-v1-n500-positional-alpha"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$collection = [IO.Path]::GetFullPath((Join-Path $repoRoot $CollectionRoot))
$source = Join-Path $collection "runs/$SourceBatchName"
$dataset = Join-Path $collection "training/$DatasetName"
$selection = [IO.Path]::GetFullPath((Join-Path $repoRoot $SelectionFile))

python (Join-Path $PSScriptRoot "export-longmemeval-training.py") `
    --collection-root $source `
    --output-dir $dataset `
    --cost-coefficient 0 `
    --cost-normalizer-tokens 100 `
    --cost-measure injected-l1-tokens `
    --split-seed longmemeval-v1 `
    --question-ids-file $selection `
    --limit $Limit `
    --include-action-features
if ($LASTEXITCODE -ne 0) { throw "Positional training export failed" }

& (Join-Path $PSScriptRoot "run-longmemeval-cql-alpha-sweep.ps1") `
    -CollectionRoot $CollectionRoot `
    -BatchName $DatasetName `
    -CqlAlphas $CqlAlphas `
    -Seeds $Seeds `
    -CqlEpochs $CqlEpochs `
    -PolicyPrefix $PolicyPrefix `
    -FeatureVersion "visible-state-hash-v5-positional-l1-actions"
if ($LASTEXITCODE -ne 0) { throw "Positional CQL alpha sweep failed" }
