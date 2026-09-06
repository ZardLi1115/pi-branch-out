[CmdletBinding()]
param(
    [string]$CollectionRoot = ".local-tdai/longmemeval-collection",
    [string]$BatchName = "oracle-v1",
    [string]$PolicyTag = "",
    [string]$Seeds = "7,17,29",
    [int]$CqlEpochs = 100
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$collection = [IO.Path]::GetFullPath((Join-Path $repoRoot $CollectionRoot))
$dataset = Join-Path $collection "training/$BatchName"
if (-not (Test-Path -LiteralPath (Join-Path $dataset "dataset-manifest.json"))) {
    throw "Training dataset not found: $dataset"
}
if (-not $PolicyTag) { $PolicyTag = $BatchName }

$parsedSeeds = $Seeds.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) | ForEach-Object {
    $value = 0
    if (-not [int]::TryParse($_.Trim(), [ref]$value)) { throw "Invalid seed: $_" }
    $value
}
if ($parsedSeeds.Count -eq 0) { throw "At least one seed is required" }

foreach ($seed in $parsedSeeds) {
    $policyRoot = Join-Path $collection "policies/$PolicyTag-seed$seed"
    pi-branch-out train-policy `
        --dataset-dir $dataset `
        --output-dir $policyRoot `
        --seed $seed `
        --cql-epochs $CqlEpochs
    if ($LASTEXITCODE -ne 0) { throw "Policy training failed for seed $seed" }

    python (Join-Path $PSScriptRoot "evaluate-longmemeval-policy.py") `
        --dataset-dir $dataset `
        --policy-dir $policyRoot `
        --output (Join-Path $policyRoot "evaluation.json")
    if ($LASTEXITCODE -ne 0) { throw "Policy evaluation failed for seed $seed" }
}
