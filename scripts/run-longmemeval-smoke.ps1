[CmdletBinding()]
param(
    [string]$CollectionRoot = ".local-tdai/longmemeval-collection",
    [string]$InstanceName = "longmemeval",
    [int]$CorePort = 8422,
    [int]$ProxyPort = 8098,
    [int]$Limit = 1
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$collection = [IO.Path]::GetFullPath((Join-Path $repoRoot $CollectionRoot))
pwsh -File (Join-Path $PSScriptRoot "prepare-longmemeval.ps1") -Root $collection

$runtime = Join-Path $repoRoot ".local-tdai/$InstanceName/runtime.json"
if (-not (Test-Path -LiteralPath $runtime)) {
    pwsh -File (Join-Path $PSScriptRoot "start-local-tdai.ps1") `
        -InstanceName $InstanceName -CorePort $CorePort -ProxyPort $ProxyPort `
        -L1IdleTimeoutSeconds 2 -L2DelayAfterL1Seconds 2 `
        -L2MinIntervalSeconds 5 -L2MaxIntervalSeconds 30
}

$batch = Join-Path $collection "runs/oracle-smoke-v1"
pwsh -File (Join-Path $PSScriptRoot "run-longmemeval.ps1") `
    -Data (Join-Path $collection "source/longmemeval_oracle.json") `
    -OutputRoot $batch -Runtime $runtime -Limit $Limit

python (Join-Path $PSScriptRoot "export-longmemeval-training.py") `
    --collection-root $batch `
    --output-dir (Join-Path $collection "training/oracle-smoke-v1")
if ($LASTEXITCODE -ne 0) { throw "LongMemEval training export failed" }
