[CmdletBinding()]
param(
    [string]$CollectionRoot = ".local-tdai/longmemeval-collection",
    [string]$InstanceName = "longmemeval-chat",
    [int]$CorePort = 8423,
    [int]$ProxyPort = 8099,
    [string]$BatchName = "oracle-smoke-chat-v1",
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
        -PromptMode chat `
        -L1IdleTimeoutSeconds 2 -L2DelayAfterL1Seconds 2 `
        -L2MinIntervalSeconds 5 -L2MaxIntervalSeconds 30
}

$batch = Join-Path $collection "runs/$BatchName"
pwsh -File (Join-Path $PSScriptRoot "run-longmemeval.ps1") `
    -Data (Join-Path $collection "source/longmemeval_oracle.json") `
    -OutputRoot $batch -Runtime $runtime -Limit $Limit -RequireActionDiversity

python (Join-Path $PSScriptRoot "export-longmemeval-training.py") `
    --collection-root $batch `
    --output-dir (Join-Path $collection "training/$BatchName")
if ($LASTEXITCODE -ne 0) { throw "LongMemEval training export failed" }
