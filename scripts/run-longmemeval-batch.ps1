[CmdletBinding()]
param(
    [string]$CollectionRoot = ".local-tdai/longmemeval-collection",
    [string]$InstanceName = "longmemeval-chat",
    [int]$CorePort = 8423,
    [int]$ProxyPort = 8099,
    [string]$BatchName = "oracle-v1",
    [int]$Offset = 0,
    [int]$Limit = 20,
    [ValidateSet("error", "skip-instance", "truncate")][string]$OverlongPolicy = "error"
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

function Test-Health([string]$Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec 5
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 300
    }
    catch { return $false }
}

function Wait-Health([string]$Url, [int]$TimeoutSeconds = 120) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Health $Url) { return }
        Start-Sleep -Seconds 2
    }
    throw "Timed out waiting for $Url"
}

$runtimeConfig = Get-Content -Raw -Encoding utf8 $runtime | ConvertFrom-Json
$coreHealth = "$($runtimeConfig.TDAI_CORE_URL.TrimEnd('/'))/health"
$proxyHealth = "$($runtimeConfig.TDAI_PROXY_URL.TrimEnd('/'))/health"
if (-not (Test-Health $coreHealth)) {
    & docker start ([string]$runtimeConfig.core_container) | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to restart $($runtimeConfig.core_container)" }
    Wait-Health $coreHealth
}
if (-not (Test-Health $proxyHealth)) {
    & docker start ([string]$runtimeConfig.proxy_container) | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to restart $($runtimeConfig.proxy_container)" }
    Wait-Health $proxyHealth
}

$selection = Join-Path $collection "selections/oracle-stratified-v1.json"
if (-not (Test-Path -LiteralPath $selection)) {
    python (Join-Path $PSScriptRoot "select-longmemeval.py") `
        --data (Join-Path $collection "source/longmemeval_oracle.json") `
        --output $selection
    if ($LASTEXITCODE -ne 0) { throw "LongMemEval selection generation failed" }
}

$batch = Join-Path $collection "runs/$BatchName"
pwsh -File (Join-Path $PSScriptRoot "run-longmemeval.ps1") `
    -Data (Join-Path $collection "source/longmemeval_oracle.json") `
    -OutputRoot $batch -Runtime $runtime -Offset $Offset -Limit $Limit `
    -QuestionIdsFile $selection -OverlongPolicy $OverlongPolicy -ContinueOnError

python (Join-Path $PSScriptRoot "export-longmemeval-training.py") `
    --collection-root $batch `
    --output-dir (Join-Path $collection "training/$BatchName")
if ($LASTEXITCODE -ne 0) { throw "LongMemEval training export failed" }

python (Join-Path $PSScriptRoot "summarize-longmemeval.py") `
    --collection-root $batch `
    --output (Join-Path $batch "summary.json")
if ($LASTEXITCODE -ne 0) { throw "LongMemEval summary failed" }
