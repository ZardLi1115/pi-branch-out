[CmdletBinding()]
param(
    [string]$CollectionRoot = ".local-tdai/longmemeval-collection",
    [string]$InstanceName = "longmemeval-chat",
    [int]$CorePort = 8423,
    [int]$ProxyPort = 8099,
    [string]$BatchName = "oracle-v1",
    [int]$Offset = 0,
    [int]$Limit = 20,
    [int]$MaxInfrastructureRestarts = 5,
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

function Wait-Docker([int]$TimeoutSeconds = 180) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $result = Invoke-DockerCommand -Arguments @("info") -TimeoutSeconds 10
        if ($result.Success) { return }
        Start-Sleep -Seconds 3
    }
    throw "Timed out waiting for Docker daemon"
}

function Invoke-DockerCommand([string[]]$Arguments, [int]$TimeoutSeconds = 30) {
    $job = Start-Job -ScriptBlock {
        param([string[]]$DockerArguments)
        $output = & docker @DockerArguments 2>&1 | Out-String
        [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = $output.Trim() }
    } -ArgumentList (, $Arguments)
    try {
        if (-not (Wait-Job -Job $job -Timeout $TimeoutSeconds)) {
            Stop-Job -Job $job
            return [pscustomobject]@{ Success = $false; ExitCode = -1; Output = "timed out" }
        }
        $value = Receive-Job -Job $job
        return [pscustomobject]@{
            Success = $value.ExitCode -eq 0
            ExitCode = $value.ExitCode
            Output = $value.Output
        }
    }
    finally {
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }
}

function Move-UnfrozenAttemptsToAudit([string]$Reason) {
    if (-not (Test-Path -LiteralPath $batch)) { return 0 }
    $itemsRoot = Join-Path $batch "items"
    if (-not (Test-Path -LiteralPath $itemsRoot)) { return 0 }
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
    $auditRoot = Join-Path $collection "outage-audit/$stamp-$Reason/items"
    $collectionFull = [IO.Path]::GetFullPath($collection)
    $itemsFull = [IO.Path]::GetFullPath($itemsRoot)
    $auditFull = [IO.Path]::GetFullPath($auditRoot)
    if (-not $itemsFull.StartsWith($collectionFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Items path escaped collection root"
    }
    if (-not $auditFull.StartsWith($collectionFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Audit path escaped collection root"
    }
    $incomplete = @(Get-ChildItem -LiteralPath $itemsFull -Directory | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $_.FullName "complete.json")) -and
        -not (Test-Path -LiteralPath (Join-Path $_.FullName "candidate-snapshot.json"))
    })
    if ($incomplete.Count -eq 0) { return 0 }
    New-Item -ItemType Directory -Force -Path $auditFull | Out-Null
    foreach ($item in $incomplete) {
        Move-Item -LiteralPath $item.FullName -Destination (Join-Path $auditFull $item.Name)
    }
    Write-Output "Archived $($incomplete.Count) unfrozen item attempt(s) to $auditFull"
    return $incomplete.Count
}

function Ensure-RuntimeHealth {
    Wait-Docker
    foreach ($container in @([string]$runtimeConfig.core_container, [string]$runtimeConfig.proxy_container)) {
        $updated = Invoke-DockerCommand -Arguments @("update", "--restart", "unless-stopped", $container)
        if (-not $updated.Success) { throw "Failed to set restart policy for ${container}: $($updated.Output)" }
    }
    if (-not (Test-Health $coreHealth)) {
        $started = Invoke-DockerCommand -Arguments @("start", [string]$runtimeConfig.core_container) -TimeoutSeconds 60
        if (-not $started.Success) { throw "Failed to restart $($runtimeConfig.core_container): $($started.Output)" }
        Wait-Health $coreHealth
    }
    if (-not (Test-Health $proxyHealth)) {
        $started = Invoke-DockerCommand -Arguments @("start", [string]$runtimeConfig.proxy_container) -TimeoutSeconds 60
        if (-not $started.Success) { throw "Failed to restart $($runtimeConfig.proxy_container): $($started.Output)" }
        Wait-Health $proxyHealth
    }
}

$runtimeConfig = Get-Content -Raw -Encoding utf8 $runtime | ConvertFrom-Json
$coreHealth = "$($runtimeConfig.TDAI_CORE_URL.TrimEnd('/'))/health"
$proxyHealth = "$($runtimeConfig.TDAI_PROXY_URL.TrimEnd('/'))/health"
$batch = Join-Path $collection "runs/$BatchName"
if (-not (Test-Health $coreHealth)) {
    $null = Move-UnfrozenAttemptsToAudit "preflight-core-down"
}
Ensure-RuntimeHealth

$selection = Join-Path $collection "selections/oracle-stratified-v1.json"
if (-not (Test-Path -LiteralPath $selection)) {
    python (Join-Path $PSScriptRoot "select-longmemeval.py") `
        --data (Join-Path $collection "source/longmemeval_oracle.json") `
        --output $selection
    if ($LASTEXITCODE -ne 0) { throw "LongMemEval selection generation failed" }
}

$collectionSucceeded = $false
for ($attempt = 0; $attempt -le $MaxInfrastructureRestarts; $attempt++) {
    Ensure-RuntimeHealth
    pwsh -File (Join-Path $PSScriptRoot "run-longmemeval.ps1") `
        -Data (Join-Path $collection "source/longmemeval_oracle.json") `
        -OutputRoot $batch -Runtime $runtime -Offset $Offset -Limit $Limit `
        -QuestionIdsFile $selection -OverlongPolicy $OverlongPolicy -ContinueOnError
    if ($LASTEXITCODE -eq 0) {
        $collectionSucceeded = $true
        break
    }
    if ($attempt -ge $MaxInfrastructureRestarts) { break }
    if (-not (Test-Health $coreHealth)) {
        $null = Move-UnfrozenAttemptsToAudit "core-down-attempt-$attempt"
    }
    Write-Output "Collection interrupted; recovering infrastructure (attempt $($attempt + 1)/$MaxInfrastructureRestarts)"
    Start-Sleep -Seconds 5
}
if (-not $collectionSucceeded) {
    throw "LongMemEval collection did not complete after $MaxInfrastructureRestarts infrastructure restart(s)"
}

python (Join-Path $PSScriptRoot "export-longmemeval-training.py") `
    --collection-root $batch `
    --output-dir (Join-Path $collection "training/$BatchName")
if ($LASTEXITCODE -ne 0) { throw "LongMemEval training export failed" }

python (Join-Path $PSScriptRoot "summarize-longmemeval.py") `
    --collection-root $batch `
    --output (Join-Path $batch "summary.json")
if ($LASTEXITCODE -ne 0) { throw "LongMemEval summary failed" }
