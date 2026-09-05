[CmdletBinding()]
param(
    [ValidateSet("pilot-v1", "remaining-v1")]
    [string]$BatchName = "pilot-v1",
    [string]$OutputRoot = "",
    [string]$Dataset = "D:\TDAI\RoadmapBench\harbor_tasks\vanilla",
    [string]$RuntimeConfig = "D:\TDAI\pi-branch-out\.local-tdai\natural\runtime.json",
    [string]$RuntimeArchive = "D:\TDAI\pi-branch-out\runtime\pi-runtime-linux-amd64.tar.gz",
    [string]$PiExtension = "D:\TDAI\TencentDB-Agent-Memory\MemoryCore\pi-plugin\index.ts"
)

$ErrorActionPreference = "Stop"
$collectionRoot = "D:\TDAI\pi-branch-out\.local-tdai\roadmapbench-collection"
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $collectionRoot $BatchName
}
if ($BatchName -eq "pilot-v1") {
    $taskNames = @(
        "fal-2.0.0-roadmap", "fbr-2.37.0-roadmap", "glz-6.3.0-roadmap",
        "dsl-2.1.0-roadmap", "opt-4.0.0-roadmap"
    )
    $maxWallSeconds = 43200
    $maxModelCalls = 500
    $maxTokenUnits = 20000000
}
else {
    $taskNames = @(
        "fyn-2.2.0-roadmap", "glz-3.0.0-roadmap", "glz-7.0.0-roadmap",
        "ktx-0.12.0-roadmap", "mko-7.0.0-roadmap", "plr-1.6.0-roadmap",
        "plr-1.18.0-roadmap", "prm-5.19.0-roadmap", "prm-6.13.0-roadmap",
        "pyg-2.0.0-roadmap", "slt-1.14.0-roadmap"
    )
    $maxWallSeconds = 86400
    $maxModelCalls = 800
    $maxTokenUnits = 40000000
}
$collector = Join-Path $PSScriptRoot "collect-roadmapbench.py"
foreach ($path in @($Dataset, $RuntimeConfig, $RuntimeArchive, $PiExtension, $collector)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

$runtime = Get-Content -Raw -Encoding utf8 $RuntimeConfig | ConvertFrom-Json
$codexRoot = Join-Path $env:USERPROFILE ".codex"
$auth = Get-Content -Raw -Encoding utf8 (Join-Path $codexRoot "auth.json") | ConvertFrom-Json
$configText = Get-Content -Raw -Encoding utf8 (Join-Path $codexRoot "config.toml")
$providerMatch = [regex]::Match($configText, '(?ms)^\[model_providers\.custom\]\s*(.*?)(?=^\[|\z)')
$urlMatch = [regex]::Match($providerMatch.Groups[1].Value, '(?m)^base_url\s*=\s*"([^"]+)"')
if (-not $urlMatch.Success -or -not $auth.OPENAI_API_KEY) {
    throw "Codex custom provider configuration is incomplete"
}

$keys = @(
    "TDAI_PROXY_URL", "TDAI_SPACE_ID", "TDAI_AGENT_SOURCE", "TDAI_WIRE_API",
    "TDAI_TEAM_ID", "TDAI_AGENT_ID", "TDAI_TASK_ID", "TDAI_USER_KEY", "TDAI_MODEL"
)
$savedEnvironment = @{}
try {
    foreach ($key in $keys) {
        $savedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, [string]$runtime.$key, "Process")
    }
    foreach ($key in @(
        "CUSTOM_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "PYTHONUTF8", "PYTHONIOENCODING", "NO_COLOR", "TERM"
    )) {
        $savedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
    }
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "Python 3 executable was not found on PATH"
    }
    $env:CUSTOM_API_KEY = [string]$auth.OPENAI_API_KEY
    $env:OPENAI_API_KEY = [string]$auth.OPENAI_API_KEY
    $env:OPENAI_BASE_URL = $urlMatch.Groups[1].Value
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:NO_COLOR = "1"
    $env:TERM = "dumb"

    $collectorArgs = @(
        "-X", "utf8", $collector,
        "--dataset", $Dataset,
        "--output-root", $OutputRoot,
        "--runtime-archive", $RuntimeArchive,
        "--pi-extension", $PiExtension,
        "--model", "tdai/gpt-5.6-luna",
        "--thinking", "medium",
        "--max-tasks", [string]$taskNames.Count,
        "--max-attempts", "2",
        "--retry-delay-seconds", "5",
        "--min-free-gib", "5",
        "--max-wall-seconds", [string]$maxWallSeconds,
        "--max-total-model-calls", [string]$maxModelCalls,
        "--max-total-token-units", [string]$maxTokenUnits,
        "--max-checkpoints", "2",
        "--min-checkpoint-gap", "10",
        "--sample-probability", "0.1",
        "--max-candidate-probes", "8",
        "--sampling-batch", "roadmapbench-$BatchName"
    )
    foreach ($taskName in $taskNames) {
        $collectorArgs += @("--include-task-name", $taskName)
    }
    & python @collectorArgs
    exit $LASTEXITCODE
}
finally {
    foreach ($key in $keys + @(
        "CUSTOM_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "PYTHONUTF8", "PYTHONIOENCODING", "NO_COLOR", "TERM"
    )) {
        [Environment]::SetEnvironmentVariable($key, $savedEnvironment[$key], "Process")
    }
}
