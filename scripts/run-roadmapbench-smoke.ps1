[CmdletBinding()]
param(
    [string]$Task = "D:\TDAI\RoadmapBench\harbor_tasks\vanilla\glz-3.0.0-roadmap",
    [string]$JobsDir = "D:\TDAI\pi-branch-out\.local-tdai\smoke\glz-3.0.0-roadmap",
    [string]$RuntimeConfig = "D:\TDAI\pi-branch-out\.local-tdai\natural\runtime.json",
    [string]$RuntimeArchive = "D:\TDAI\pi-branch-out\runtime\pi-runtime-linux-amd64.tar.gz",
    [string]$PiExtension = "D:\TDAI\TencentDB-Agent-Memory\MemoryCore\pi-plugin\index.ts"
)

$ErrorActionPreference = "Stop"
foreach ($path in @($Task, $RuntimeConfig, $RuntimeArchive, $PiExtension)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}
if (Test-Path -LiteralPath $JobsDir) {
    $JobsDir = Join-Path $JobsDir ("retry-" + [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ"))
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
try {
    foreach ($key in $keys) {
        [Environment]::SetEnvironmentVariable($key, [string]$runtime.$key, "Process")
    }
    $env:CUSTOM_API_KEY = [string]$auth.OPENAI_API_KEY
    $env:OPENAI_API_KEY = [string]$auth.OPENAI_API_KEY
    $env:OPENAI_BASE_URL = $urlMatch.Groups[1].Value
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:NO_COLOR = "1"
    $env:TERM = "dumb"
    & pi-branch-out natural `
        --task $Task `
        --jobs-dir $JobsDir `
        --model "tdai/gpt-5.6-luna" `
        --harbor-bin harbor `
        --pi-thinking medium `
        --checkpoint-boundary model-call `
        --max-checkpoints 2 `
        --min-checkpoint-gap 10 `
        --sample-probability 0.1 `
        --max-candidate-probes 8 `
        --sampling-batch sampling-smoke-v1 `
        --pi-runtime-archive $RuntimeArchive `
        --pi-extension $PiExtension
    exit $LASTEXITCODE
}
finally {
    foreach ($key in $keys + @(
        "CUSTOM_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "PYTHONUTF8", "PYTHONIOENCODING", "NO_COLOR", "TERM"
    )) {
        [Environment]::SetEnvironmentVariable($key, $null, "Process")
    }
}
