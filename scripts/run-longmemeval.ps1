[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Data,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$Runtime = ".local-tdai/longmemeval/runtime.json",
    [int]$Offset = 0,
    [int]$Limit = 0,
    [string]$QuestionId = "",
    [string]$Ratios = "0,0.2,0.4,0.6,0.8,1",
    [ValidateSet("error", "skip-instance", "truncate")][string]$OverlongPolicy = "error",
    [string]$AnswerModel = "gpt-5.6-luna",
    [string]$JudgeModel = "gpt-5.6-luna",
    [switch]$RequireActionDiversity
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$codexRoot = Join-Path $env:USERPROFILE ".codex"
$configText = Get-Content -Raw -Encoding utf8 (Join-Path $codexRoot "config.toml")
$providerMatch = [regex]::Match($configText, '(?ms)^\[model_providers\.custom\]\s*(.*?)(?=^\[|\z)')
$urlMatch = [regex]::Match($providerMatch.Groups[1].Value, '(?m)^base_url\s*=\s*"([^"]+)"')
$auth = Get-Content -Raw -Encoding utf8 (Join-Path $codexRoot "auth.json") | ConvertFrom-Json
if (-not $urlMatch.Success -or -not $auth.OPENAI_API_KEY) {
    throw "Codex custom provider configuration is incomplete"
}

$previousBaseUrl = $env:OPENAI_BASE_URL
$previousApiKey = $env:OPENAI_API_KEY
try {
    $env:OPENAI_BASE_URL = $urlMatch.Groups[1].Value
    $env:OPENAI_API_KEY = [string]$auth.OPENAI_API_KEY
    $arguments = @(
        "tsx", "scripts/collect-longmemeval.ts",
        "--data", (Resolve-Path -LiteralPath $Data).Path,
        "--output-root", [IO.Path]::GetFullPath($OutputRoot),
        "--runtime", (Resolve-Path -LiteralPath $Runtime).Path,
        "--offset", [string]$Offset,
        "--ratios", $Ratios,
        "--overlong-policy", $OverlongPolicy,
        "--answer-model", $AnswerModel,
        "--judge-model", $JudgeModel
    )
    if ($Limit -gt 0) { $arguments += @("--limit", [string]$Limit) }
    if ($QuestionId) { $arguments += @("--question-id", $QuestionId) }
    if ($RequireActionDiversity) { $arguments += "--require-action-diversity" }
    & npx @arguments
    if ($LASTEXITCODE -ne 0) { throw "LongMemEval collection failed with exit code $LASTEXITCODE" }
}
finally {
    $env:OPENAI_BASE_URL = $previousBaseUrl
    $env:OPENAI_API_KEY = $previousApiKey
}
