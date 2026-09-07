[CmdletBinding()]
param(
    [string]$Data = ".local-tdai/longmemeval-collection/source/longmemeval_oracle.json",
    [string]$OutputRoot = ".local-tdai/longmemeval-collection/runs/oracle-v201-agent-search",
    [string]$Runtime = ".local-tdai/longmemeval-v201-noinject/runtime.json",
    [string]$QuestionIdsFile = ".local-tdai/longmemeval-collection/selections/oracle-stratified-v1.json",
    [int]$Offset = 0,
    [int]$Limit = 1,
    [int]$MaxAgentTurns = 4,
    [int]$SearchLimit = 5,
    [string]$AgentModel = "gpt-5.6-luna",
    [string]$JudgeModel = "gpt-5.6-luna",
    [ValidateSet("error", "skip-instance", "truncate")][string]$OverlongPolicy = "error",
    [switch]$ContinueOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$dataPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $Data))
$outputPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputRoot))
$runtimePath = [IO.Path]::GetFullPath((Join-Path $repoRoot $Runtime))
$selectionPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $QuestionIdsFile))
$configText = Get-Content -Raw -Encoding UTF8 (Join-Path $env:USERPROFILE ".codex\config.toml")
$providerMatch = [regex]::Match($configText, '(?ms)^\[model_providers\.custom\]\s*(.*?)(?=^\[|\z)')
if (-not $providerMatch.Success) { throw "Codex custom provider section was not found" }
$urlMatch = [regex]::Match($providerMatch.Groups[1].Value, '(?m)^base_url\s*=\s*"([^"]+)"')
if (-not $urlMatch.Success) { throw "Codex custom provider base_url was not found" }
$auth = Get-Content -Raw -Encoding UTF8 (Join-Path $env:USERPROFILE ".codex\auth.json") | ConvertFrom-Json
$env:OPENAI_BASE_URL = $urlMatch.Groups[1].Value.TrimEnd("/")
$env:OPENAI_API_KEY = [string]$auth.OPENAI_API_KEY
if (-not $env:OPENAI_API_KEY) { throw "OPENAI_API_KEY was not found" }

$arguments = @(
    "tsx", "scripts/collect-longmemeval-agent-search.ts",
    "--data", $dataPath, "--output-root", $outputPath, "--runtime", $runtimePath,
    "--question-ids-file", $selectionPath, "--offset", $Offset, "--limit", $Limit,
    "--agent-model", $AgentModel, "--judge-model", $JudgeModel,
    "--max-agent-turns", $MaxAgentTurns, "--search-limit", $SearchLimit,
    "--overlong-policy", $OverlongPolicy
)
if ($ContinueOnError) { $arguments += "--continue-on-error" }
& npx @arguments
if ($LASTEXITCODE -ne 0) { throw "LongMemEval native agent-search collection failed with exit code $LASTEXITCODE" }
