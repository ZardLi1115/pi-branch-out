[CmdletBinding()]
param(
    [string]$RuntimeConfig = "D:\TDAI\pi-branch-out\.local-tdai\natural\runtime.json",
    [string]$RuntimeArchive = "D:\TDAI\pi-branch-out\runtime\pi-runtime-linux-amd64.tar.gz",
    [string]$Image = "znpt/roadmapbench-glz-3.0.0-roadmap:latest"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$tdaiRoot = (Resolve-Path -LiteralPath "D:\TDAI\TencentDB-Agent-Memory").Path
$runtime = Get-Content -Raw -Encoding utf8 $RuntimeConfig | ConvertFrom-Json
$runId = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$output = Join-Path $repoRoot ".local-tdai\wiring-smoke\$runId"
New-Item -ItemType Directory -Path $output | Out-Null
$envFile = Join-Path $output "container.env"
$containerScript = Join-Path $output "run.sh"
$envLines = @(
    "TDAI_PROXY_URL=http://host.docker.internal:8096",
    "TDAI_SPACE_ID=$($runtime.TDAI_SPACE_ID)",
    "TDAI_AGENT_SOURCE=codex",
    "TDAI_WIRE_API=responses",
    "TDAI_TEAM_ID=$($runtime.TDAI_TEAM_ID)",
    "TDAI_AGENT_ID=$($runtime.TDAI_AGENT_ID)",
    "TDAI_TASK_ID=$($runtime.TDAI_TASK_ID)",
    "TDAI_USER_KEY=$($runtime.TDAI_USER_KEY)",
    "TDAI_MODEL=gpt-5.6-luna",
    "PI_BRANCH_OUT_MODEL_CALL_DIR=/output",
    "PI_BRANCH_OUT_TASK_NAME=tdai-wiring-smoke"
)
Set-Content -LiteralPath $envFile -Value $envLines -Encoding utf8
$scriptText = @'
set -eu
mkdir -p /tmp/pi-runtime /tmp/pi-session
tar -xzf /input/pi-runtime.tar.gz -C /tmp/pi-runtime --strip-components=1
/tmp/pi-runtime/bin/node /tmp/pi-runtime/lib/node_modules/@mariozechner/pi-coding-agent/dist/cli.js \
  --print --mode json --session-dir /tmp/pi-session --thinking low \
  --model tdai/gpt-5.6-luna \
  --extension /tdai/MemoryCore/pi-plugin/index.ts \
  --extension /branch/extensions/tdai-conversation-id.ts \
  --extension /branch/extensions/tdai-model-call-collector.ts \
  "Use the bash tool to run pwd, then answer with one short sentence." </dev/null
'@
Set-Content -LiteralPath $containerScript -Value $scriptText -Encoding utf8 -NoNewline

function New-DockerMountSpec([string]$HostPath, [string]$ContainerPath, [switch]$ReadOnly) {
    $normalized = $HostPath -replace '\\', '/'
    return $normalized + ":" + $ContainerPath + $(if ($ReadOnly) { ":ro" } else { "" })
}

try {
    $arguments = @(
        "run", "--rm",
        "--add-host", "host.docker.internal:host-gateway",
        "--env-file", $envFile,
        "-v", (New-DockerMountSpec -HostPath $RuntimeArchive -ContainerPath "/input/pi-runtime.tar.gz" -ReadOnly),
        "-v", (New-DockerMountSpec -HostPath $containerScript -ContainerPath "/input/run.sh" -ReadOnly),
        "-v", (New-DockerMountSpec -HostPath $repoRoot -ContainerPath "/branch" -ReadOnly),
        "-v", (New-DockerMountSpec -HostPath $tdaiRoot -ContainerPath "/tdai" -ReadOnly),
        "-v", (New-DockerMountSpec -HostPath $output -ContainerPath "/output"),
        "--entrypoint", "/bin/sh",
        $Image,
        "-lc",
        "/bin/sh /input/run.sh"
    )
    $dockerLog = Join-Path $output "docker-output.log"
    & docker @arguments *> $dockerLog
    $dockerExitCode = $LASTEXITCODE
    if ($dockerExitCode -ne 0) {
        $tail = (Get-Content -Encoding utf8 $dockerLog | Select-Object -Last 40) -join [Environment]::NewLine
        throw "Pi wiring smoke exited with code $dockerExitCode`n$tail"
    }
    $statesPath = Join-Path $output "model-call-states.jsonl"
    if (-not (Test-Path -LiteralPath $statesPath)) {
        $tail = (Get-Content -Encoding utf8 $dockerLog | Select-Object -Last 40) -join [Environment]::NewLine
        throw "model-call states were not created`n$tail"
    }
    $states = Get-Content -Encoding utf8 $statesPath | ForEach-Object { $_ | ConvertFrom-Json }
    $ready = @($states | Where-Object recall_snapshot_status -eq "ready").Count
    if ($states.Count -lt 2 -or $ready -lt 1) {
        throw "wiring smoke did not produce a ready post-initialization recall checkpoint"
    }
    [pscustomobject]@{
        output = $output
        model_calls = $states.Count
        ready_checkpoints = $ready
        bridge_errors = @($states | Where-Object recall_snapshot_status -eq "bridge-error").Count
    } | ConvertTo-Json -Compress
}
finally {
    if (Test-Path -LiteralPath $envFile) {
        Remove-Item -LiteralPath $envFile -Force
    }
    if (Test-Path -LiteralPath $containerScript) {
        Remove-Item -LiteralPath $containerScript -Force
    }
}
