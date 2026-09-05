[CmdletBinding()]
param(
    [string]$InstanceName = "natural",
    [int]$CorePort = 8420,
    [int]$ProxyPort = 8096,
    [string]$Model = "gpt-5.6-luna",
    [string]$CoreImage = "tdai-memory-core-local:latest",
    [string]$ProxyImage = "tdai-memory-proxy-local:latest",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
if ($InstanceName -notmatch '^[a-z0-9][a-z0-9-]{0,40}$') {
    throw "InstanceName must contain only lowercase letters, digits and hyphens"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$stateRoot = Join-Path $repoRoot ".local-tdai\$InstanceName"
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
$codexRoot = Join-Path $env:USERPROFILE ".codex"
$configText = Get-Content -Raw -Encoding utf8 (Join-Path $codexRoot "config.toml")
$providerMatch = [regex]::Match(
    $configText,
    '(?ms)^\[model_providers\.custom\]\s*(.*?)(?=^\[|\z)'
)
if (-not $providerMatch.Success) {
    throw "Codex custom provider section was not found"
}
$urlMatch = [regex]::Match($providerMatch.Groups[1].Value, '(?m)^base_url\s*=\s*"([^"]+)"')
if (-not $urlMatch.Success) {
    throw "Codex custom provider base_url was not found"
}
$upstreamUrl = $urlMatch.Groups[1].Value.TrimEnd("/")
$auth = Get-Content -Raw -Encoding utf8 (Join-Path $codexRoot "auth.json") | ConvertFrom-Json
$upstreamKey = [string]$auth.OPENAI_API_KEY
if (-not $upstreamKey) {
    throw "OPENAI_API_KEY was not found in Codex auth.json"
}

function ConvertTo-YamlString([string]$Value) {
    return $Value.Replace("\", "\\").Replace('"', '\"')
}

function Wait-Http([string]$Url, [int]$TimeoutSeconds = 120) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                return
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Timed out waiting for $Url"
}

function Invoke-CorePost([string]$Path, [hashtable]$Body, [string]$UserKey = "") {
    $headers = @{ "x-tdai-service-id" = "default" }
    if ($UserKey) {
        $headers["x-tdai-user-key"] = $UserKey
    }
    return Invoke-RestMethod `
        -Uri "http://127.0.0.1:$CorePort$Path" `
        -Method Post `
        -ContentType "application/json; charset=utf-8" `
        -Headers $headers `
        -Body ($Body | ConvertTo-Json -Depth 10 -Compress)
}

$safeUrl = ConvertTo-YamlString $upstreamUrl
$safeKey = ConvertTo-YamlString $upstreamKey
$safeModel = ConvertTo-YamlString $Model
$coreConfig = Join-Path $stateRoot "tdai-gateway.yaml"
$coreYaml = @"
deployMode: standalone
stateBackend: local
server:
  port: 8420
  host: 0.0.0.0
data:
  baseDir: /data/tdai-memory
llm:
  baseUrl: "$safeUrl"
  apiKey: "$safeKey"
  model: "$safeModel"
  maxTokens: 32000
  timeoutMs: 300000
memory:
  promptMode: code
  capture: { enabled: true }
  extraction:
    enabled: true
    enableDedup: true
    maxMemoriesPerSession: 20
  persona:
    triggerEveryN: 50
    maxScenes: 15
  pipeline:
    everyNConversations: 5
    enableWarmup: true
    l1IdleTimeoutSeconds: 600
    l2DelayAfterL1Seconds: 90
    l2MinIntervalSeconds: 900
    l2MaxIntervalSeconds: 3600
  recall:
    enabled: true
    maxResults: 36
    scoreThreshold: 0.3
    strategy: hybrid
    timeoutMs: 5000
  storeBackend: sqlite
  embedding:
    provider: none
skill:
  enabled: false
"@
Set-Content -LiteralPath $coreConfig -Value $coreYaml -Encoding utf8

$network = "tdai-$InstanceName"
$coreContainer = "tdai-$InstanceName-core"
$proxyContainer = "tdai-$InstanceName-proxy"
$coreVolume = "tdai-$InstanceName-core-data"
$proxyVolume = "tdai-$InstanceName-proxy-data"
if ($Recreate) {
    foreach ($container in @($proxyContainer, $coreContainer)) {
        & docker container inspect $container *> $null
        if ($LASTEXITCODE -eq 0) {
            & docker rm -f $container *> $null
        }
    }
}
foreach ($container in @($coreContainer, $proxyContainer)) {
    & docker container inspect $container *> $null
    if ($LASTEXITCODE -eq 0) {
        throw "Container already exists: $container (use -Recreate to replace containers; volumes are preserved)"
    }
}
& docker network inspect $network *> $null
if ($LASTEXITCODE -ne 0) {
    & docker network create $network *> $null
}

$coreMount = ($coreConfig -replace '\\', '/') + ":/data/config/tdai-gateway.yaml:ro"
& docker run -d `
    --name $coreContainer `
    --network $network `
    --network-alias memory-core `
    -p "${CorePort}:8420" `
    -v "${coreVolume}:/data/tdai-memory" `
    -v $coreMount `
    -e TDAI_GATEWAY_PORT=8420 `
    -e TDAI_GATEWAY_HOST=0.0.0.0 `
    -e TDAI_GATEWAY_API_KEY= `
    -e TDAI_DATA_DIR=/data/tdai-memory `
    $CoreImage *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start $coreContainer"
}
Wait-Http "http://127.0.0.1:$CorePort/health"

$adminKeyPath = Join-Path $stateRoot "admin-key.txt"
if (Test-Path -LiteralPath $adminKeyPath) {
    $adminKey = (Get-Content -Raw -Encoding utf8 $adminKeyPath).Trim()
}
else {
    $bytes = [byte[]]::new(24)
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    $token = [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "A").Replace("/", "B")
    $adminKey = "sk-mem-$token"
    $created = Invoke-CorePost "/v3/internal/meta/user/init-admin" @{
        username = "admin-$InstanceName"
        user_key = $adminKey
    }
    if ($created.code -ne 0) {
        throw "MemoryCore admin initialization failed: code=$($created.code)"
    }
    Set-Content -LiteralPath $adminKeyPath -Value $adminKey -Encoding utf8 -NoNewline
}

$verified = Invoke-CorePost "/v3/meta/auth/verify" @{ user_key = $adminKey }
if ($verified.code -ne 0) {
    throw "MemoryCore admin key verification failed"
}
$userId = [string]$verified.data.user.user_id
$teams = Invoke-CorePost "/v3/meta/team/list" @{ user_id = $userId; limit = 100 } $adminKey
$teamId = [string]$teams.data.items[0].team_id
$agents = Invoke-CorePost "/v3/meta/agent/list" @{ team_id = $teamId; limit = 100 } $adminKey
$agentId = [string]$agents.data.items[0].agent_id
if (-not $teamId -or -not $agentId) {
    throw "Default Team/Agent was not created"
}
$task = Invoke-CorePost "/v3/meta/task/create" @{
    team_id = $teamId
    creator_user_id = $userId
    title = "RoadmapBench budget collection"
    description = "pi-branch-out isolated experiment task"
} $adminKey
$taskId = [string]$task.data.task_id
if ($taskId) {
    $null = Invoke-CorePost "/v3/meta/task-agent/link" @{
        task_id = $taskId
        agent_id = $agentId
        role_in_task = "primary"
    } $adminKey
}

$proxyConfig = Join-Path $stateRoot "proxy.yaml"
$proxyYaml = @"
server:
  host: 0.0.0.0
  port: 8096
  forwardTimeoutMs: 600000
upstream:
  url: "$safeUrl"
  apiKey: "$safeKey"
  agents:
    codex:
      url: "$safeUrl"
      apiKey: "$safeKey"
log:
  file: ""
  level: info
  backend: console
tdai:
  enabled: true
  endpoint: "http://memory-core:8420"
  apiKey: ""
  serviceId: default
  memory:
    enabled: true
    inject: true
    writeL0: true
    recallL1: true
    injectL2L3: true
    l1Limit: 36
    timeoutMs: 5000
skill:
  endpoint: "http://memory-core:8420"
  serviceToken: ""
  serviceId: default
  timeoutMs: 5000
knowledge:
  enabled: false
auth:
  enabled: true
  url: "http://memory-core:8420"
  timeoutMs: 5000
sessionInit:
  enabled: true
  maxRetries: 3
  injectAgentContext: true
  injectTaskContext: true
  headerAutoSelect:
    enabled: true
    teamHeader: "x-team-id"
    agentHeader: "x-agent-id"
    taskHeader: "x-task-id"
    onMismatch: "form"
injection:
  enabled: true
  injectors: [tdai-memory]
extraction:
  enabled: true
  extractors: [tdai-memory]
storage:
  enabled: true
  backend: sqlite
  ttlDays: 7
  sqlite:
    dbPath: "/data/tdai-memory-proxy/storage.db"
  fs:
    fsRoot: "/data/tdai-memory-proxy/fs-storage"
redis:
  enabled: false
"@
Set-Content -LiteralPath $proxyConfig -Value $proxyYaml -Encoding utf8
$proxyMount = ($proxyConfig -replace '\\', '/') + ":/data/config.yaml:ro"
& docker run -d `
    --name $proxyContainer `
    --network $network `
    --network-alias proxy `
    --add-host host.docker.internal:host-gateway `
    -p "${ProxyPort}:8096" `
    -v "${proxyVolume}:/data/tdai-memory-proxy" `
    -v $proxyMount `
    $ProxyImage *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start $proxyContainer"
}
Wait-Http "http://127.0.0.1:$ProxyPort/health"

$runtime = @{
    instance_name = $InstanceName
    TDAI_PROXY_URL = "http://127.0.0.1:$ProxyPort"
    TDAI_SPACE_ID = "default"
    TDAI_AGENT_SOURCE = "codex"
    TDAI_WIRE_API = "responses"
    TDAI_TEAM_ID = $teamId
    TDAI_AGENT_ID = $agentId
    TDAI_TASK_ID = $taskId
    TDAI_USER_KEY = $adminKey
    TDAI_MODEL = $Model
    core_container = $coreContainer
    proxy_container = $proxyContainer
    core_volume = $coreVolume
    proxy_volume = $proxyVolume
}
$runtimePath = Join-Path $stateRoot "runtime.json"
Set-Content -LiteralPath $runtimePath -Value ($runtime | ConvertTo-Json -Depth 5) -Encoding utf8
Write-Output $runtimePath
