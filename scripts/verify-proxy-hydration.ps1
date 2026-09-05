[CmdletBinding()]
param(
    [string]$WiringRoot = "D:\TDAI\pi-branch-out\.local-tdai\wiring-smoke",
    [string]$ProxyContainer = "tdai-natural-proxy",
    [int]$ProxyPort = 8096
)

$ErrorActionPreference = "Stop"
$latest = Get-ChildItem -LiteralPath $WiringRoot -Directory | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $latest) {
    throw "No wiring smoke output was found"
}
$snapshot = Get-ChildItem -LiteralPath $latest.FullName -Recurse -Filter "recall-snapshot.json" | Select-Object -First 1
if (-not $snapshot) {
    throw "No ready recall snapshot was found"
}
$snapshotValue = Get-Content -Raw -Encoding utf8 $snapshot.FullName | ConvertFrom-Json
$conversationId = [string]$snapshotValue.conversation_id
if (-not $conversationId) {
    throw "Recall snapshot has no conversation id"
}

& docker restart $ProxyContainer *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Proxy restart failed"
}
$deadline = [DateTime]::UtcNow.AddSeconds(90)
do {
    try {
        $health = Invoke-WebRequest -Uri "http://127.0.0.1:$ProxyPort/health" -UseBasicParsing -TimeoutSec 5
        if ($health.StatusCode -eq 200) { break }
    }
    catch {
        Start-Sleep -Seconds 2
    }
} while ([DateTime]::UtcNow -lt $deadline)
if (-not $health -or $health.StatusCode -ne 200) {
    throw "Proxy did not become healthy after restart"
}

$headers = @{
    "x-conversation-id" = "codex:$conversationId"
    "x-tdai-service-id" = "default"
}
$results = @()
foreach ($endpoint in @("atomic/search", "conversation/search")) {
    $headers["x-conversation-id"] = "codex:$conversationId"
    try {
        $response = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$ProxyPort/memory-bridge/v3/$endpoint" `
            -Method Post `
            -ContentType "application/json; charset=utf-8" `
            -Headers $headers `
            -Body '{"query":"hydration check","limit":5}'
    }
    catch {
        $detail = [string]$_.ErrorDetails.Message
        if ($detail -notmatch "session not initialized") {
            throw
        }
        $headers["x-conversation-id"] = $conversationId
        $response = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$ProxyPort/memory-bridge/v3/$endpoint" `
            -Method Post `
            -ContentType "application/json; charset=utf-8" `
            -Headers $headers `
            -Body '{"query":"hydration check","limit":5}'
    }
    if ($response.code -ne 0) {
        throw "Hydrated bridge query failed for $endpoint"
    }
    $results += $endpoint
}
[pscustomobject]@{
    proxy_restarted = $true
    persisted_session_resolved = $true
    bridge_queries = $results
} | ConvertTo-Json -Compress
