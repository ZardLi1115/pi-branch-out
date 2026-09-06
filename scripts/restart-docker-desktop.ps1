[CmdletBinding()]
param(
    [string]$DockerDesktopPath = "C:\Program Files\Docker\Docker\frontend\Docker Desktop.exe",
    [int]$TimeoutSeconds = 240
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $DockerDesktopPath)) {
    throw "Docker Desktop executable not found: $DockerDesktopPath"
}

Get-Process -Name "Docker Desktop", "com.docker.backend", "com.docker.build" -ErrorAction SilentlyContinue |
    Stop-Process -Force
Start-Sleep -Seconds 3
Start-Process -FilePath $DockerDesktopPath -WindowStyle Hidden

$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
while ([DateTime]::UtcNow -lt $deadline) {
    $job = Start-Job -ScriptBlock {
        & docker version --format '{{.Server.Version}}' 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { throw "docker version failed" }
    }
    try {
        if (Wait-Job -Job $job -Timeout 10) {
            $version = (Receive-Job -Job $job -ErrorAction SilentlyContinue | Out-String).Trim()
            if ($version) {
                Write-Output "Docker daemon ready: $version"
                exit 0
            }
        }
        else {
            Stop-Job -Job $job
        }
    }
    finally {
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 3
}
throw "Timed out waiting for Docker Desktop daemon"
