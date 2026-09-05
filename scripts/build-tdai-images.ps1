[CmdletBinding()]
param(
    [string]$TdaiRoot = "D:\TDAI\TencentDB-Agent-Memory",
    [string]$BaseImage = "mcr.microsoft.com/devcontainers/javascript-node:1-22-bookworm",
    [string]$AptMirror = "mirrors.aliyun.com",
    [string]$CoreTag = "tdai-memory-core-local:latest",
    [string]$ProxyTag = "tdai-memory-proxy-local:latest"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $TdaiRoot).Path
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pi-branch-out-tdai-build-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null

function Invoke-TdaiBuild {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$ContextPath,
        [Parameter(Mandatory)][string]$ImageTag,
        [switch]$PassAptMirror
    )
    $sourceDockerfile = Join-Path $ContextPath "Dockerfile"
    $temporaryDockerfile = Join-Path $temporaryRoot "$Name.Dockerfile"
    $content = Get-Content -Raw -Encoding utf8 $sourceDockerfile
    $content = $content -replace '(?m)^# syntax=.*\r?\n', ''
    $content = $content -replace 'FROM node:22-slim', "FROM $BaseImage"
    Set-Content -LiteralPath $temporaryDockerfile -Value $content -Encoding utf8
    $arguments = @("build", "-f", $temporaryDockerfile, "-t", $ImageTag)
    if ($PassAptMirror) {
        $arguments += @("--build-arg", "APT_MIRROR=$AptMirror")
    }
    $arguments += $ContextPath
    & docker @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name image build failed with exit code $LASTEXITCODE"
    }
}

try {
    Invoke-TdaiBuild -Name "memory-core" -ContextPath (Join-Path $root "MemoryCore") -ImageTag $CoreTag -PassAptMirror
    Invoke-TdaiBuild -Name "memory-proxy" -ContextPath (Join-Path $root "MemoryProxy") -ImageTag $ProxyTag
    Write-Output $CoreTag
    Write-Output $ProxyTag
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
