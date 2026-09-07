[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$TdaiRoot,
    [Parameter(Mandatory)][string]$ExpectedCommit,
    [Parameter(Mandatory)][string]$Version,
    [string]$BaseImage = "mcr.microsoft.com/devcontainers/javascript-node:1-22-bookworm",
    [string]$AptMirror = "mirrors.aliyun.com",
    [string]$CoreTag = "",
    [string]$ProxyTag = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:DOCKER_BUILDKIT = "1"

$root = (Resolve-Path -LiteralPath $TdaiRoot).Path
$actualCommit = (& git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $ExpectedCommit) {
    throw "TDAI source must be $Version ($ExpectedCommit); found $actualCommit"
}
$dirty = & git -C $root status --porcelain
if ($LASTEXITCODE -ne 0 -or $dirty) {
    throw "TDAI source tree must be clean before building"
}
if (-not $CoreTag) { $CoreTag = "tdai-memory-core-local:$Version" }
if (-not $ProxyTag) { $ProxyTag = "tdai-memory-proxy-local:$Version" }

$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pi-branch-out-tdai-release-build-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null

function Invoke-TdaiReleaseBuild {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$ContextPath,
        [Parameter(Mandatory)][string]$ImageTag,
        [switch]$PassAptMirror
    )
    $sourceDockerfile = Join-Path $ContextPath "Dockerfile"
    if (-not (Test-Path -LiteralPath $sourceDockerfile)) {
        throw "Dockerfile not found: $sourceDockerfile"
    }
    $temporaryDockerfile = Join-Path $temporaryRoot "$Name.Dockerfile"
    $content = Get-Content -Raw -Encoding UTF8 $sourceDockerfile
    $content = $content -replace '(?m)^# syntax=.*\r?\n', ''
    if ($BaseImage) {
        $content = $content -replace 'FROM node:22-slim', "FROM $BaseImage"
    }
    Set-Content -LiteralPath $temporaryDockerfile -Value $content -Encoding UTF8
    $arguments = @("build", "-f", $temporaryDockerfile, "-t", $ImageTag)
    if ($PassAptMirror) { $arguments += @("--build-arg", "APT_MIRROR=$AptMirror") }
    $arguments += $ContextPath
    & docker @arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name image build failed with exit code $LASTEXITCODE" }
}

try {
    Invoke-TdaiReleaseBuild `
        -Name "memory-core" `
        -ContextPath (Join-Path $root "MemoryCore") `
        -ImageTag $CoreTag `
        -PassAptMirror
    Invoke-TdaiReleaseBuild `
        -Name "memory-proxy" `
        -ContextPath (Join-Path $root "MemoryProxy") `
        -ImageTag $ProxyTag
    Write-Output $CoreTag
    Write-Output $ProxyTag
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
