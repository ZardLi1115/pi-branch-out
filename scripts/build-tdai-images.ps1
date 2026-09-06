[CmdletBinding()]
param(
    [string]$TdaiRoot = "D:\TDAI\TencentDB-Agent-Memory-v2.0.0-beta.1",
    [string]$BaseImage = "mcr.microsoft.com/devcontainers/javascript-node:1-22-bookworm",
    [string]$AptMirror = "mirrors.aliyun.com",
    [string]$CoreTag = "tdai-memory-core-local:v2.0.0-beta.1",
    [string]$ProxyTag = "tdai-memory-proxy-local:v2.0.0-beta.1"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $TdaiRoot).Path
$expectedCommit = "41444344ce11467a5b5ad6aa032f5e261da1f4d2"
$actualCommit = (& git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) {
    throw "TDAI source must be v2.0.0-beta.1 ($expectedCommit); found $actualCommit"
}
$dirty = & git -C $root status --porcelain
if ($LASTEXITCODE -ne 0 -or $dirty) {
    throw "TDAI v2.0.0-beta.1 source tree must be clean before building"
}
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pi-branch-out-tdai-build-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null

function Invoke-TdaiBuild {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$ContextPath,
        [Parameter(Mandatory)][string]$ImageTag,
        [switch]$PassAptMirror,
        [string]$Dockerfile = ""
    )
    $sourceDockerfile = if ($Dockerfile) { $Dockerfile } else { Join-Path $ContextPath "Dockerfile" }
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
    $coreDockerfile = Join-Path $root "MemoryCore\Dockerfile"
    if (-not (Test-Path -LiteralPath $coreDockerfile)) {
        $coreDockerfile = Join-Path (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\runtime")) "MemoryCore.v2.0.0-beta.1.Dockerfile"
    }
    Invoke-TdaiBuild -Name "memory-core" -ContextPath (Join-Path $root "MemoryCore") -ImageTag $CoreTag -PassAptMirror -Dockerfile $coreDockerfile
    $proxyDockerfile = Join-Path $root "MemoryProxy\Dockerfile"
    $costGuardPackage = Join-Path $root "MemoryProxy\packages\cost-guard\package.json"
    $costGuardSource = Join-Path $root "MemoryProxy\packages\cost-guard\src"
    if (
        -not (Test-Path -LiteralPath $proxyDockerfile) -or
        -not (Test-Path -LiteralPath $costGuardPackage) -or
        -not (Test-Path -LiteralPath $costGuardSource)
    ) {
        $proxyDockerfile = Join-Path (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\runtime")) "MemoryProxy.v2.0.0-beta.1.Dockerfile"
    }
    Invoke-TdaiBuild -Name "memory-proxy" -ContextPath (Join-Path $root "MemoryProxy") -ImageTag $ProxyTag -Dockerfile $proxyDockerfile
    Write-Output $CoreTag
    Write-Output $ProxyTag
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
