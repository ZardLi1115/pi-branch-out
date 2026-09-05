[CmdletBinding()]
param(
    [string]$RuntimePlatform = "linux/amd64",
    [string]$NodeImage = "docker.m.daocloud.io/library/node:22-bookworm-slim"
)

$ErrorActionPreference = "Stop"
$repoDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runtimeDir = Join-Path $repoDir "runtime"
$exportDir = Join-Path $runtimeDir "export"
$architecture = ($RuntimePlatform -replace "^linux/", "")
$archive = Join-Path $runtimeDir "pi-runtime-linux-$architecture.tar.gz"

if (Test-Path -LiteralPath $exportDir) {
    $resolvedExport = (Resolve-Path -LiteralPath $exportDir).Path
    if (-not $resolvedExport.StartsWith($runtimeDir, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove export directory outside runtime root: $resolvedExport"
    }
    Remove-Item -LiteralPath $resolvedExport -Recurse -Force
}

try {
    & docker build `
        --platform $RuntimePlatform `
        --build-arg "NODE_IMAGE=$NodeImage" `
        --target archive `
        --output "type=local,dest=$exportDir" `
        $runtimeDir
    if ($LASTEXITCODE -ne 0) {
        throw "docker build failed with exit code $LASTEXITCODE"
    }

    Move-Item -LiteralPath (Join-Path $exportDir "pi-runtime.tar.gz") -Destination $archive -Force
    if (-not (Test-Path -LiteralPath $archive) -or (Get-Item -LiteralPath $archive).Length -le 0) {
        throw "Pi runtime archive was not created: $archive"
    }
    Write-Output $archive
}
finally {
    if (Test-Path -LiteralPath $exportDir) {
        Remove-Item -LiteralPath $exportDir -Recurse -Force
    }
}
