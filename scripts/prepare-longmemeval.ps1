[CmdletBinding()]
param(
    [string]$Root = ".local-tdai/longmemeval-collection",
    [switch]$IncludeMedium
)

$ErrorActionPreference = "Stop"
$sourceRoot = Join-Path ([IO.Path]::GetFullPath($Root)) "source"
New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null
$files = @(
    "longmemeval_oracle.json",
    "longmemeval_s_cleaned.json"
)
if ($IncludeMedium) { $files += "longmemeval_m_cleaned.json" }
$records = @()
foreach ($name in $files) {
    $target = Join-Path $sourceRoot $name
    if (-not (Test-Path -LiteralPath $target)) {
        $url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/$name"
        Invoke-WebRequest -Uri $url -OutFile $target
    }
    $records += [ordered]@{
        file = $name
        bytes = (Get-Item -LiteralPath $target).Length
        sha256 = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$provenance = [ordered]@{
    benchmark = "LongMemEval"
    source = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
    license = "MIT"
    downloaded_at = [DateTime]::UtcNow.ToString("o")
    files = $records
}
Set-Content -LiteralPath (Join-Path $sourceRoot "provenance.json") `
    -Value ($provenance | ConvertTo-Json -Depth 5) -Encoding utf8
Write-Output $sourceRoot
