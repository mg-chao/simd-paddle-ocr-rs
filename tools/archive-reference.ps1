[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Bundle,
    [Parameter(Mandatory = $true)][ValidatePattern('^[a-z0-9][a-z0-9-]*$')][string]$BaselineId,
    [string]$ArchiveDirectory = (Join-Path $PSScriptRoot '../../reference-archives'),
    [string]$Binary = (Join-Path $PSScriptRoot '../target/release/simd-paddle-ocr-rs.exe'),
    [string]$Catalog = (Join-Path $PSScriptRoot '../docs/baselines.json')
)
$ErrorActionPreference = 'Stop'
$bundlePath = (Resolve-Path -LiteralPath $Bundle).Path
$binaryPath = (Resolve-Path -LiteralPath $Binary).Path
$manifestPath = Join-Path $bundlePath 'manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$validator = if ($manifest.kind -eq 'model_import_contracts') { 'validate-import' } else { 'validate' }
& $binaryPath $validator $bundlePath
if ($LASTEXITCODE -ne 0) { throw 'Bundle validation failed; no archive created.' }
$archiveRoot = [IO.Path]::GetFullPath($ArchiveDirectory)
if ($archiveRoot.StartsWith($bundlePath.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Archive directory must be outside the source bundle.'
}
[IO.Directory]::CreateDirectory($archiveRoot) | Out-Null
$archivePath = Join-Path $archiveRoot "$BaselineId.zip"
$sidecarPath = "$archivePath.sha256"
$restorePath = Join-Path (Join-Path $archiveRoot 'verification') $BaselineId
foreach ($destination in @($archivePath, $sidecarPath, $restorePath)) {
    if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
}
$entries = [Collections.Generic.List[string]]::new()
foreach ($entry in Get-ChildItem -LiteralPath $bundlePath -Recurse -Force) {
    if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse points cannot be archived: $($entry.FullName)" }
    if (-not $entry.PSIsContainer) { $entries.Add([IO.Path]::GetRelativePath($bundlePath, $entry.FullName).Replace('\', '/')) }
}
$entries.Sort([StringComparer]::Ordinal)
$stream = [IO.FileStream]::new($archivePath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try {
    $zip = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create, $true)
    try {
        foreach ($relative in $entries) {
            $entry = $zip.CreateEntry($relative, [IO.Compression.CompressionLevel]::Fastest)
            $entry.LastWriteTime = [DateTimeOffset]::new(1980, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
            $output = $entry.Open()
            $input = [IO.File]::OpenRead((Join-Path $bundlePath $relative))
            try { $input.CopyTo($output, 65536) } finally { $input.Dispose(); $output.Dispose() }
        }
    } finally { $zip.Dispose() }
} finally { $stream.Dispose() }
[IO.Compression.ZipFile]::ExtractToDirectory($archivePath, $restorePath)
& $binaryPath $validator $restorePath
if ($LASTEXITCODE -ne 0) { throw 'Restored archive failed validation; catalog was not updated.' }
$archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
"$archiveHash  $BaselineId.zip" | Set-Content -LiteralPath $sidecarPath -Encoding utf8NoBOM
$catalogPath = [IO.Path]::GetFullPath($Catalog)
$catalogRows = @()
if (Test-Path -LiteralPath $catalogPath) { $catalogRows = @(Get-Content -LiteralPath $catalogPath -Raw | ConvertFrom-Json) }
if ($catalogRows | Where-Object { $_.id -eq $BaselineId }) { throw "Catalog already contains $BaselineId" }
$assets = @($manifest.models)
if ($manifest.dictionaries) { $assets += @($manifest.dictionaries) }
$record = [ordered]@{
    id = $BaselineId
    kind = if ($manifest.kind) { $manifest.kind } else { 'ocr' }
    archive_file = "$BaselineId.zip"
    archive_sha256 = $archiveHash
    archive_bytes = (Get-Item -LiteralPath $archivePath).Length
    manifest_sha256 = $manifestHash
    source_tree_sha256 = $manifest.provenance.source_inputs.tree_sha256
    reference_git_head = $manifest.provenance.git.head
    contract = $manifest.contract
    assets = @($assets | Select-Object id, path, sha256, bytes)
    restore_command = "Expand-Archive -LiteralPath '$BaselineId.zip' -DestinationPath '$BaselineId-restored'; simd-paddle-ocr-rs $validator '$BaselineId-restored'"
    verification = 'archive extracted into a new directory; all declared payloads validated'
}
$catalogRows += [pscustomobject]$record
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($catalogPath)) | Out-Null
ConvertTo-Json -InputObject @($catalogRows) -Depth 40 | Set-Content -LiteralPath $catalogPath -Encoding utf8NoBOM
Write-Output "Archived and verified $BaselineId -> $archivePath"
