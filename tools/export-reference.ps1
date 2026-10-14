[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$ReferenceRepo = (Join-Path $PSScriptRoot '../../SimdPaddleOCR'),
    [string]$InputPath,
    [string]$ReplayBundle,
    [ValidateSet('canonical', 'cpu')][string]$Profile = 'canonical',
    [ValidateSet('checkpoints', 'all')][string]$Nodes = 'checkpoints',
    [ValidateSet('ocr', 'step12')][string]$CaptureScope = 'ocr',
    [ValidateSet('tiny', 'small', 'medium')][string]$DetectorModel = 'tiny',
    [ValidateSet('default', 'compact', 'expanded')][string]$FixtureSet = 'default',
    [ValidateRange(0, 2147483647)][int]$MaxTracedLines = 3,
    [ValidateRange(1, 2147483647)][int]$Count = 20,
    [ValidateRange(0, 2147483647)][int]$Warmup = 1,
    [ValidateRange(1, 2147483647)][int]$Repeat = 3
)
$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path -LiteralPath $ReferenceRepo).Path
$exporterPath = Join-Path $PSScriptRoot 'reference-export'
$projectPath = Join-Path $exporterPath 'ReferenceExport.csproj'
$outputPath = [IO.Path]::GetFullPath($Out)
if (Test-Path -LiteralPath $outputPath) { throw "Output already exists: $outputPath. Choose a new bundle directory." }
if ($InputPath -and $ReplayBundle) { throw 'Choose InputPath or ReplayBundle.' }
if ($ReplayBundle) {
    $replayPath = (Resolve-Path -LiteralPath $ReplayBundle).Path
    $replayManifest = Get-Content -LiteralPath (Join-Path $replayPath 'manifest.json') -Raw | ConvertFrom-Json
    if (-not $PSBoundParameters.ContainsKey('Profile')) { $Profile = $replayManifest.contract.profile }
    if (-not $PSBoundParameters.ContainsKey('Nodes')) { $Nodes = $replayManifest.contract.nodes }
    if (-not $PSBoundParameters.ContainsKey('MaxTracedLines')) { $MaxTracedLines = $replayManifest.contract.max_traced_lines }
    if (-not $PSBoundParameters.ContainsKey('CaptureScope') -and $replayManifest.contract.capture_scope) { $CaptureScope = $replayManifest.contract.capture_scope }
    if (-not $PSBoundParameters.ContainsKey('DetectorModel') -and $CaptureScope -eq 'step12') { $DetectorModel = $replayManifest.contract.model }
    if ($Profile -notin @('canonical', 'cpu') -or $Nodes -notin @('checkpoints', 'all')) { throw 'Invalid replay profile or nodes.' }
}
$arguments = @('--reference-repo', $repoPath, '--exporter-source', $exporterPath, '--out', $outputPath,
    '--profile', $Profile, '--nodes', $Nodes, '--max-traced-lines', "$MaxTracedLines", '--count', "$Count",
    '--warmup', "$Warmup", '--repeat', "$Repeat", '--capture-scope', $CaptureScope,
    '--detector-model', $DetectorModel, '--fixture-set', $FixtureSet)
if ($InputPath) { $arguments += @('--input', (Resolve-Path -LiteralPath $InputPath).Path) }
if ($ReplayBundle) { $arguments += @('--replay', $replayPath) }

# Build outside the inference environment. Restore the caller's environment even on failure.
& dotnet build $projectPath -c Release "-p:ReferenceRepo=$repoPath" --nologo
if ($LASTEXITCODE -ne 0) { throw "Reference exporter build failed ($LASTEXITCODE)." }
$controls = @{
    DOTNET_PROCESSOR_COUNT = '1'
    DOTNET_EnableHWIntrinsic = $(if ($Profile -eq 'canonical') { '0' } else { '1' })
    PPOCR_NHWC = $(if ($Profile -eq 'canonical') { '0' } else { '1' })
    SIMD_OCR_NOFUSE = $(if ($Profile -eq 'canonical') { '1' } else { '0' })
    PPOCR_DET_THREADS = '1'
    PPOCR_DUMP_CONV = $null
    PPOCR_DUMP_NODES = $null
    PPOCR_DUMP_PLAN = $null
    PPOCR_DUMP_GRAPH = $null
}
$previous = @{}
try {
    foreach ($name in $controls.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        if ($null -eq $controls[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
        else { [Environment]::SetEnvironmentVariable($name, $controls[$name], 'Process') }
    }
    & dotnet (Join-Path $exporterPath 'bin/Release/net10.0/SimdPaddleOCR.ReferenceExport.dll') @arguments
    if ($LASTEXITCODE -ne 0) { throw "Reference export failed ($LASTEXITCODE). Partial output remains at $outputPath; it has no completed manifest." }
}
finally {
    foreach ($name in $previous.Keys) {
        if ($null -eq $previous[$name]) { Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue }
        else { [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process') }
    }
}
