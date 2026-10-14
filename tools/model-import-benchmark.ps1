[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$ReferenceRepo = (Join-Path $PSScriptRoot '../../SimdPaddleOCR'),
    [ValidateSet('cls', 'det', 'rec', 'small_det', 'small_rec', 'medium_det', 'medium_rec', 'all')]
    [string[]]$Models = @('cls', 'det', 'rec', 'small_det', 'small_rec', 'medium_det', 'medium_rec', 'all'),
    [ValidateRange(0, 2147483647)][int]$Warmup = 3,
    [ValidateRange(1, 2147483647)][int]$Samples = 30
)
$ErrorActionPreference = 'Stop'
$referencePath = (Resolve-Path -LiteralPath $ReferenceRepo).Path
$exporterPath = Join-Path $PSScriptRoot 'reference-export'
$outputPath = [IO.Path]::GetFullPath($Out)
if (Test-Path -LiteralPath $outputPath) { throw "Output already exists: $outputPath." }
& dotnet build (Join-Path $exporterPath 'ReferenceExport.csproj') -c Release "-p:ReferenceRepo=$referencePath" --nologo
if ($LASTEXITCODE -ne 0) { throw "Exporter build failed ($LASTEXITCODE)." }
New-Item -ItemType Directory -Path $outputPath | Out-Null
$controls = @{
    DOTNET_PROCESSOR_COUNT = '1'; DOTNET_EnableHWIntrinsic = '0'; PPOCR_NHWC = '0'; SIMD_OCR_NOFUSE = '1'
    PPOCR_DUMP_CONV = $null; PPOCR_DUMP_NODES = $null; PPOCR_DUMP_PLAN = $null; PPOCR_DUMP_GRAPH = $null
}
$previous = @{}
try {
    foreach ($name in $controls.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $controls[$name], 'Process')
    }
    foreach ($modelName in ($Models | Select-Object -Unique)) {
        # A fresh process for every model, including the sequential all-model case.
        & dotnet (Join-Path $exporterPath 'bin/Release/net10.0/SimdPaddleOCR.ReferenceExport.dll') `
            --benchmark-imports --reference-repo $referencePath --model $modelName `
            --warmup "$Warmup" --samples "$Samples" --out (Join-Path $outputPath "csharp-$modelName.json")
        if ($LASTEXITCODE -ne 0) { throw "Import benchmark $modelName failed ($LASTEXITCODE)." }
    }
}
finally {
    foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process') }
}
