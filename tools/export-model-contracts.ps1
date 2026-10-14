[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$ReferenceRepo = (Join-Path $PSScriptRoot '../../SimdPaddleOCR')
)
$ErrorActionPreference = 'Stop'
$referencePath = (Resolve-Path -LiteralPath $ReferenceRepo).Path
$exporterPath = Join-Path $PSScriptRoot 'reference-export'
$outputPath = [IO.Path]::GetFullPath($Out)
if (Test-Path -LiteralPath $outputPath) { throw "Output already exists: $outputPath." }
& dotnet build (Join-Path $exporterPath 'ReferenceExport.csproj') -c Release "-p:ReferenceRepo=$referencePath" --nologo
if ($LASTEXITCODE -ne 0) { throw "Exporter build failed ($LASTEXITCODE)." }
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
    & dotnet (Join-Path $exporterPath 'bin/Release/net10.0/SimdPaddleOCR.ReferenceExport.dll') `
        --export-imports --reference-repo $referencePath --exporter-source $exporterPath --out $outputPath
    if ($LASTEXITCODE -ne 0) { throw "Import contract export failed ($LASTEXITCODE); partial output has no completed manifest." }
}
finally {
    foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process') }
}
