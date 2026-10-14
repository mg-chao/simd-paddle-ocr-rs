[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$ReferenceRepo = (Join-Path $PSScriptRoot '../../SimdPaddleOCR'),
    [ValidateRange(0, 2147483647)][int]$Warmup = 3,
    [ValidateRange(1, 2147483647)][int]$Samples = 30
)
$ErrorActionPreference = 'Stop'
$rustRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$referencePath = (Resolve-Path -LiteralPath $ReferenceRepo).Path
$outputPath = [IO.Path]::GetFullPath($Out)
if (Test-Path -LiteralPath $outputPath) { throw "Output already exists: $outputPath" }
[IO.Directory]::CreateDirectory($outputPath) | Out-Null
Push-Location $rustRoot
try {
    & cargo build --release --locked --bin import-benchmark
    if ($LASTEXITCODE -ne 0) { throw 'Timing binary build failed.' }
    $timingExe = Join-Path $outputPath 'import-timing.exe'
    Copy-Item -LiteralPath (Join-Path $rustRoot 'target/release/import-benchmark.exe') -Destination $timingExe
    & cargo build --release --locked --bin import-benchmark --features heap-profiler
    if ($LASTEXITCODE -ne 0) { throw 'Heap profiling binary build failed.' }
    $heapExe = Join-Path $outputPath 'import-heap.exe'
    Copy-Item -LiteralPath (Join-Path $rustRoot 'target/release/import-benchmark.exe') -Destination $heapExe
} finally { Pop-Location }
$modelNames = @('cls', 'det', 'rec', 'small_det', 'small_rec', 'medium_det', 'medium_rec')
foreach ($modelName in @($modelNames + 'all')) {
    $names = if ($modelName -eq 'all') { $modelNames } else { @($modelName) }
    $rows = @{}
    foreach ($mode in @('timing', 'heap')) {
        $executable = if ($mode -eq 'timing') { $timingExe } else { $heapExe }
        $startInfo = [Diagnostics.ProcessStartInfo]::new($executable)
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        foreach ($name in $names) { $startInfo.ArgumentList.Add((Join-Path $referencePath "models/$name.onnx")) }
        foreach ($value in @('--warmup', "$Warmup", '--repeat', "$Samples")) { $startInfo.ArgumentList.Add($value) }
        $process = [Diagnostics.Process]::Start($startInfo)
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $observedPeak = 0L
        while (-not $process.WaitForExit(5)) {
            try { $process.Refresh(); $observedPeak = [Math]::Max($observedPeak, $process.PeakWorkingSet64) }
            catch [InvalidOperationException] { if (-not $process.HasExited) { throw } }
        }
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        if ($process.ExitCode -ne 0) { throw "Rust $mode benchmark $modelName failed: $stderr" }
        $row = $stdout | ConvertFrom-Json
        $row | Add-Member -NotePropertyName observed_process_peak_working_set_bytes -NotePropertyValue $observedPeak
        $row | Add-Member -NotePropertyName process_memory_semantics -NotePropertyValue 'Parent sampled OS high-water resident memory while child lived; lower bound if final samples were missed.'
        $rows[$mode] = $row
        $row | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $outputPath "rust-$mode-$modelName.json") -Encoding utf8NoBOM
        $process.Dispose()
    }
    'Rust {0}: p50={1:F3} ms p95={2:F3} ms; retained heap={3:N0} bytes peak heap={4:N0} bytes' -f $modelName, $rows.timing.median_ms, $rows.timing.p95_ms, $rows.heap.samples[0].retained_heap_bytes, (($rows.heap.samples | Measure-Object peak_heap_bytes -Maximum).Maximum)
}
