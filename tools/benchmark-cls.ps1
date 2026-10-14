[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][Alias('ReferenceBundle')][string]$Reference,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$ReferenceRepo = (Join-Path $PSScriptRoot '../../SimdPaddleOCR'),
    [ValidateRange(1, 2147483647)][int]$Rounds = 3,
    [ValidateRange(0, 2147483647)][int]$Warmup = 30,
    [ValidateRange(1, 2147483647)][Alias('Samples')][int]$Repeat = 100
)
$ErrorActionPreference = 'Stop'
$rustRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$bundlePath = (Resolve-Path -LiteralPath $Reference).Path
$referencePath = (Resolve-Path -LiteralPath $ReferenceRepo).Path
$outputPath = [IO.Path]::GetFullPath($Out)
$exporterPath = Join-Path $PSScriptRoot 'reference-export'
if (Test-Path -LiteralPath $outputPath) { throw "Output already exists: $outputPath" }
[IO.Directory]::CreateDirectory($outputPath) | Out-Null

Push-Location $rustRoot
try {
    & cargo build --release --locked --bin cls-benchmark
    if ($LASTEXITCODE -ne 0) { throw 'Uninstrumented CLS benchmark build failed.' }
    $timingExe = Join-Path $outputPath 'cls-timing.exe'
    Copy-Item -LiteralPath (Join-Path $rustRoot 'target/release/cls-benchmark.exe') -Destination $timingExe
    & cargo build --release --locked --bin cls-benchmark --features heap-profiler
    if ($LASTEXITCODE -ne 0) { throw 'Allocation profiling CLS benchmark build failed.' }
    $heapExe = Join-Path $outputPath 'cls-heap.exe'
    Copy-Item -LiteralPath (Join-Path $rustRoot 'target/release/cls-benchmark.exe') -Destination $heapExe
    & dotnet build (Join-Path $exporterPath 'ReferenceExport.csproj') -c Release "-p:ReferenceRepo=$referencePath" --nologo
    if ($LASTEXITCODE -ne 0) { throw 'C# CLS benchmark build failed.' }
} finally { Pop-Location }

$controls = @{
    DOTNET_PROCESSOR_COUNT = '1'; DOTNET_EnableHWIntrinsic = '0'
    PPOCR_NHWC = '0'; SIMD_OCR_NOFUSE = '1'; PPOCR_DET_THREADS = '1'
    PPOCR_DUMP_CONV = $null; PPOCR_DUMP_NODES = $null
    PPOCR_DUMP_PLAN = $null; PPOCR_DUMP_GRAPH = $null
}
function Invoke-ClsMeasurement([string]$Mode, [int]$Round) {
    $executable = switch ($Mode) { 'csharp' { 'dotnet' } 'rust-timing' { $timingExe } 'rust-heap' { $heapExe } }
    $startInfo = [Diagnostics.ProcessStartInfo]::new($executable)
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($name in $controls.Keys) {
        if ($null -eq $controls[$name]) { $startInfo.Environment.Remove($name) | Out-Null }
        else { $startInfo.Environment[$name] = $controls[$name] }
    }
    if ($Mode -eq 'csharp') {
        $startInfo.ArgumentList.Add((Join-Path $exporterPath 'bin/Release/net10.0/SimdPaddleOCR.ReferenceExport.dll'))
        $startInfo.ArgumentList.Add('--benchmark-cls')
    }
    foreach ($value in @('--reference', $bundlePath, '--warmup', "$Warmup", '--samples', "$Repeat")) {
        $startInfo.ArgumentList.Add($value)
    }
    $process = [Diagnostics.Process]::Start($startInfo)
    try {
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $observedPeak = 0L
        while (-not $process.WaitForExit(10)) {
            try { $process.Refresh(); $observedPeak = [Math]::Max($observedPeak, $process.PeakWorkingSet64) }
            catch [InvalidOperationException] { if (-not $process.HasExited) { throw } }
        }
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        if ($process.ExitCode -ne 0) { throw "$Mode CLS benchmark round $Round failed ($($process.ExitCode)): $stderr" }
        if ($stderr) { Write-Host $stderr.TrimEnd() }
        $report = $stdout | ConvertFrom-Json
        if ($report.schema_version -ne 1 -or $report.profile -ne 'canonical' -or $report.inputs.Count -eq 0) {
            throw "$Mode emitted an invalid benchmark report."
        }
        if ($Mode -eq 'rust-heap' -and ($report.warm_allocation_count -ne 0 -or $report.warm_allocation_bytes -ne 0)) {
            throw 'Warm Rust CLS inference must allocate zero times and zero bytes.'
        }
        $report | Add-Member -NotePropertyName round -NotePropertyValue $Round
        $report | Add-Member -NotePropertyName mode -NotePropertyValue $Mode
        $report | Add-Member -NotePropertyName observed_process_peak_working_set_bytes -NotePropertyValue $observedPeak
        $report | Add-Member -NotePropertyName process_memory_semantics -NotePropertyValue 'Parent sampled process high-water resident memory; includes bundle validation, model loading, runtime, inputs and bookkeeping. This is not activation memory or a managed/native heap peak.'
        $report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $outputPath ("{0}-round-{1:D2}.json" -f $Mode, $Round)) -Encoding utf8NoBOM
        Write-Host ('{0} round {1}: corpus p50={2:F3} ms p95={3:F3} ms; {4:F1} inputs/s' -f $Mode, $Round, $report.corpus.p50_ms, $report.corpus.p95_ms, $report.corpus.inputs_per_second)
        return $report
    } finally { $process.Dispose() }
}

# Each timing pair runs in fresh processes. Alternating process order limits
# systematic filesystem/cache/thermal bias. Allocation-instrumented runs follow
# every timing pair and are excluded from latency comparisons.
$measurements = [Collections.Generic.List[object]]::new()
$orders = [Collections.Generic.List[object]]::new()
for ($round = 1; $round -le $Rounds; $round++) {
    $order = if (($round % 2) -eq 1) { @('csharp', 'rust-timing') } else { @('rust-timing', 'csharp') }
    $orders.Add([pscustomobject]@{ round = $round; modes = $order })
    foreach ($mode in $order) { $measurements.Add((Invoke-ClsMeasurement $mode $round)) }
}
for ($round = 1; $round -le $Rounds; $round++) { $measurements.Add((Invoke-ClsMeasurement 'rust-heap' $round)) }
$first = $measurements[0]
foreach ($report in $measurements) {
    if ($report.model_sha256 -ne $first.model_sha256 -or $report.manifest_sha256 -ne $first.manifest_sha256 -or $report.inputs.Count -ne $first.inputs.Count) {
        throw 'Benchmark processes consumed different model or corpus identities.'
    }
    for ($index = 0; $index -lt $first.inputs.Count; $index++) {
        $expected = $first.inputs[$index]; $actual = $report.inputs[$index]
        if ($actual.case_id -ne $expected.case_id -or $actual.artifact_id -ne $expected.artifact_id -or $actual.sha256 -ne $expected.sha256 -or
            ($actual.shape -join ',') -ne ($expected.shape -join ',')) { throw 'Benchmark input ordering/identity differs between processes.' }
    }
}
$summary = [ordered]@{
    schema_version = 1; reference = $bundlePath; model_sha256 = $first.model_sha256; manifest_sha256 = $first.manifest_sha256
    rounds = $Rounds; warmup_sweeps = $Warmup; measured_sweeps = $Repeat; captured_inputs = $first.inputs.Count
    process_orders = $orders.ToArray()
    measurement = 'Fresh-process canonical C# and scalar Rust graph timing on identical frozen inputs. Warmup/measured counts mean complete corpus sweeps. Allocation instrumentation is isolated in separate Rust processes; no speed target or SIMD claim.'
    zero_rust_warm_allocations = $true
    environment = [ordered]@{
        os = [Runtime.InteropServices.RuntimeInformation]::OSDescription
        architecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
        logical_processors = [Environment]::ProcessorCount
        rustc = (& rustc -V)
        dotnet_sdk = (& dotnet --version)
        rustflags = $env:RUSTFLAGS
        rust_timing_sha256 = (Get-FileHash -LiteralPath $timingExe -Algorithm SHA256).Hash.ToLowerInvariant()
        rust_heap_sha256 = (Get-FileHash -LiteralPath $heapExe -Algorithm SHA256).Hash.ToLowerInvariant()
        csharp_exporter_sha256 = (Get-FileHash -LiteralPath (Join-Path $exporterPath 'bin/Release/net10.0/SimdPaddleOCR.ReferenceExport.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    runs = @($measurements | ForEach-Object { [ordered]@{
        mode = $_.mode; round = $_.round; load_ms = $_.load_ms; prepare_ms = $_.prepare_ms; first_sweep_ms = $_.first_sweep_ms
        corpus_p50_ms = $_.corpus.p50_ms; corpus_p95_ms = $_.corpus.p95_ms; inputs_per_second = $_.corpus.inputs_per_second
        warm_allocation_count = $_.warm_allocation_count; warm_allocation_bytes = $_.warm_allocation_bytes
        activation_bytes = $_.activation_bytes; naive_activation_bytes = $_.naive_activation_bytes
        observed_process_peak_working_set_bytes = $_.observed_process_peak_working_set_bytes
    } })
}
$summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $outputPath 'summary.json') -Encoding utf8NoBOM
Write-Host "Saved CLS timing samples and allocation measurements: $outputPath"
