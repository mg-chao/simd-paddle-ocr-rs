[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Reference,
    [Parameter(Mandatory=$true)][string]$Out,
    [string]$ReferenceRepo = (Join-Path $PSScriptRoot '../../SimdPaddleOCR'),
    [ValidateSet('det_graph','pipeline','stages','mixed_shapes')][string[]]$Modes = @('det_graph','pipeline','stages','mixed_shapes'),
    [string]$Case,
    [ValidateRange(1,100)][int]$Rounds=6,
    [ValidateRange(0,100)][int]$HeapRounds=3,
    [ValidateRange(0,2147483647)][int]$Warmup=30,
    [ValidateRange(1,2147483647)][int]$Repeat=100
)
$ErrorActionPreference='Stop'
$rustRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$bundlePath=(Resolve-Path -LiteralPath $Reference).Path
$referencePath=(Resolve-Path -LiteralPath $ReferenceRepo).Path
$outputPath=[IO.Path]::GetFullPath($Out)
$exporterPath=Join-Path $PSScriptRoot 'reference-export'
if(Test-Path -LiteralPath $outputPath){throw "Output already exists: $outputPath"}
[IO.Directory]::CreateDirectory($outputPath)|Out-Null
Push-Location $rustRoot
try {
    & cargo build --release --locked --offline --bin step12-benchmark
    if($LASTEXITCODE -ne 0){throw 'Uninstrumented benchmark build failed'}
    $timingExe=Join-Path $outputPath 'step12-timing.exe'
    Copy-Item -LiteralPath (Join-Path $rustRoot 'target/release/step12-benchmark.exe') -Destination $timingExe
    & cargo build --release --locked --offline --bin step12-benchmark --features heap-profiler
    if($LASTEXITCODE -ne 0){throw 'Allocation benchmark build failed'}
    $heapExe=Join-Path $outputPath 'step12-heap.exe'
    Copy-Item -LiteralPath (Join-Path $rustRoot 'target/release/step12-benchmark.exe') -Destination $heapExe
    & dotnet build (Join-Path $exporterPath 'ReferenceExport.csproj') -c Release "-p:ReferenceRepo=$referencePath" --nologo
    if($LASTEXITCODE -ne 0){throw 'C# benchmark build failed'}
} finally {Pop-Location}
$exporterDll=Join-Path $exporterPath 'bin/Release/net10.0/SimdPaddleOCR.ReferenceExport.dll'
$nativeDll=Join-Path $exporterPath 'bin/Release/net10.0/Sdcb.SimdPaddleOCR.dll'
$nativeGitHead=(& git -C $referencePath rev-parse HEAD)
$sourceFiles=@()
Push-Location $rustRoot
try {
    $sourceFiles=@(& git ls-files --cached --others --exclude-standard | Where-Object {$_ -match '\.(rs|cs|ps1|toml)$' -or $_ -eq 'Cargo.lock'} | Sort-Object -Unique | ForEach-Object {
        [ordered]@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
    $sourceHead=(& git rev-parse HEAD)
} finally {Pop-Location}
$sourceFingerprint=[Text.Encoding]::UTF8.GetBytes((@($sourceFiles|ForEach-Object {"$($_.path) $($_.sha256)"}) -join "`n"))
$sourceSha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($sourceFingerprint)).ToLowerInvariant()
$sourceFiles|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $outputPath 'source-files.json') -Encoding utf8NoBOM
$controls=@{
    DOTNET_PROCESSOR_COUNT='1'; DOTNET_EnableHWIntrinsic='0'; DOTNET_TieredCompilation='0'; DOTNET_TieredPGO='0'; PPOCR_NHWC='0';
    SIMD_OCR_NOFUSE='1'; PPOCR_DET_THREADS='1'; PPOCR_DUMP_CONV=$null;
    PPOCR_DUMP_NODES=$null; PPOCR_DUMP_PLAN=$null; PPOCR_DUMP_GRAPH=$null
}
function Invoke-Step12Run([string]$Mode,[string]$Engine,[int]$Round) {
    $executable=if($Engine -eq 'csharp'){'dotnet'}elseif($Engine -eq 'rust-timing'){$timingExe}else{$heapExe}
    $start=[Diagnostics.ProcessStartInfo]::new($executable)
    $start.UseShellExecute=$false;$start.CreateNoWindow=$true
    $start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    foreach($name in $controls.Keys){if($null -eq $controls[$name]){$start.Environment.Remove($name)|Out-Null}else{$start.Environment[$name]=$controls[$name]}}
    if($Engine -eq 'csharp'){$start.ArgumentList.Add($exporterDll);$start.ArgumentList.Add('--benchmark-step12')}
    foreach($value in @('--reference',$bundlePath,'--mode',$Mode,'--warmup',"$Warmup",'--samples',"$Repeat")){$start.ArgumentList.Add($value)}
    if($Case){$start.ArgumentList.Add('--case');$start.ArgumentList.Add($Case)}
    $process=[Diagnostics.Process]::Start($start)
    try {
        $stdoutTask=$process.StandardOutput.ReadToEndAsync();$stderrTask=$process.StandardError.ReadToEndAsync()
        $observedPeak=0L
        while(-not $process.WaitForExit(100)){
            try{$process.Refresh();$observedPeak=[Math]::Max($observedPeak,$process.PeakWorkingSet64)} catch [InvalidOperationException] {if(-not $process.HasExited){throw}}
        }
        $stdout=$stdoutTask.GetAwaiter().GetResult();$stderr=$stderrTask.GetAwaiter().GetResult()
        if($process.ExitCode -ne 0){throw "$Engine $Mode round $Round failed ($($process.ExitCode)): $stderr"}
        $report=$stdout|ConvertFrom-Json
        if($report.schema_version -ne 1 -or $report.profile -ne 'canonical' -or $report.mode -ne $Mode -or $report.inputs.Count -eq 0 -or $report.warmup -ne $Warmup -or $report.samples -ne $Repeat){throw 'Invalid benchmark report'}
        if($Engine -eq 'rust-heap' -and $Mode -ne 'mixed_shapes' -and ($report.warm_allocation_count -ne 0 -or $report.warm_allocation_bytes -ne 0)){throw 'Prepared scalar calls must allocate zero times and zero bytes'}
        $report|Add-Member -NotePropertyName engine -NotePropertyValue $Engine
        $report|Add-Member -NotePropertyName round -NotePropertyValue $Round
        $report|Add-Member -NotePropertyName observed_process_peak_working_set_bytes -NotePropertyValue $observedPeak
        $report|Add-Member -NotePropertyName process_memory_semantics -NotePropertyValue 'Parent sampled child OS high-water resident memory; includes validation, model loading, corpus and runtime. Excludes no startup phase and can miss the final observation. Not a tensor or managed/native heap peak.'
        $report|ConvertTo-Json -Depth 30|Set-Content -LiteralPath (Join-Path $outputPath ("{0}-{1}-round-{2:D2}.json" -f $Mode,$Engine,$Round)) -Encoding utf8NoBOM
        Write-Host ('{0} {1} round {2}: p50={3:F3}ms p95={4:F3}ms' -f $Mode,$Engine,$Round,$report.corpus.p50_ms,$report.corpus.p95_ms)
        return $report
    } finally {$process.Dispose()}
}
$reports=[Collections.Generic.List[object]]::new();$orders=[Collections.Generic.List[object]]::new()
foreach($mode in $Modes){
    $first=$null
    for($round=1;$round -le $Rounds;$round++){
        $order=if($round%2 -eq 1){@('csharp','rust-timing')}else{@('rust-timing','csharp')}
        $orders.Add([pscustomobject]@{mode=$mode;round=$round;order=$order})
        foreach($engine in $order){
            $report=Invoke-Step12Run $mode $engine $round
            if($null -eq $first){$first=$report}else{
                if($report.model_sha256 -ne $first.model_sha256 -or $report.cls_model_sha256 -ne $first.cls_model_sha256 -or $report.manifest_sha256 -ne $first.manifest_sha256 -or $report.inputs.Count -ne $first.inputs.Count){throw 'Timing processes consumed different model/corpus identities'}
                if($mode -eq 'mixed_shapes' -and (ConvertTo-Json -InputObject $report.shape_sequence -Compress -Depth 3) -ne (ConvertTo-Json -InputObject $first.shape_sequence -Compress -Depth 3)){throw 'Mixed-shape execution sequences differ'}
                for($i=0;$i -lt $first.inputs.Count;$i++){
                    $a=$first.inputs[$i];$b=$report.inputs[$i]
                    if($a.case_id -ne $b.case_id -or $a.artifact_id -ne $b.artifact_id -or $a.sha256 -ne $b.sha256 -or $a.stage -ne $b.stage -or ($a.shape -join ',') -ne ($b.shape -join ',')){throw "Stage identity differs between engines: $mode input $i"}
                }
            }
            $reports.Add($report)
        }
    }
    for($round=1;$round -le $HeapRounds;$round++){$reports.Add((Invoke-Step12Run $mode 'rust-heap' $round))}
}
$summary=[ordered]@{
    schema_version=1;reference=$bundlePath;rounds=$Rounds;heap_rounds=$HeapRounds;warmup=$Warmup;samples=$Repeat;case=$Case;modes=$Modes
    process_orders=$orders.ToArray()
    measurement='Same frozen corpus/model identities, scalar CPU/NCHW, serial execution. Timing pairs alternate fresh-process order. Allocator instrumentation is separate; shape changes and growth are separately reported. No universal speed target.'
    environment=[ordered]@{os=[Runtime.InteropServices.RuntimeInformation]::OSDescription;architecture=[Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString();logical_processors=[Environment]::ProcessorCount;rustc=(& rustc -V);dotnet_sdk=(& dotnet --version);rustflags=$env:RUSTFLAGS;runtime_controls=$controls;source_git_head=$sourceHead;source_files_sha256=$sourceSha256;reference_git_head=$nativeGitHead;rust_timing_sha256=(Get-FileHash -LiteralPath $timingExe -Algorithm SHA256).Hash.ToLowerInvariant();rust_heap_sha256=(Get-FileHash -LiteralPath $heapExe -Algorithm SHA256).Hash.ToLowerInvariant();csharp_exporter_sha256=(Get-FileHash -LiteralPath $exporterDll -Algorithm SHA256).Hash.ToLowerInvariant();csharp_native_sha256=(Get-FileHash -LiteralPath $nativeDll -Algorithm SHA256).Hash.ToLowerInvariant()}
    runs=@($reports|ForEach-Object{[ordered]@{mode=$_.mode;engine=$_.engine;round=$_.round;load_ms=$_.load_ms;prepare_ms=$_.prepare_ms;p50_ms=$_.corpus.p50_ms;p95_ms=$_.corpus.p95_ms;warm_allocation_count=$_.warm_allocation_count;warm_allocation_bytes=$_.warm_allocation_bytes;activation_bytes=$_.activation_bytes;retained_activation_bytes=$_.retained_activation_bytes;observed_process_peak_working_set_bytes=$_.observed_process_peak_working_set_bytes}})
}
$summary|ConvertTo-Json -Depth 15|Set-Content -LiteralPath (Join-Path $outputPath 'summary.json') -Encoding utf8NoBOM
Write-Host "Saved timing samples and memory reports: $outputPath"
