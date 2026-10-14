[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Reference,
    [Parameter(Mandatory=$true)][string]$Out,
    [string]$ReferenceRepo=(Join-Path $PSScriptRoot '../../SimdPaddleOCR'),
    [ValidateSet('rec_graph','rec_public_owned','rec_public_borrowed','ocr_owned','ocr_borrowed','mixed_widths','owned_copies','node_profile')][string[]]$Modes=@('rec_graph','rec_public_owned','rec_public_borrowed','ocr_owned','ocr_borrowed','mixed_widths','owned_copies'),
    [string]$Case,
    [ValidateRange(1,100)][int]$Rounds=6,
    [ValidateRange(0,100)][int]$HeapRounds=3,
    [ValidateRange(0,2147483647)][int]$Warmup=30,
    [ValidateRange(1,2147483647)][int]$Repeat=100
)
$ErrorActionPreference='Stop'
if(@($Modes | Select-Object -Unique).Count -ne $Modes.Count){throw 'Duplicate benchmark modes are not allowed'}
$rustRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$bundlePath=(Resolve-Path -LiteralPath $Reference).Path
$referencePath=(Resolve-Path -LiteralPath $ReferenceRepo).Path
$outputPath=[IO.Path]::GetFullPath($Out)
if(Test-Path -LiteralPath $outputPath){throw 'Measurement directory exists; choose a new identity.'}
[IO.Directory]::CreateDirectory($outputPath)|Out-Null
Push-Location $rustRoot
try {
    & cargo build --release --locked --offline --bin ocr-benchmark
    if($LASTEXITCODE -ne 0){throw 'Timing build failed'}
    $timingExe=Join-Path $outputPath 'ocr-timing.exe'
    Copy-Item -LiteralPath (Join-Path $rustRoot 'target/release/ocr-benchmark.exe') -Destination $timingExe
    & cargo build --release --locked --offline --bin ocr-benchmark --features heap-profiler
    if($LASTEXITCODE -ne 0){throw 'Heap build failed'}
    $heapExe=Join-Path $outputPath 'ocr-heap.exe'
    Copy-Item -LiteralPath (Join-Path $rustRoot 'target/release/ocr-benchmark.exe') -Destination $heapExe
    & dotnet build (Join-Path $PSScriptRoot 'reference-export/ReferenceExport.csproj') -c Release "-p:ReferenceRepo=$referencePath" --nologo
    if($LASTEXITCODE -ne 0){throw 'C# benchmark build failed'}
    $sourceFiles=@(& git ls-files --cached --others --exclude-standard | Where-Object {$_ -match '\.(rs|cs|csproj|props|targets|ps1|py|toml)$' -or $_ -eq 'Cargo.lock' -or $_ -match '(^|/)(packages.lock|global)\.json$'} | Sort-Object -Unique | ForEach-Object {
        [ordered]@{path=$_;bytes=(Get-Item -LiteralPath $_).Length;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
} finally {Pop-Location}
$exporterPath=Join-Path $PSScriptRoot 'reference-export/bin/Release/net10.0'
$snapshot=Join-Path $outputPath 'csharp'
[IO.Directory]::CreateDirectory($snapshot)|Out-Null
Get-ChildItem -LiteralPath $exporterPath -File | ForEach-Object {Copy-Item -LiteralPath $_.FullName -Destination $snapshot}
$dll=Join-Path $snapshot 'SimdPaddleOCR.ReferenceExport.dll'
$executables=@($timingExe,$heapExe)+@(Get-ChildItem -LiteralPath $snapshot -File | ForEach-Object {$_.FullName}) | ForEach-Object {[ordered]@{path=[IO.Path]::GetRelativePath($outputPath,$_).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant();bytes=(Get-Item -LiteralPath $_).Length}}
$metadata=[ordered]@{schema_version=1;kind='tiny_ocr_measurement_snapshot';reference=$bundlePath;manifest_sha256=(Get-FileHash -LiteralPath (Join-Path $bundlePath 'manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant();rounds=$Rounds;heap_rounds=$HeapRounds;warmup=$Warmup;samples=$Repeat;case=$Case;modes=$Modes;rust_git_head=(& git -C $rustRoot rev-parse HEAD);reference_git_head=(& git -C $referencePath rev-parse HEAD);rustc=(& rustc --version);os=[System.Runtime.InteropServices.RuntimeInformation]::OSDescription;architecture=[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString();processor_count=[Environment]::ProcessorCount;executables=$executables;source_files=$sourceFiles}
$metadata.rustflags=[Environment]::GetEnvironmentVariable('RUSTFLAGS')
foreach($file in $sourceFiles){
    $saved=Join-Path (Join-Path $outputPath 'source') $file.path
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($saved))|Out-Null
    Copy-Item -LiteralPath (Join-Path $rustRoot $file.path) -Destination $saved
    if((Get-FileHash -LiteralPath $saved -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256){throw 'Source changed during benchmark snapshot'}
}
$sourceIdentity=[Text.Encoding]::UTF8.GetBytes((@($sourceFiles|ForEach-Object {"$($_.path)`0$($_.sha256)`n"}) -join ''))
$metadata.source_sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($sourceIdentity)).ToLowerInvariant()
$controls=@{DOTNET_PROCESSOR_COUNT='1';DOTNET_EnableHWIntrinsic='0';DOTNET_TieredCompilation='0';DOTNET_TieredPGO='0';PPOCR_NHWC='0';SIMD_OCR_NOFUSE='1';PPOCR_DET_THREADS='1';PPOCR_DUMP_CONV=$null;PPOCR_DUMP_NODES=$null;PPOCR_DUMP_PLAN=$null;PPOCR_DUMP_GRAPH=$null}
$metadata.runtime_controls=$controls
$referenceSources=@(& git -C $referencePath ls-files --cached --others --exclude-standard | Where-Object {$_ -match '\.(cs|csproj|props|targets)$' -or $_ -match '(^|/)(packages.lock|global)\.json$'} | Sort-Object -Unique | ForEach-Object {
    $origin=Join-Path $referencePath $_
    [ordered]@{path=$_;bytes=(Get-Item -LiteralPath $origin).Length;sha256=(Get-FileHash -LiteralPath $origin -Algorithm SHA256).Hash.ToLowerInvariant()}
})
foreach($file in $referenceSources){
    $saved=Join-Path (Join-Path $outputPath 'reference-source') $file.path
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($saved))|Out-Null
    Copy-Item -LiteralPath (Join-Path $referencePath $file.path) -Destination $saved
    if((Get-FileHash -LiteralPath $saved -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256){throw 'Reference source changed during snapshot'}
}
$metadata.reference_source_files=$referenceSources
$referenceIdentity=[Text.Encoding]::UTF8.GetBytes((@($referenceSources|ForEach-Object {"$($_.path)`0$($_.sha256)`n"}) -join ''))
$metadata.reference_source_sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($referenceIdentity)).ToLowerInvariant()
$metadata|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $outputPath 'snapshot.json') -Encoding utf8NoBOM
function Invoke-OcrRun([string]$mode,[string]$engine,[int]$round) {
    $exe=if($engine -eq 'csharp'){'dotnet'}elseif($engine -eq 'rust-timing'){$timingExe}else{$heapExe}
    $start=[Diagnostics.ProcessStartInfo]::new($exe);$start.UseShellExecute=$false;$start.CreateNoWindow=$true;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    foreach($key in $controls.Keys){if($null -eq $controls[$key]){$start.Environment.Remove($key)|Out-Null}else{$start.Environment[$key]=$controls[$key]}}
    if($engine -eq 'csharp'){$start.ArgumentList.Add($dll);$start.ArgumentList.Add('--benchmark-ocr')}
    foreach($arg in @('--reference',$bundlePath,'--mode',$mode,'--warmup',"$Warmup",'--samples',"$Repeat")){$start.ArgumentList.Add($arg)}
    if($Case -and $mode -ne 'mixed_widths'){$start.ArgumentList.Add('--case');$start.ArgumentList.Add($Case)}
    $process=[Diagnostics.Process]::new();$process.StartInfo=$start
    if(-not $process.Start()){throw 'Benchmark process failed to start'}
    $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync();$resident=0L
    while(-not $process.WaitForExit(1000)){try{$process.Refresh();$resident=[Math]::Max($resident,$process.WorkingSet64)}catch{}}
    $output=$stdout.GetAwaiter().GetResult();$errorText=$stderr.GetAwaiter().GetResult()
    if($process.ExitCode -ne 0){throw "$engine $mode round $round failed: $errorText"}
    $report=$output|ConvertFrom-Json
    $report|Add-Member -NotePropertyName process_resident_high_water_bytes -NotePropertyValue $resident
    $report|Add-Member -NotePropertyName round -NotePropertyValue $round
    $report|Add-Member -NotePropertyName executable_kind -NotePropertyValue $engine
    $path=Join-Path $outputPath "$mode-$engine-r$round.json"
    $report|ConvertTo-Json -Depth 40|Set-Content -LiteralPath $path -Encoding utf8NoBOM
    Write-Host "$mode $engine round $round p50=$($report.corpus.p50_ms) p95=$($report.corpus.p95_ms)"
    $process.Dispose()
}
foreach($mode in $Modes) {
    $paired=$mode -in @('rec_graph','rec_public_owned','ocr_owned')
    for($round=1;$round -le $Rounds;$round++) {
        if($paired -and $round%2 -eq 1){Invoke-OcrRun $mode 'csharp' $round}
        Invoke-OcrRun $mode 'rust-timing' $round
        if($paired -and $round%2 -eq 0){Invoke-OcrRun $mode 'csharp' $round}
    }
    for($round=1;$round -le $HeapRounds;$round++){Invoke-OcrRun $mode 'rust-heap' $round}
}
Write-Host "Saved independently scoped tiny OCR measurements: $outputPath"
