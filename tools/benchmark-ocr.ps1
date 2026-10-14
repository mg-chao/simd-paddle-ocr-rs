[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Reference,
    [Parameter(Mandatory=$true)][string]$Out,
    [string]$ReferenceRepo=(Join-Path $PSScriptRoot '../../SimdPaddleOCR'),
    [string]$StopFile,
    [string]$IdleProvenanceFile,
    [ValidateSet('rec_graph','rec_public_owned','rec_public_borrowed','ocr_owned','ocr_borrowed','mixed_widths','owned_copies','node_profile','public_stage_profile')][string[]]$Modes=@('rec_graph','rec_public_owned','rec_public_borrowed','ocr_owned','ocr_borrowed','mixed_widths','owned_copies'),
    [ValidateSet('canonical','cpu')][string]$CSharpProfile='canonical',
    [ValidateRange(1,2147483647)][long]$ProcessorAffinityMask=1,
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
$bundleManifest=Get-Content -LiteralPath (Join-Path $bundlePath 'manifest.json') -Raw | ConvertFrom-Json
if($bundleManifest.contract.profile -ne 'canonical' -or $bundleManifest.contract.model -notin @('tiny','small','medium')){throw 'Requires a shipped-model canonical reference'}
$referencePath=(Resolve-Path -LiteralPath $ReferenceRepo).Path
$outputPath=[IO.Path]::GetFullPath($Out)
if(Test-Path -LiteralPath $outputPath){throw 'Measurement directory exists; choose a new identity.'}
[IO.Directory]::CreateDirectory($outputPath)|Out-Null
. (Join-Path $PSScriptRoot 'benchmark-process.ps1')
$rawProcessPath=Join-Path $outputPath 'raw-processes'
[IO.Directory]::CreateDirectory($rawProcessPath)|Out-Null
$stopFilePath=if($StopFile){[IO.Path]::GetFullPath($StopFile)}else{$null}
$series=[ordered]@{schema_version=1;kind='ocr_measurement_series';status='preparing';started_at_utc=[DateTimeOffset]::UtcNow.ToString('o');completed_at_utc=$null;stop_file=$stopFilePath;process_records=@();completed_reports=@();error=$null;idle_provenance_status='not_provided'}
function Save-OcrSeries { Write-OcrRecordJson (Join-Path $rawProcessPath 'series.json') $series }
function Stop-OcrSeriesIfRequested {
    if($stopFilePath -and (Test-Path -LiteralPath $stopFilePath)){throw [OperationCanceledException]::new('Stop file observed; completed records are preserved.')}
}
function Invoke-OcrBuild([string]$executable,[string[]]$arguments,[string]$identity) {
    try {
        $run=Invoke-OcrRecordedProcess -RecordDirectory $rawProcessPath -Identity $identity -Executable $executable -Arguments $arguments -WorkingDirectory $rustRoot -StopFile $stopFilePath -Engine 'build'
    } finally {
        $recordPath=Join-Path $rawProcessPath "$identity.record.json"
        if(Test-Path -LiteralPath $recordPath){$series.process_records+=('raw-processes/'+[IO.Path]::GetFileName($recordPath));Save-OcrSeries}
    }
    if($run.Record.status -eq 'interrupted_before_start'){throw [OperationCanceledException]::new('Stop file observed before build.')}
    if($run.Record.exit_code -ne 0){throw "Build failed; see $($run.RecordPath) and raw stdout/stderr."}
    if($run.Record.stop_requested){throw [OperationCanceledException]::new('Stop file observed during build; completed raw records are preserved.')}
}
Save-OcrSeries
try {
Stop-OcrSeriesIfRequested
$idleProvenance=$null
if($IdleProvenanceFile) {
    $origin=(Resolve-Path -LiteralPath $IdleProvenanceFile).Path
    $saved=Join-Path $rawProcessPath 'idle-provenance.json'
    Copy-Item -LiteralPath $origin -Destination $saved
    $idleProvenance=[ordered]@{path='raw-processes/idle-provenance.json';bytes=(Get-Item -LiteralPath $saved).Length;sha256=(Get-FileHash -LiteralPath $saved -Algorithm SHA256).Hash.ToLowerInvariant()}
    $series.idle_provenance_status='provided';$series.idle_provenance=$idleProvenance
}
Push-Location $rustRoot
try {
    $profilingArguments=if($Modes -contains 'public_stage_profile'){@('--features','profiling')}else{@()}
    Invoke-OcrBuild 'cargo' (@('build','--release','--locked','--offline','--bin','ocr-benchmark')+$profilingArguments) 'build-rust-timing'
    $timingExe=Join-Path $outputPath 'ocr-timing.exe'
    Copy-Item -LiteralPath (Join-Path $rustRoot 'target/release/ocr-benchmark.exe') -Destination $timingExe
    $heapFeatures=if($Modes -contains 'public_stage_profile'){'heap-profiler,profiling'}else{'heap-profiler'}
    Invoke-OcrBuild 'cargo' @('build','--release','--locked','--offline','--bin','ocr-benchmark','--features',$heapFeatures) 'build-rust-heap'
    $heapExe=Join-Path $outputPath 'ocr-heap.exe'
    Copy-Item -LiteralPath (Join-Path $rustRoot 'target/release/ocr-benchmark.exe') -Destination $heapExe
    Invoke-OcrBuild 'dotnet' @('build',(Join-Path $PSScriptRoot 'reference-export/ReferenceExport.csproj'),'-c','Release',"-p:ReferenceRepo=$referencePath",'--nologo') 'build-csharp'
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
$metadata=[ordered]@{schema_version=2;kind='ocr_measurement_snapshot';model_family=$bundleManifest.contract.model;csharp_profile=$CSharpProfile;processor_affinity_mask=$ProcessorAffinityMask;power_plan=(& powercfg /getactivescheme);reference=$bundlePath;manifest_sha256=(Get-FileHash -LiteralPath (Join-Path $bundlePath 'manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant();rounds=$Rounds;heap_rounds=$HeapRounds;warmup=$Warmup;samples=$Repeat;case=$Case;modes=$Modes;rust_git_head=(& git -C $rustRoot rev-parse HEAD);reference_git_head=(& git -C $referencePath rev-parse HEAD);rustc=(& rustc --version);os=[System.Runtime.InteropServices.RuntimeInformation]::OSDescription;architecture=[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString();processor_count=[Environment]::ProcessorCount;executables=$executables;source_files=$sourceFiles}
$metadata.rustflags=[Environment]::GetEnvironmentVariable('RUSTFLAGS')
$metadata.idle_provenance=$idleProvenance
$metadata.stop_file=$stopFilePath
foreach($file in $sourceFiles){
    $saved=Join-Path (Join-Path $outputPath 'source') $file.path
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($saved))|Out-Null
    Copy-Item -LiteralPath (Join-Path $rustRoot $file.path) -Destination $saved
    if((Get-FileHash -LiteralPath $saved -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256){throw 'Source changed during benchmark snapshot'}
}
$sourceIdentity=[Text.Encoding]::UTF8.GetBytes((@($sourceFiles|ForEach-Object {"$($_.path)`0$($_.sha256)`n"}) -join ''))
$metadata.source_sha256=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($sourceIdentity)).ToLowerInvariant()
$controls=@{DOTNET_PROCESSOR_COUNT='1';DOTNET_EnableHWIntrinsic='0';DOTNET_TieredCompilation='0';DOTNET_TieredPGO='0';PPOCR_NHWC='0';SIMD_OCR_NOFUSE='1';PPOCR_DET_THREADS='1';PPOCR_DUMP_CONV=$null;PPOCR_DUMP_NODES=$null;PPOCR_DUMP_PLAN=$null;PPOCR_DUMP_GRAPH=$null}
if($CSharpProfile -eq 'cpu'){$controls.DOTNET_EnableHWIntrinsic='1';$controls.DOTNET_TieredCompilation='1';$controls.DOTNET_TieredPGO='1';$controls.PPOCR_NHWC='1';$controls.SIMD_OCR_NOFUSE='0'}
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
    $validation=$engine -eq 'csharp-validation'
    $exe=if($engine -in @('csharp','csharp-validation')){'dotnet'}elseif($engine -eq 'rust-timing'){$timingExe}else{$heapExe}
    $arguments=@()
    if($engine -in @('csharp','csharp-validation')){
        $command=if($validation){'--validate-ocr-public'}else{'--benchmark-ocr'}
        $arguments+=@($dll,$command,'--profile',$CSharpProfile)
    }
    $arguments+=@('--reference',$bundlePath,'--mode',$mode,'--warmup',"$Warmup")
    if(-not $validation){$arguments+=@('--samples',"$Repeat")}
    if($Case -and $mode -ne 'mixed_widths'){$arguments+=@('--case',$Case)}
    $identity="$mode-$engine-r$round"
    try {
        $run=Invoke-OcrRecordedProcess -RecordDirectory $rawProcessPath -Identity $identity -Executable $exe -Arguments $arguments -EnvironmentOverrides $controls -WorkingDirectory $rustRoot -ProcessorAffinityMask $ProcessorAffinityMask -StopFile $stopFilePath -Mode $mode -Engine $engine -Round $round
    } finally {
        $recordPath=Join-Path $rawProcessPath "$identity.record.json"
        if(Test-Path -LiteralPath $recordPath){$series.process_records+=('raw-processes/'+[IO.Path]::GetFileName($recordPath));Save-OcrSeries}
    }
    if($run.Record.status -eq 'interrupted_before_start'){throw [OperationCanceledException]::new('Stop file observed before next independent process.')}
    if($run.Record.exit_code -ne 0){throw "$engine $mode round $round failed; see $($run.RecordPath) and its raw payloads."}
    $report=Get-Content -LiteralPath $run.StdoutPath -Raw -Encoding utf8|ConvertFrom-Json
    if($validation) {
        if($report.kind -ne 'ocr_public_semantic_validation' -or $report.status -ne 'passed' -or $report.model_family -ne $bundleManifest.contract.model -or $report.profile -ne $CSharpProfile -or $report.warmup -ne $Warmup -or $report.manifest_sha256 -ne $metadata.manifest_sha256){throw 'Public semantic preflight identity differs'}
        $series.public_validation=[ordered]@{record='raw-processes/'+[IO.Path]::GetFileName($run.RecordPath);status='passed';mode=$mode;profile=$CSharpProfile;warmup=$Warmup;recognition_checks=$report.recognition_checks;ocr_checks=$report.ocr_checks}
        Save-OcrSeries
        if($run.Record.stop_requested){throw [OperationCanceledException]::new('Stop file observed during public preflight.')}
        return
    }
    $report|Add-Member -NotePropertyName process_resident_high_water_bytes -NotePropertyValue $run.Record.process_resident_high_water_bytes
    $report|Add-Member -NotePropertyName round -NotePropertyValue $round
    $report|Add-Member -NotePropertyName executable_kind -NotePropertyValue $engine
    $path=Join-Path $outputPath "$mode-$engine-r$round.json"
    $report|ConvertTo-Json -Depth 40|Set-Content -LiteralPath $path -Encoding utf8NoBOM
    $series.completed_reports+=([ordered]@{path=[IO.Path]::GetFileName($path);bytes=(Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()});Save-OcrSeries
    Write-Host "$mode $engine round $round p50=$($report.corpus.p50_ms) p95=$($report.corpus.p95_ms)"
    if($run.Record.stop_requested){throw [OperationCanceledException]::new('Stop file observed during active process; its completed record is preserved.')}
}
$series.status='validating_public';Save-OcrSeries
$recPublic=$Modes -contains 'rec_public_owned'
$ocrPublic=$Modes -contains 'ocr_owned'
if($recPublic -or $ocrPublic) {
    $validationMode=if($recPublic -and $ocrPublic){'both'}elseif($recPublic){'rec_public_owned'}else{'ocr_owned'}
    Invoke-OcrRun $validationMode 'csharp-validation' 0
}
$series.status='measuring';Save-OcrSeries
foreach($mode in $Modes) {
    $paired=$mode -in @('rec_graph','rec_public_owned','ocr_owned')
    for($round=1;$round -le $Rounds;$round++) {
        if($paired -and $round%2 -eq 1){Invoke-OcrRun $mode 'csharp' $round}
        Invoke-OcrRun $mode 'rust-timing' $round
        if($paired -and $round%2 -eq 0){Invoke-OcrRun $mode 'csharp' $round}
    }
    for($round=1;$round -le $HeapRounds;$round++){Invoke-OcrRun $mode 'rust-heap' $round}
}
$series.status='completed';$series.completed_at_utc=[DateTimeOffset]::UtcNow.ToString('o');Save-OcrSeries
Write-Host "Saved independently scoped OCR measurements: $outputPath"
} catch {
    $series.status=if($_.Exception -is [OperationCanceledException]){'interrupted'}else{'failed'}
    $series.error=$_.Exception.ToString();$series.completed_at_utc=[DateTimeOffset]::UtcNow.ToString('o');Save-OcrSeries
    throw
}
