[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Out,
    [Parameter(Mandatory=$true)][string[]]$References,
    [string]$ReferenceRepo=(Join-Path $PSScriptRoot '../../SimdPaddleOCR'),
    [switch]$Profiling,
    [string]$StopFile
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'benchmark-process.ps1')
. (Join-Path $PSScriptRoot 'benchmark-build.ps1')
$rustRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$nativeRoot=(Resolve-Path -LiteralPath $ReferenceRepo).Path
$outputPath=[IO.Path]::GetFullPath($Out)
if(Test-Path -LiteralPath $outputPath){throw 'Prepared build directory exists; choose a new identity'}
[IO.Directory]::CreateDirectory($outputPath)|Out-Null
$rawPath=Join-Path $outputPath 'raw-processes'
$manifestPath=Join-Path $outputPath 'build.json'
$stopPath=if($StopFile){[IO.Path]::GetFullPath($StopFile)}else{$null}
$timingFeatureNames=if($Profiling){@('profiling')}else{@()}
$heapFeatureNames=@('heap-profiler')+$(if($Profiling){@('profiling')}else{@()})
$build=[ordered]@{schema_version=1;kind='ocr_prepared_benchmark_build';status='preparing';started_at_utc=[DateTimeOffset]::UtcNow.ToString('o');completed_at_utc=$null;profiling=[bool]$Profiling;timing_features=@($timingFeatureNames);heap_features=@($heapFeatureNames);effective_build_flags=(Get-OcrBuildFlags);process_records=@();error=$null}
function Save-OcrPreparedBuild { Write-OcrRecordJson $manifestPath $build }
function Invoke-OcrPreparationBuild([string]$Executable,[string[]]$Arguments,[string]$Identity) {
    try {$run=Invoke-OcrRecordedProcess -RecordDirectory $rawPath -Identity $Identity -Executable $Executable -Arguments $Arguments -WorkingDirectory $rustRoot -StopFile $stopPath -Engine 'build'}
    finally {if(Test-Path -LiteralPath (Join-Path $rawPath "$Identity.record.json")){$build.process_records+=('raw-processes/'+$Identity+'.record.json');Save-OcrPreparedBuild}}
    if($run.Record.status -eq 'interrupted_before_start' -or $run.Record.stop_requested){throw [OperationCanceledException]::new('Prepared build interrupted; all completed build records retained')}
    if($run.Record.exit_code -ne 0){throw "Prepared build failed: $Identity; exact stdout/stderr retained"}
    return $run
}
function Copy-OcrPreparationSource([string]$Origin,[string]$Prefix,[object[]]$Entries) {
    foreach($entry in $Entries){
        $target=Join-Path (Join-Path $outputPath $Prefix) $entry.path
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))|Out-Null
        Copy-Item -LiteralPath (Join-Path $Origin $entry.path) -Destination $target
        $null=Resolve-OcrIdentityPath (Join-Path $outputPath $Prefix) $entry
    }
}
Save-OcrPreparedBuild
try {
    if($stopPath -and (Test-Path -LiteralPath $stopPath)){throw [OperationCanceledException]::new('Stop observed before benchmark preparation')}
    $build.source_files=@(Get-OcrSourceInventory $rustRoot)
    $build.reference_source_files=@(Get-OcrSourceInventory $nativeRoot -Native)
    $build.source_sha256=Get-OcrInventoryHash $build.source_files
    $build.reference_source_sha256=Get-OcrInventoryHash $build.reference_source_files
    $build.toolchain=Get-OcrBuildToolchain $rustRoot
    $build.references=@($References | ForEach-Object {Get-OcrReferenceIdentity (Resolve-Path -LiteralPath $_).Path})
    if($build.references.Count -eq 0 -or @($build.references.manifest_sha256|Sort-Object -Unique).Count -ne $build.references.Count){throw 'Prepared reference bindings must be nonempty and distinct'}
    Copy-OcrPreparationSource $rustRoot 'source' $build.source_files
    Copy-OcrPreparationSource $nativeRoot 'reference-source' $build.reference_source_files
    Push-Location $rustRoot
    try {
        $build.rust_git_head=(& git rev-parse HEAD)
        $build.reference_git_head=(& git -C $nativeRoot rev-parse HEAD)
        $build.rustc=(& rustc --version)
        $build.dotnet_sdk=(& dotnet --version)
        $timingFeatures=if($Profiling){@('--features','profiling')}else{@()}
        $timing=Invoke-OcrPreparationBuild 'cargo' (@('build','--release','--locked','--offline','--bin','ocr-benchmark','--message-format=json')+$timingFeatures) 'build-rust-timing'
        Copy-Item -LiteralPath (Get-OcrCargoExecutable $timing) -Destination (Join-Path $outputPath 'ocr-timing.exe')
        $heapFeatures=if($Profiling){'heap-profiler,profiling'}else{'heap-profiler'}
        $heap=Invoke-OcrPreparationBuild 'cargo' @('build','--release','--locked','--offline','--bin','ocr-benchmark','--features',$heapFeatures,'--message-format=json') 'build-rust-heap'
        Copy-Item -LiteralPath (Get-OcrCargoExecutable $heap) -Destination (Join-Path $outputPath 'ocr-heap.exe')
        $null=Invoke-OcrPreparationBuild 'dotnet' @('build',(Join-Path $PSScriptRoot 'reference-export/ReferenceExport.csproj'),'-c','Release',"-p:ReferenceRepo=$nativeRoot",'--nologo') 'build-csharp'
    } finally {Pop-Location}
    $nativeOutput=Join-Path $outputPath 'csharp'
    [IO.Directory]::CreateDirectory($nativeOutput)|Out-Null
    Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'reference-export/bin/Release/net10.0') -File | ForEach-Object {Copy-Item -LiteralPath $_.FullName -Destination $nativeOutput}
    $paths=@('ocr-timing.exe','ocr-heap.exe')+@(Get-ChildItem -LiteralPath $nativeOutput -File | ForEach-Object {'csharp/'+$_.Name})
    $build.executables=@($paths | ForEach-Object {
        $path=Join-Path $outputPath $_
        [ordered]@{path=$_;bytes=(Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
    Assert-OcrSameInventory $build.source_files @(Get-OcrSourceInventory $rustRoot) 'Production/tool source during build'
    Assert-OcrSameInventory $build.reference_source_files @(Get-OcrSourceInventory $nativeRoot -Native) 'Native source during build'
    $flags=Get-OcrBuildFlags
    if(($flags|ConvertTo-Json -Compress) -cne ($build.effective_build_flags|ConvertTo-Json -Compress)){throw 'Effective build flags changed during preparation'}
    $build.status='prepared';$build.completed_at_utc=[DateTimeOffset]::UtcNow.ToString('o');Save-OcrPreparedBuild
    foreach($reference in $References){$null=Assert-OcrPreparedBuild -ManifestPath $manifestPath -RustRoot $rustRoot -ReferenceRoot $nativeRoot -ReferenceBundle (Resolve-Path -LiteralPath $reference).Path -Profiling:$Profiling}
    Write-Output "Prepared one frozen timing/heap/native build: $manifestPath"
} catch {
    $build.status=if($_.Exception -is [OperationCanceledException]){'interrupted'}else{'failed'}
    $build.error=$_.Exception.ToString();$build.completed_at_utc=[DateTimeOffset]::UtcNow.ToString('o');Save-OcrPreparedBuild
    throw
}
