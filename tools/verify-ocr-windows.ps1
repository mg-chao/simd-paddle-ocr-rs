[CmdletBinding()]
param([string]$Out='target/validation/windows-tiny-ocr-final-v1')
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$output=[IO.Path]::GetFullPath((Join-Path $root $Out))
if(Test-Path -LiteralPath $output){throw 'Validation output exists; choose a fresh identity'}
[IO.Directory]::CreateDirectory($output)|Out-Null
function Invoke-Gate([string]$name,[string]$exe,[string[]]$arguments){
    $start=[Diagnostics.ProcessStartInfo]::new($exe);$start.WorkingDirectory=$root;$start.UseShellExecute=$false;$start.CreateNoWindow=$true;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    foreach($arg in $arguments){$start.ArgumentList.Add($arg)}
    $process=[Diagnostics.Process]::new();$process.StartInfo=$start
    $started=[DateTimeOffset]::UtcNow
    if(-not $process.Start()){throw "$name failed to start"}
    $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync();$process.WaitForExit()
    $stdout.GetAwaiter().GetResult() | Set-Content -LiteralPath (Join-Path $output "$name.log") -Encoding utf8NoBOM
    $stderr.GetAwaiter().GetResult() | Set-Content -LiteralPath (Join-Path $output "$name.stderr.log") -Encoding utf8NoBOM
    if($process.ExitCode -ne 0){throw "$name failed; inspect $output"}
    [ordered]@{name=$name;exe=$exe;arguments=$arguments;started_at_utc=$started.ToString('o');completed_at_utc=[DateTimeOffset]::UtcNow.ToString('o');exit_code=$process.ExitCode}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $output "$name.command.json") -Encoding utf8NoBOM
    $process.Dispose();Write-Host "PASS $name"
}
function Source-Identity {
    $paths=@('Cargo.toml','Cargo.lock','rust-toolchain.toml')+@(Get-ChildItem -LiteralPath (Join-Path $root 'src'),(Join-Path $root 'tests') -Filter '*.rs' -File -Recurse | ForEach-Object {[IO.Path]::GetRelativePath($root,$_.FullName).Replace('\','/')})
    @($paths|Sort-Object -Unique|ForEach-Object {[ordered]@{path=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $root $_) -Algorithm SHA256).Hash.ToLowerInvariant();bytes=(Get-Item -LiteralPath (Join-Path $root $_)).Length}})
}
$sourceBefore=Source-Identity
$previous=@{}
foreach($name in @('SIMDOCR_IMPORT_REFERENCE','SIMDOCR_REC_REFERENCE','SIMDOCR_OCR_REFERENCE')){$previous[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try {
$env:SIMDOCR_IMPORT_REFERENCE=Join-Path $root 'target/step12-fixtures/imports'
$env:SIMDOCR_REC_REFERENCE=Join-Path $root 'target/step12-fixtures/tiny-rec'
$env:SIMDOCR_OCR_REFERENCE=Join-Path $root 'target/step12-fixtures/tiny-ocr'
Invoke-Gate 'fmt' 'cargo' @('fmt','--all','--','--check')
Invoke-Gate 'normal-tests' 'cargo' @('test','--all-features','--locked','--offline')
Invoke-Gate 'clippy' 'cargo' @('clippy','--all-features','--all-targets','--locked','--offline','--','-D','warnings')
Invoke-Gate 'python-tests' 'python' @('-m','unittest','discover','-s','tools/tests')
Invoke-Gate 'imports' 'cargo' @('test','--release','--locked','--offline','--test','model_import','all_shipped_models_match_authoritative_imports','--','--ignored')
Invoke-Gate 'detector-classifier-regression' 'cargo' @('test','--release','--locked','--offline','--test','step12_harness','compact_real_models_match_all_detector_and_pixel_stages','--','--ignored')
Invoke-Gate 'line-api-regression' 'cargo' @('test','--release','--locked','--offline','--test','line_pipeline','--','--ignored')
Invoke-Gate 'rec-ci' 'cargo' @('test','--release','--locked','--offline','--test','rec_harness','all_frozen_tiny_rec_lines_match_authoritative_native_reference','--','--ignored')
Invoke-Gate 'ocr-ci-and-corruption' 'cargo' @('test','--release','--locked','--offline','--test','ocr_harness','--','--ignored')
Invoke-Gate 'cli-build' 'cargo' @('build','--release','--locked','--offline','--bin','simd-paddle-ocr-rs')
$binary=Join-Path $output 'simd-paddle-ocr-rs.exe'
Copy-Item -LiteralPath (Join-Path $root 'target/release/simd-paddle-ocr-rs.exe') -Destination $binary
foreach($scope in @('full','compact')){
    $reference=Join-Path $root $(if($scope -eq 'full'){'reference-bundles/tiny-ocr-allnodes-v1'}else{'reference-bundles/tiny-ocr-compact-v1'})
    foreach($stage in @('rec','ocr')){
        $candidate=Join-Path $output "$scope-$stage-candidate"
        Invoke-Gate "$scope-$stage-run" $binary @("run-$stage",$reference,'--out',$candidate,'--json')
        Invoke-Gate "$scope-$stage-compare" $binary @("compare-$stage",$reference,$candidate,'--json')
    }
}
$sourceAfter=Source-Identity
if(($sourceBefore|ConvertTo-Json -Depth 4 -Compress) -ne ($sourceAfter|ConvertTo-Json -Depth 4 -Compress)){throw 'Library/test source changed during verification; use a fresh stable run'}
$logs=@(Get-ChildItem -LiteralPath $output -File|ForEach-Object {[ordered]@{path=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant();bytes=$_.Length}})
[ordered]@{schema_version=1;status='all_native_windows_gates_passed';os=[System.Runtime.InteropServices.RuntimeInformation]::OSDescription;rustc=(& rustc -Vv);source_files=$sourceAfter;binary_sha256=(Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant();files=$logs}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $output 'proof.json') -Encoding utf8NoBOM
Write-Host "Saved native Windows verification: $output"
} finally {
    foreach($name in $previous.Keys){if($null -eq $previous[$name]){Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue}else{[Environment]::SetEnvironmentVariable($name,$previous[$name],'Process')}}
}
