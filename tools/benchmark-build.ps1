# Shared identity contract for one frozen benchmark build and its independent runs.
function Get-OcrSourceInventory([string]$Root, [switch]$Native) {
    $paths=@(& git -C $Root ls-files --cached --others --exclude-standard)
    if($LASTEXITCODE -ne 0){throw 'Cannot enumerate benchmark source identity'}
    $paths=$paths | Where-Object {
        if($Native){$_ -match '\.(cs|csproj|props|targets)$' -or $_ -match '(^|/)(packages.lock|global)\.json$'}
        else{$_ -match '\.(rs|cs|csproj|props|targets|ps1|py|toml)$' -or $_ -eq 'Cargo.lock' -or $_ -match '(^|/)(packages.lock|global)\.json$'}
    } | Sort-Object -Unique
    return @($paths | ForEach-Object {
        $path=Join-Path $Root $_
        [ordered]@{path=$_;bytes=(Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
}

function Get-OcrInventoryHash([object[]]$Entries) {
    $encoded=[Text.Encoding]::UTF8.GetBytes((@($Entries | ForEach-Object {"$($_.path)`0$($_.sha256)`n"}) -join ''))
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($encoded)).ToLowerInvariant()
}

function Get-OcrCargoExecutable([object]$Run) {
    $artifacts=@(Get-Content -LiteralPath $Run.StdoutPath | ForEach-Object {$_|ConvertFrom-Json} | Where-Object {$_.reason -ceq 'compiler-artifact' -and $_.target.name -ceq 'ocr-benchmark' -and $_.executable})
    if($artifacts.Count -ne 1){throw 'Cargo did not identify exactly one benchmark executable'}
    return $artifacts[0].executable
}

function Get-OcrBuildFlags {
    $flags=[ordered]@{}
    foreach($name in @('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_RUSTFLAGS','RUSTC','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','RUSTUP_TOOLCHAIN','CARGO_BUILD_TARGET','CARGO_TARGET_DIR','CARGO_INCREMENTAL','CARGO_PROFILE_RELEASE_OPT_LEVEL','CARGO_PROFILE_RELEASE_LTO','CARGO_PROFILE_RELEASE_CODEGEN_UNITS','CARGO_PROFILE_RELEASE_DEBUG','CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS','CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS','CARGO_PROFILE_RELEASE_PANIC','CARGO_PROFILE_RELEASE_STRIP','CARGO_PROFILE_RELEASE_INCREMENTAL')) {
        $flags[$name]=[Environment]::GetEnvironmentVariable($name)
    }
    return $flags
}

function Get-OcrBuildToolchain([string]$RustRoot) {
    Push-Location $RustRoot
    try {
        $rustc=(& rustc --version);if($LASTEXITCODE -ne 0){throw 'Rust toolchain identity unavailable'}
        $sdk=(& dotnet --version);if($LASTEXITCODE -ne 0){throw '.NET SDK identity unavailable'}
        $runtimes=@(& dotnet --list-runtimes);if($LASTEXITCODE -ne 0){throw '.NET runtime identities unavailable'}
        return [ordered]@{rustc=$rustc;dotnet_sdk=$sdk;dotnet_runtimes=$runtimes}
    } finally {Pop-Location}
}

function Assert-OcrSameInventory([object[]]$Expected, [object[]]$Actual, [string]$Label) {
    if($Expected.Count -eq 0 -or $Expected.Count -ne $Actual.Count){throw "$Label inventory differs"}
    $seen=@{}
    for($index=0;$index -lt $Expected.Count;$index++) {
        $entry=$Expected[$index];$current=$Actual[$index]
        if($seen.ContainsKey($entry.path)){throw "Duplicate $Label identity"};$seen[$entry.path]=$true
        if($entry.path -cne $current.path -or $entry.bytes -ne $current.bytes -or $entry.sha256 -cne $current.sha256){throw "$Label identity differs: $($entry.path)"}
    }
}

function Resolve-OcrIdentityPath([string]$Root, [object]$Entry) {
    $relative=[string]$Entry.path
    if(-not $relative -or $relative.Contains('\') -or $relative.Contains(':') -or [IO.Path]::IsPathRooted($relative) -or @($relative.Split('/') | Where-Object {$_ -in @('','..','.')}).Count){throw 'Unsafe frozen identity path'}
    $base=[IO.Path]::GetFullPath($Root).TrimEnd([IO.Path]::DirectorySeparatorChar)+[IO.Path]::DirectorySeparatorChar
    $path=[IO.Path]::GetFullPath((Join-Path $Root $relative))
    if(-not $path.StartsWith($base,[StringComparison]::OrdinalIgnoreCase) -or -not (Test-Path -LiteralPath $path -PathType Leaf)){throw "Missing frozen identity: $relative"}
    if($Entry.sha256 -cnotmatch '^[0-9a-f]{64}$' -or $Entry.bytes -lt 0 -or (Get-Item -LiteralPath $path).Length -ne $Entry.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $Entry.sha256){throw "Corrupt frozen identity: $relative"}
    return $path
}

function Get-OcrReferenceIdentity([string]$Root) {
    $manifestPath=Join-Path $Root 'manifest.json'
    $manifest=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if($manifest.schema_version -ne 1 -or $manifest.contract.profile -cne 'canonical' -or $manifest.contract.capture_scope -cne 'ocr' -or $manifest.contract.model -notin @('tiny','small','medium')){throw 'Prepared build requires a shipped-model canonical full-OCR reference'}
    $roles=@($manifest.models | ForEach-Object {$_.id} | Sort-Object)
    if(($roles -join ',') -cne 'cls,det,dictionary,rec'){throw 'Full-OCR model roles differ'}
    foreach($asset in $manifest.models){$null=Resolve-OcrIdentityPath $Root $asset}
    return [ordered]@{manifest_bytes=(Get-Item -LiteralPath $manifestPath).Length;manifest_sha256=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant();model_family=$manifest.contract.model;contract=$manifest.contract;models=@($manifest.models)}
}

function Assert-OcrPreparedBuild {
    [CmdletBinding()]
    param([Parameter(Mandatory=$true)][string]$ManifestPath,[Parameter(Mandatory=$true)][string]$RustRoot,[Parameter(Mandatory=$true)][string]$ReferenceRoot,[Parameter(Mandatory=$true)][string]$ReferenceBundle,[switch]$Profiling)
    $manifestPath=(Resolve-Path -LiteralPath $ManifestPath).Path
    $root=[IO.Path]::GetDirectoryName($manifestPath)
    $build=Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if($build.schema_version -ne 1 -or $build.kind -cne 'ocr_prepared_benchmark_build' -or $build.status -cne 'prepared' -or $build.profiling -isnot [bool] -or $build.profiling -ne [bool]$Profiling){throw 'Frozen benchmark build contract/features differ'}
    $timingFeatures=if($Profiling){@('profiling')}else{@()}
    $heapFeatures=@('heap-profiler')+$(if($Profiling){@('profiling')}else{@()})
    if(@($build.timing_features).Count -ne $timingFeatures.Count -or (@($build.timing_features) -join ',') -cne ($timingFeatures -join ',') -or @($build.heap_features).Count -ne $heapFeatures.Count -or (@($build.heap_features) -join ',') -cne ($heapFeatures -join ',')){throw 'Frozen timing/heap feature identities differ'}
    $flags=Get-OcrBuildFlags
    $expectedKeys=@($build.effective_build_flags.PSObject.Properties.Name | Sort-Object)
    $actualKeys=@($flags.Keys | Sort-Object)
    if(($expectedKeys -join ',') -cne ($actualKeys -join ',')){throw 'Effective build flag set differs'}
    foreach($name in $flags.Keys){if($build.effective_build_flags.$name -cne $flags[$name]){throw "Effective build flag differs: $name"}}
    $toolchain=Get-OcrBuildToolchain $RustRoot
    if(($build.toolchain|ConvertTo-Json -Depth 5 -Compress) -cne ($toolchain|ConvertTo-Json -Depth 5 -Compress)){throw 'Frozen compiler/runtime identity differs'}
    $sources=@(Get-OcrSourceInventory $RustRoot);$nativeSources=@(Get-OcrSourceInventory $ReferenceRoot -Native)
    Assert-OcrSameInventory @($build.source_files) $sources 'Production/tool source'
    Assert-OcrSameInventory @($build.reference_source_files) $nativeSources 'Native source'
    if($build.source_sha256 -cne (Get-OcrInventoryHash $sources) -or $build.reference_source_sha256 -cne (Get-OcrInventoryHash $nativeSources)){throw 'Frozen source inventory digest differs'}
    foreach($entry in $build.source_files){$null=Resolve-OcrIdentityPath (Join-Path $root 'source') $entry}
    foreach($entry in $build.reference_source_files){$null=Resolve-OcrIdentityPath (Join-Path $root 'reference-source') $entry}
    $currentReference=Get-OcrReferenceIdentity $ReferenceBundle
    $bound=@($build.references | Where-Object {$_.manifest_sha256 -ceq $currentReference.manifest_sha256})
    if($bound.Count -ne 1 -or ($bound[0]|ConvertTo-Json -Depth 40 -Compress) -cne ($currentReference|ConvertTo-Json -Depth 40 -Compress)){throw 'Frozen model/dictionary/options/reference binding differs'}
    $seen=@{}
    foreach($entry in $build.executables){
        if($seen.ContainsKey($entry.path)){throw 'Duplicate frozen executable identity'};$seen[$entry.path]=$true
        $null=Resolve-OcrIdentityPath $root $entry
    }
    foreach($role in @('ocr-timing.exe','ocr-heap.exe','csharp/SimdPaddleOCR.ReferenceExport.dll','csharp/Sdcb.SimdPaddleOCR.dll','csharp/SimdPaddleOCR.ReferenceExport.runtimeconfig.json','csharp/SimdPaddleOCR.ReferenceExport.deps.json')) {
        if(-not $seen.ContainsKey($role)){throw "Missing frozen executable role: $role"}
    }
    return [pscustomobject]@{Root=$root;ManifestPath=$manifestPath;Manifest=$build;ManifestSha256=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant();SourceFiles=$sources;ReferenceSources=$nativeSources}
}

function Copy-OcrPreparedExecutables([object]$Prepared, [string]$Destination) {
    foreach($entry in $Prepared.Manifest.executables) {
        $origin=Resolve-OcrIdentityPath $Prepared.Root $entry
        $target=Join-Path $Destination $entry.path
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))|Out-Null
        if(Test-Path -LiteralPath $target){throw 'Frozen executable destination already exists'}
        Copy-Item -LiteralPath $origin -Destination $target
        $null=Resolve-OcrIdentityPath $Destination $entry
    }
}
