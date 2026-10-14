function Write-OcrRecordJson([string]$Path, [object]$Value) {
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 40), [Text.UTF8Encoding]::new($false))
}

function Invoke-OcrRecordedProcess {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)][string]$RecordDirectory,
        [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9_.-]+$')][string]$Identity,
        [Parameter(Mandatory=$true)][string]$Executable,
        [string[]]$Arguments=@(),
        [System.Collections.IDictionary]$EnvironmentOverrides=@{},
        [string]$WorkingDirectory,
        [object]$ProcessorAffinityMask=$null,
        [string]$StopFile,
        [string]$Mode,
        [string]$Engine,
        [int]$Round=0
    )
    $directory=[IO.Path]::GetFullPath($RecordDirectory)
    [IO.Directory]::CreateDirectory($directory)|Out-Null
    $stdoutPath=Join-Path $directory "$Identity.stdout"
    $stderrPath=Join-Path $directory "$Identity.stderr"
    $recordPath=Join-Path $directory "$Identity.record.json"
    foreach($path in @($stdoutPath,$stderrPath,$recordPath)) {
        if(Test-Path -LiteralPath $path){throw "Process identity already exists: $path"}
    }
    $record=[ordered]@{
        schema_version=1;kind='ocr_recorded_process';identity=$Identity;mode=$Mode;engine=$Engine;round=$Round
        executable=$Executable;arguments=@($Arguments);working_directory=$WorkingDirectory
        requested_at_utc=[DateTimeOffset]::UtcNow.ToString('o');started_at_utc=$null;completed_at_utc=$null
        process_id=$null;requested_affinity_mask=$ProcessorAffinityMask;observed_affinity_mask=$null
        affinity_applied=$false;exit_code=$null;status='not_started';error=$null
        stop_file=$StopFile;stop_observed_at_utc=$null;stop_requested=$false
        process_resident_high_water_bytes=0L;resident_sample_count=0
        resident_scope='Sampled whole-process working set, including validation/loading; an unsampled zero is unavailable, not zero memory.'
    }
    $process=[Diagnostics.Process]::new()
    $start=[Diagnostics.ProcessStartInfo]::new($Executable)
    $start.UseShellExecute=$false;$start.CreateNoWindow=$true
    $start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    if($WorkingDirectory){$start.WorkingDirectory=$WorkingDirectory}
    foreach($argument in $Arguments){$start.ArgumentList.Add($argument)}
    foreach($key in $EnvironmentOverrides.Keys) {
        if($null -eq $EnvironmentOverrides[$key]){$start.Environment.Remove($key)|Out-Null}
        else{$start.Environment[$key]=[string]$EnvironmentOverrides[$key]}
    }
    $process.StartInfo=$start
    $stdoutFile=$null;$stderrFile=$null;$stdoutCopy=$null;$stderrCopy=$null
    $started=$false;$failure=$null
    try {
        $stdoutFile=[IO.File]::Open($stdoutPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        $stderrFile=[IO.File]::Open($stderrPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        if($StopFile -and (Test-Path -LiteralPath $StopFile)) {
            $record.status='interrupted_before_start';$record.stop_requested=$true
            $record.stop_observed_at_utc=[DateTimeOffset]::UtcNow.ToString('o')
        } else {
            $record.status='launching'
            if(-not $process.Start()){throw 'Process failed to start'}
            $started=$true;$record.process_id=$process.Id
            $record.started_at_utc=[DateTimeOffset]::UtcNow.ToString('o')
            # Copy pipe bytes directly; decoding/reformatting never changes the raw evidence.
            $stdoutCopy=$process.StandardOutput.BaseStream.CopyToAsync($stdoutFile)
            $stderrCopy=$process.StandardError.BaseStream.CopyToAsync($stderrFile)
            if($null -ne $ProcessorAffinityMask) {
                $record.status='applying_affinity'
                $process.ProcessorAffinity=[IntPtr][long]$ProcessorAffinityMask
                $record.observed_affinity_mask=$process.ProcessorAffinity.ToInt64()
                if($record.observed_affinity_mask -ne [long]$ProcessorAffinityMask){throw 'Observed affinity differs from requested mask'}
                $record.affinity_applied=$true
            }
            $record.status='running'
            while(-not $process.WaitForExit(1000)) {
                if($stdoutCopy.IsFaulted){$stdoutCopy.GetAwaiter().GetResult()}
                if($stderrCopy.IsFaulted){$stderrCopy.GetAwaiter().GetResult()}
                try {
                    $process.Refresh()
                    $record.process_resident_high_water_bytes=[Math]::Max($record.process_resident_high_water_bytes,$process.WorkingSet64)
                    $record.resident_sample_count++
                } catch {}
                if($StopFile -and -not $record.stop_requested -and (Test-Path -LiteralPath $StopFile)) {
                    $record.stop_requested=$true;$record.stop_observed_at_utc=[DateTimeOffset]::UtcNow.ToString('o')
                }
            }
            $record.exit_code=$process.ExitCode
            $stdoutCopy.GetAwaiter().GetResult();$stderrCopy.GetAwaiter().GetResult()
            $record.status=if($record.exit_code -eq 0){'completed'}else{'exit_failed'}
            if($StopFile -and -not $record.stop_requested -and (Test-Path -LiteralPath $StopFile)) {
                $record.stop_requested=$true;$record.stop_observed_at_utc=[DateTimeOffset]::UtcNow.ToString('o')
            }
        }
    } catch {
        $failure=$_
        $record.error=$_.Exception.ToString()
        $record.status=if($record.status -eq 'applying_affinity'){'affinity_failed'}elseif(-not $started){'launch_failed'}else{'capture_failed'}
        if($started) {
            try {if(-not $process.HasExited){$process.Kill($true)};$process.WaitForExit();$record.exit_code=$process.ExitCode} catch {}
        }
    } finally {
        foreach($copy in @($stdoutCopy,$stderrCopy)) {
            if($null -ne $copy){try{$copy.GetAwaiter().GetResult()}catch{if(-not $record.error){$record.error=$_.Exception.ToString()}}}
        }
        if($null -ne $stdoutFile){$stdoutFile.Dispose()}
        if($null -ne $stderrFile){$stderrFile.Dispose()}
        $record.completed_at_utc=[DateTimeOffset]::UtcNow.ToString('o')
        foreach($item in @(@('raw_stdout',$stdoutPath),@('raw_stderr',$stderrPath))) {
            $path=$item[1]
            if(Test-Path -LiteralPath $path) {
                $record[$item[0]]=[ordered]@{path=[IO.Path]::GetFileName($path);bytes=(Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
            }
        }
        Write-OcrRecordJson $recordPath $record
        $process.Dispose()
    }
    if($null -ne $failure){throw $failure}
    return [pscustomobject]@{Record=$record;RecordPath=$recordPath;StdoutPath=$stdoutPath;StderrPath=$stderrPath}
}
