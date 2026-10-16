param()
$ErrorActionPreference = 'Stop'
$data = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../tests/baseline'))
$manifest = Get-Content -LiteralPath (Join-Path $data 'manifest.json') -Raw | ConvertFrom-Json
foreach ($fixture in $manifest.fixtures | Where-Object transform -eq 'identity') {
    $destination = [IO.Path]::GetFullPath((Join-Path $data $fixture.file))
    if (-not $destination.StartsWith($data + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Fixture path escapes corpus directory: $($fixture.file)"
    }
    if (-not (Test-Path -LiteralPath $destination)) {
        New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($destination)) | Out-Null
        $temporary = $destination + '.download'
        try {
            Invoke-WebRequest -Uri $fixture.url -OutFile $temporary
            $actual = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne $fixture.sha256) { throw "Download checksum mismatch: $($fixture.id)" }
            Move-Item -LiteralPath $temporary -Destination $destination
        } finally {
            if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
        }
    }
    $actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $fixture.sha256) { throw "Corpus checksum mismatch: $($fixture.id)" }
    Write-Output "$($fixture.id): SHA-256 verified"
}
