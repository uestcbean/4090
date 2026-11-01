$ErrorActionPreference = 'Stop'
$revision = 'b968826d9c46dd6066d109eabc6255188de91218'
$destination = 'D:\ai\models\Qwen3-8B\' + $revision
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$metadataPath = Join-Path $destination 'download-manifest.json'
& curl.exe --proxy http://127.0.0.1:10809 -fsSL --retry 3 --max-time 60 "https://huggingface.co/api/models/Qwen/Qwen3-8B/revision/${revision}?blobs=true" -o $metadataPath
if ($LASTEXITCODE -ne 0) { throw 'Model metadata download failed' }
$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
foreach ($entry in $metadata.siblings) {
    $name = $entry.rfilename
    if ($name -eq '.gitattributes') { continue }
    if ($name.Contains('/') -or $name.Contains('\')) { throw 'Unexpected nested model file' }
    $target = Join-Path $destination $name
    if ((Test-Path -LiteralPath $target) -and ((Get-Item -LiteralPath $target).Length -eq $entry.size)) {
        if (-not $entry.lfs -or ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower() -eq $entry.lfs.sha256)) {
            Write-Output "Verified existing: $name"
            continue
        }
        throw "Existing file checksum mismatch: $name"
    }
    Write-Output "Downloading: $name ($($entry.size) bytes)"
    & curl.exe --proxy http://127.0.0.1:10809 -fL -C - --retry 5 --retry-delay 3 --connect-timeout 30 "https://huggingface.co/Qwen/Qwen3-8B/resolve/$revision/${name}?download=true" -o $target
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $name" }
    if ((Get-Item -LiteralPath $target).Length -ne $entry.size) { throw "Size mismatch: $name" }
    if ($entry.lfs -and ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower() -ne $entry.lfs.sha256)) { throw "SHA256 mismatch: $name" }
}
Write-Output "MODEL_DOWNLOAD_VERIFIED $destination"
