$ErrorActionPreference = "Continue"
$base = "https://spandh.dcs.shef.ac.uk/gridcorpus"
$videoDir = "C:\Projects\LipReader\training\raw\video"
$alignDir = "C:\Projects\LipReader\training\raw\align"
New-Item -ItemType Directory -Force -Path $videoDir | Out-Null
New-Item -ItemType Directory -Force -Path $alignDir | Out-Null

$speakers = 1..34 | Where-Object { $_ -ne 21 }
foreach ($s in $speakers) {
    $vf = Join-Path $videoDir "s$s.mpg_vcd.zip"
    $af = Join-Path $alignDir "s$s.tar"
    if (-not (Test-Path $vf)) {
        Write-Output "downloading video s$s ..."
        curl.exe -sL --retry 5 --retry-delay 3 -C - -o $vf "$base/s$s/video/s$s.mpg_vcd.zip"
    } else {
        Write-Output "video s$s already present"
    }
    if (-not (Test-Path $af)) {
        Write-Output "downloading align s$s ..."
        curl.exe -sL --retry 5 --retry-delay 3 -C - -o $af "$base/s$s/align/s$s.tar"
    }
}
Write-Output "ALL_DOWNLOADS_DONE"
