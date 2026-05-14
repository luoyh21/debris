<#
.SYNOPSIS
  Cloudflare Quick Tunnel manager for Streamlit (8501) and FastAPI (8502).

.DESCRIPTION
  start              : launch two cloudflared quick tunnels in the background,
                       wait ~10s, write URLs to logs/tunnels.json + Desktop,
                       and print them.
  stop               : kill all cloudflared.exe processes started from this repo.
  status             : list running tunnel processes + the latest public URLs.
  logs               : tail the last 15 lines of each tunnel log.
  restart            : stop + start.
  install-autostart  : copy a silent launcher .vbs into shell:Startup so the
                       tunnels run automatically every time you log in.
  uninstall-autostart: remove the launcher from shell:Startup.

  Quick Tunnels need no Cloudflare account, but URLs change every restart
  (i.e. only when cloudflared crashes or you reboot). Combined with autostart,
  the typical URL stays alive for days/weeks at a time.

.EXAMPLE
  .\scripts\tunnels.ps1 start
  .\scripts\tunnels.ps1 status
  .\scripts\tunnels.ps1 install-autostart
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "status", "logs", "restart",
                  "install-autostart", "uninstall-autostart")]
    [string]$Action = "status"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$CfdExe   = Join-Path $RepoRoot "tools\cloudflared.exe"
$LogDir   = Join-Path $RepoRoot "logs"
$Tunnels  = @(
    @{ Port = 8501; Label = "Streamlit"; Log = "tunnel_8501.log" }
    @{ Port = 8502; Label = "FastAPI";   Log = "tunnel_8502.log" }
)
$TunnelsJson = Join-Path $RepoRoot "logs\tunnels.json"
$DesktopTxt  = Join-Path ([Environment]::GetFolderPath('Desktop')) "Debris-Tunnels.txt"
$VbsPath     = Join-Path $RepoRoot "scripts\tunnels_autostart.vbs"
$StartupLnk  = Join-Path ([Environment]::GetFolderPath('Startup')) "DebrisTunnels.vbs"
$NamedEnv    = Join-Path $RepoRoot ".env.tunnel.local"
$NamedLog    = Join-Path $LogDir   "tunnel_named.log"

function Read-NamedEnv {
    if (-not (Test-Path $NamedEnv)) { return $null }
    $kv = @{}
    Get-Content $NamedEnv | Where-Object { $_ -match "^[A-Z]" } | ForEach-Object {
        $p = $_.Split("=", 2)
        if ($p.Count -eq 2) { $kv[$p[0]] = $p[1] }
    }
    if ($kv.ContainsKey("CF_TUNNEL_TOKEN")) { return $kv } else { return $null }
}

function Ensure-Tools {
    if (-not (Test-Path $CfdExe)) {
        Write-Host "Downloading cloudflared.exe ..." -ForegroundColor Yellow
        $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        New-Item -ItemType Directory -Force (Split-Path $CfdExe) | Out-Null
        Invoke-WebRequest -Uri $url -OutFile $CfdExe -UseBasicParsing -TimeoutSec 120
    }
    if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory $LogDir | Out-Null }
}

function Get-TunnelProcs {
    Get-WmiObject Win32_Process -Filter "Name='cloudflared.exe'" |
        Where-Object { $_.CommandLine -like "*$CfdExe*" }
}

function Stop-Tunnels {
    $procs = Get-TunnelProcs
    if (-not $procs) {
        Write-Host "No active cloudflared processes from this repo." -ForegroundColor Gray
        return
    }
    foreach ($p in $procs) {
        Write-Host ("Stopping PID {0}" -f $p.ProcessId) -ForegroundColor Yellow
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep 1
}

function Get-PublicUrl([string]$LogPath) {
    if (-not (Test-Path $LogPath)) { return $null }
    $hit = Get-Content $LogPath -ErrorAction SilentlyContinue |
            Select-String -Pattern "https://[\w\-]+\.trycloudflare\.com" |
            Select-Object -Last 1
    if ($hit -and $hit.Matches[0].Value) { return $hit.Matches[0].Value }
    return $null
}

function Show-Status {
    $procs = Get-TunnelProcs
    Write-Host ""
    Write-Host "===================== Tunnel status =====================" -ForegroundColor Cyan
    foreach ($t in $Tunnels) {
        $proc = $procs | Where-Object { $_.CommandLine -like "*localhost:$($t.Port)*" } | Select-Object -First 1
        $url  = Get-PublicUrl (Join-Path $LogDir $t.Log)
        $pid_ = if ($proc) { "PID $($proc.ProcessId)" } else { "stopped" }
        $shown_url = if ($url) { $url } else { "-" }
        Write-Host ("  [{0,4}] {1,-10} {2,-10}  -> {3}" -f $t.Port, $t.Label, $pid_, $shown_url) -ForegroundColor White
    }
    Write-Host "=========================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Persist-Urls {
    # 写 logs/tunnels.json（JSON）+ 桌面文本文件
    $payload = @{
        generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
        tunnels      = @()
    }
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $deskLines = @(
        "Space Debris System - Public Tunnel URLs (trycloudflare quick tunnel)",
        "Generated at: $now",
        ""
    )
    foreach ($t in $Tunnels) {
        $url = Get-PublicUrl (Join-Path $LogDir $t.Log)
        $payload.tunnels += @{
            port  = $t.Port
            label = $t.Label
            url   = $url
        }
        if ($url) {
            $deskLines += "[$($t.Port)] $($t.Label.PadRight(10))  $url"
        } else {
            $deskLines += "[$($t.Port)] $($t.Label.PadRight(10))  (not ready yet)"
        }
    }
    $deskLines += ""
    $deskLines += "Note: trycloudflare URLs change whenever cloudflared restarts."
    $deskLines += "      The URL stays stable as long as the cloudflared process keeps running"
    $deskLines += "      (typically days to weeks unless you reboot the machine)."
    $payload | ConvertTo-Json -Depth 5 | Out-File -FilePath $TunnelsJson -Encoding UTF8
    $deskLines -join "`r`n" | Out-File -FilePath $DesktopTxt -Encoding UTF8
}

function Start-NamedTunnel($env_) {
    $token = $env_["CF_TUNNEL_TOKEN"]
    $hostUI  = $env_["CF_TUNNEL_HOSTNAME_UI"]
    $hostAPI = $env_["CF_TUNNEL_HOSTNAME_API"]
    if (Test-Path $NamedLog) { Remove-Item $NamedLog -Force }
    $stdoutLog = Join-Path $LogDir "tunnel_named_stdout.log"
    if (Test-Path $stdoutLog) { Remove-Item $stdoutLog -Force }
    Start-Process -FilePath $CfdExe -WindowStyle Hidden `
        -ArgumentList @("tunnel", "--no-autoupdate", "run", "--token", $token) `
        -RedirectStandardError $NamedLog -RedirectStandardOutput $stdoutLog | Out-Null
    Write-Host "Started named tunnel (cloudflared)." -ForegroundColor Green
    Start-Sleep -Seconds 5

    $urlUI  = "https://$hostUI"
    $urlAPI = "https://$hostAPI"

    @{
        generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
        mode         = "named"
        tunnel_id    = $env_["CF_TUNNEL_ID"]
        tunnels      = @(
            @{ port = 8501; label = "Streamlit"; url = $urlUI  },
            @{ port = 8502; label = "FastAPI";   url = $urlAPI }
        )
    } | ConvertTo-Json -Depth 5 | Out-File -FilePath $TunnelsJson -Encoding UTF8

    @(
        "Space Debris System - Public Tunnel URLs (named tunnel, STABLE)",
        "Generated at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
        "",
        "[8501] Streamlit   $urlUI",
        "[8502] FastAPI     $urlAPI",
        "",
        "These URLs are PERMANENT (DNS-backed CNAMEs to cfargotunnel.com).",
        "They survive reboots, cloudflared restarts and tunnel reconnects.",
        "Just keep cloudflared.exe running (the autostart vbs does this for you)."
    ) -join "`r`n" | Out-File -FilePath $DesktopTxt -Encoding UTF8

    Write-Host ""
    Write-Host "===================== Named tunnel =====================" -ForegroundColor Green
    Write-Host ("  [8501] Streamlit  -> {0}" -f $urlUI)
    Write-Host ("  [8502] FastAPI    -> {0}" -f $urlAPI)
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host ("URLs saved to: {0}" -f $DesktopTxt) -ForegroundColor Gray
}

function Start-Tunnels {
    Ensure-Tools
    Stop-Tunnels

    $named = Read-NamedEnv
    if ($named) {
        Write-Host "Detected $NamedEnv -> launching named tunnel mode." -ForegroundColor Cyan
        Start-NamedTunnel $named
        return
    }

    Write-Host "No .env.tunnel.local found -> falling back to quick tunnels." -ForegroundColor Yellow
    foreach ($t in $Tunnels) {
        $log = Join-Path $LogDir $t.Log
        if (Test-Path $log) { Remove-Item $log -Force }
        $args_ = @("tunnel", "--url", "http://localhost:$($t.Port)",
                   "--logfile", $log, "--no-autoupdate")
        Start-Process -FilePath $CfdExe -ArgumentList $args_ -WindowStyle Hidden
        Write-Host ("Started tunnel [{0}] {1}" -f $t.Port, $t.Label) -ForegroundColor Green
    }
    Write-Host "Waiting for trycloudflare to assign URLs ..." -ForegroundColor Gray
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 1500
        $ready = $true
        foreach ($t in $Tunnels) {
            if (-not (Get-PublicUrl (Join-Path $LogDir $t.Log))) { $ready = $false; break }
        }
        if ($ready) { break }
    }
    Persist-Urls
    Show-Status
    Write-Host ("URLs were saved to:") -ForegroundColor Gray
    Write-Host ("  - $TunnelsJson") -ForegroundColor Gray
    Write-Host ("  - $DesktopTxt") -ForegroundColor Gray
}

function Install-Autostart {
    Ensure-Tools
    $named = Read-NamedEnv
    if ($named) {
        # Direct cloudflared run with token (no PowerShell wrapper needed).
        $token  = $named["CF_TUNNEL_TOKEN"]
        $vbsCmd = '"' + $CfdExe + '" tunnel --no-autoupdate run --token "' + $token + '" --logfile "' + $NamedLog + '"'
        @(
            "' Auto-generated for named tunnel (he-ting.com).",
            "' Silently launches the cloudflared tunnel on user login.",
            'Set ws = CreateObject("WScript.Shell")',
            "ws.Run ""$vbsCmd"", 0, False"
        ) -join "`r`n" | Out-File -FilePath $StartupLnk -Encoding ASCII
        Write-Host "Installed autostart entry (named tunnel mode):" -ForegroundColor Green
    } else {
        $ps1 = Join-Path $RepoRoot "scripts\tunnels.ps1"
        $vbsContent = @"
' Auto-generated by tunnels.ps1 install-autostart
' Silently launches the cloudflared quick tunnels on user login.
Set ws = CreateObject("WScript.Shell")
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & _
      "$ps1" & """ start"
ws.Run cmd, 0, False
"@
        $vbsContent | Out-File -FilePath $VbsPath -Encoding ASCII
        Copy-Item -Path $VbsPath -Destination $StartupLnk -Force
        Write-Host "Installed autostart entry (quick tunnel mode):" -ForegroundColor Green
    }
    Write-Host ("  $StartupLnk") -ForegroundColor Gray
    Write-Host "It will silently launch tunnels every time you log in to Windows." -ForegroundColor Gray
    Write-Host "Remove with: .\scripts\tunnels.ps1 uninstall-autostart" -ForegroundColor Gray
}

function Uninstall-Autostart {
    if (Test-Path $StartupLnk) {
        Remove-Item $StartupLnk -Force
        Write-Host "Removed: $StartupLnk" -ForegroundColor Yellow
    } else {
        Write-Host "Autostart entry not found (already removed?)." -ForegroundColor Gray
    }
    if (Test-Path $VbsPath) { Remove-Item $VbsPath -Force }
}

function Show-Logs {
    foreach ($t in $Tunnels) {
        $log = Join-Path $LogDir $t.Log
        if (-not (Test-Path $log)) { continue }
        Write-Host "===== $($t.Log) (port $($t.Port)) =====" -ForegroundColor Cyan
        Get-Content $log -Tail 15
        Write-Host ""
    }
}

switch ($Action) {
    "start"               { Start-Tunnels }
    "stop"                { Stop-Tunnels; Show-Status }
    "status"              { Show-Status }
    "logs"                { Show-Logs }
    "restart"             { Stop-Tunnels; Start-Tunnels }
    "install-autostart"   { Install-Autostart }
    "uninstall-autostart" { Uninstall-Autostart }
}
