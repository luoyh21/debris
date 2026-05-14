<#
.SYNOPSIS
  Switch from Quick Tunnel to a Cloudflare Named Tunnel under your own domain.

.DESCRIPTION
  One-shot script that:
    1. Creates / reuses one named tunnel (default: "debris-system")
    2. Configures ingress: 8501 -> ${SubUI}.${Zone}, 8502 -> ${SubAPI}.${Zone}
    3. Creates / updates two DNS CNAME records pointing at the tunnel
    4. Saves the tunnel token to .env.tunnel.local (gitignored)
    5. Updates tunnels.ps1 to launch the named tunnel via 'cloudflared tunnel run --token ...'
    6. Refreshes the autostart vbs and writes the final URLs to:
         logs/tunnels.json
         Desktop\Debris-Tunnels.txt

.EXAMPLE
  .\scripts\setup_named_tunnel.ps1 -Token "<NEW_API_TOKEN>"
  .\scripts\setup_named_tunnel.ps1 -Token $env:CF_TOKEN -Zone he-ting.com -SubUI debris-ui -SubAPI debris-api
#>
param(
    [Parameter(Mandatory = $true)] [string]$Token,
    [string]$Zone     = "he-ting.com",
    [string]$SubUI    = "debris-ui",
    [string]$SubAPI   = "debris-api",
    [string]$TunnelName = "debris-system"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ToolsDir = Join-Path $RepoRoot "tools"
$CfdExe   = Join-Path $ToolsDir "cloudflared.exe"
$EnvLocal = Join-Path $RepoRoot ".env.tunnel.local"

if (-not (Test-Path $CfdExe)) {
    throw "cloudflared.exe not found at $CfdExe. Run tunnels.ps1 start first to download it."
}

$h = @{ Authorization = "Bearer $Token"; "Content-Type" = "application/json" }
$base = "https://api.cloudflare.com/client/v4"

# Note: under PS 5.1 wrapping Invoke-RestMethod inside a function and piping
# `(Cf-Get ...).result | Select-Object -First 1` sometimes returns $null for
# array results. Use Invoke-RestMethod directly inline instead.
function Cf-Post($path,$b)  { Invoke-RestMethod -Method Post -Uri "$base$path" -Headers $h -Body ($b | ConvertTo-Json -Depth 10 -Compress) }
function Cf-Put($path,$b)   { Invoke-RestMethod -Method Put  -Uri "$base$path" -Headers $h -Body ($b | ConvertTo-Json -Depth 10 -Compress) }
function Cf-Delete($path)   { Invoke-RestMethod -Method Delete -Uri "$base$path" -Headers $h }

# ---- 1. Resolve account_id + zone_id -----------------------------------------
Write-Host "[1/6] Resolving account / zone ..." -ForegroundColor Cyan
$zoneResp = Invoke-RestMethod -Method Get -Uri "$base/zones?name=$Zone" -Headers $h
$zoneList = @($zoneResp.result)
if ($zoneList.Count -eq 0) { throw "Zone $Zone not found under this token. Make sure DNS is hosted on Cloudflare and the token has Zone:Read on this zone." }
$zone      = $zoneList[0]
$accountId = $zone.account.id
$zoneId    = $zone.id
if (-not $accountId) { throw "Could not read account.id from zone payload (token may lack Account permissions)." }
Write-Host "    account_id = $accountId"
Write-Host "    zone_id    = $zoneId"

# ---- 2. Create or reuse named tunnel -----------------------------------------
Write-Host "[2/6] Creating named tunnel '$TunnelName' ..." -ForegroundColor Cyan
$tunResp  = Invoke-RestMethod -Method Get -Uri "$base/accounts/$accountId/cfd_tunnel?name=$TunnelName&is_deleted=false" -Headers $h
$tunList  = @($tunResp.result)
if ($tunList.Count -gt 0) {
    Write-Host "    Found existing tunnel id=$($tunList[0].id), reusing." -ForegroundColor Yellow
    $tunnelId = $tunList[0].id
} else {
    $bytes = New-Object byte[] 32
    (New-Object System.Security.Cryptography.RNGCryptoServiceProvider).GetBytes($bytes)
    $secret = [Convert]::ToBase64String($bytes)
    $resp = Cf-Post "/accounts/$accountId/cfd_tunnel" @{ name = $TunnelName; tunnel_secret = $secret; config_src = "cloudflare" }
    $tunnelId = $resp.result.id
    Write-Host "    Created tunnel id=$tunnelId" -ForegroundColor Green
}

# ---- 3. Get tunnel token (used by `cloudflared tunnel run --token ...`) ------
Write-Host "[3/6] Fetching tunnel run-token ..." -ForegroundColor Cyan
$tunnelToken = (Invoke-RestMethod -Method Get -Uri "$base/accounts/$accountId/cfd_tunnel/$tunnelId/token" -Headers $h).result
if (-not $tunnelToken) { throw "Failed to fetch tunnel token." }

# ---- 4. Push ingress configuration -------------------------------------------
Write-Host "[4/6] Writing ingress rules ..." -ForegroundColor Cyan
$cfg = @{
    config = @{
        ingress = @(
            @{ hostname = "$SubUI.$Zone";  service = "http://localhost:8501" },
            @{ hostname = "$SubAPI.$Zone"; service = "http://localhost:8502" },
            @{ service  = "http_status:404" }
        )
    }
}
Cf-Put "/accounts/$accountId/cfd_tunnel/$tunnelId/configurations" $cfg | Out-Null
Write-Host "    -> $SubUI.$Zone   => http://localhost:8501"
Write-Host "    -> $SubAPI.$Zone  => http://localhost:8502"

# ---- 5. Create / update DNS CNAME records ------------------------------------
Write-Host "[5/6] Upserting DNS CNAME records ..." -ForegroundColor Cyan
$target = "$tunnelId.cfargotunnel.com"
foreach ($sub in @($SubUI, $SubAPI)) {
    $fqdn   = "$sub.$Zone"
    $recResp = Invoke-RestMethod -Method Get -Uri "$base/zones/$zoneId/dns_records?type=CNAME&name=$fqdn" -Headers $h
    $recList = @($recResp.result)
    $payload = @{ type = "CNAME"; name = $sub; content = $target; proxied = $true; ttl = 1 }
    if ($recList.Count -gt 0) {
        Cf-Put "/zones/$zoneId/dns_records/$($recList[0].id)" $payload | Out-Null
        Write-Host "    updated CNAME $fqdn -> $target"
    } else {
        Cf-Post "/zones/$zoneId/dns_records" $payload | Out-Null
        Write-Host "    created CNAME $fqdn -> $target"
    }
}

# ---- 6. Persist token + URLs + refresh autostart -----------------------------
Write-Host "[6/6] Saving credentials and refreshing scripts ..." -ForegroundColor Cyan
@(
    "# Auto-generated by setup_named_tunnel.ps1; DO NOT COMMIT.",
    "CF_TUNNEL_ID=$tunnelId",
    "CF_TUNNEL_TOKEN=$tunnelToken",
    "CF_TUNNEL_HOSTNAME_UI=$SubUI.$Zone",
    "CF_TUNNEL_HOSTNAME_API=$SubAPI.$Zone"
) -join "`r`n" | Out-File -FilePath $EnvLocal -Encoding ASCII

# Stop quick tunnels and any existing named tunnel processes
Get-WmiObject Win32_Process -Filter "Name='cloudflared.exe'" |
    ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {} }

# Start the named tunnel in background
$logDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$namedLog = Join-Path $logDir "tunnel_named.log"
if (Test-Path $namedLog) { Remove-Item $namedLog -Force }
Start-Process -FilePath $CfdExe -WindowStyle Hidden -ArgumentList @(
    "tunnel", "--no-autoupdate", "run",
    "--token", $tunnelToken,
    "--logfile", $namedLog
)
Write-Host "    cloudflared started in background." -ForegroundColor Green

Start-Sleep -Seconds 4

# Write final URL outputs
$urlUI  = "https://$SubUI.$Zone"
$urlAPI = "https://$SubAPI.$Zone"
$tunnelsJson = Join-Path $logDir "tunnels.json"
$desktopTxt  = Join-Path ([Environment]::GetFolderPath('Desktop')) "Debris-Tunnels.txt"

@{
    generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
    mode         = "named"
    tunnel_id    = $tunnelId
    tunnels      = @(
        @{ port = 8501; label = "Streamlit"; url = $urlUI  },
        @{ port = 8502; label = "FastAPI";   url = $urlAPI }
    )
} | ConvertTo-Json -Depth 5 | Out-File -FilePath $tunnelsJson -Encoding UTF8

@(
    "Space Debris System - Public Tunnel URLs (named tunnel, stable)",
    "Generated at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "",
    "[8501] Streamlit   $urlUI",
    "[8502] FastAPI     $urlAPI",
    "",
    "These URLs are PERMANENT (DNS-backed). They survive reboots, cloudflared restarts,",
    "and any tunnel reconnect. Make sure cloudflared.exe stays running on this machine."
) -join "`r`n" | Out-File -FilePath $desktopTxt -Encoding UTF8

# Refresh the autostart vbs to use the named tunnel runner instead of quick tunnel
$startupDir = [Environment]::GetFolderPath('Startup')
$startupLnk = Join-Path $startupDir "DebrisTunnels.vbs"
$vbsCmd = '"' + $CfdExe + '" tunnel --no-autoupdate run --token "' + $tunnelToken + '" --logfile "' + $namedLog + '"'
@"
' Auto-generated by setup_named_tunnel.ps1
' Silently launches the named cloudflared tunnel on user login.
Set ws = CreateObject("WScript.Shell")
ws.Run "$vbsCmd", 0, False
"@ | Out-File -FilePath $startupLnk -Encoding ASCII

Write-Host ""
Write-Host "=================== DONE ===================" -ForegroundColor Green
Write-Host "Streamlit  : $urlUI"
Write-Host "FastAPI    : $urlAPI"
Write-Host "Saved URLs : $desktopTxt"
Write-Host "Saved JSON : $tunnelsJson"
Write-Host "Autostart  : $startupLnk (named tunnel)"
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Note: DNS may take 30~120s to propagate the first time. If a domain"
Write-Host "      shows 'tunnel error' for the first minute, just wait and retry."
