# 将 PostgreSQL 角色密码同步为 .env 中的 DB_PASSWORD（优先走 Docker db 容器）
Set-Location $PSScriptRoot\..
$py = if (Test-Path .\venv\Scripts\python.exe) { ".\venv\Scripts\python.exe" } else { "python" }
& $py scripts\sync_db_password.py
