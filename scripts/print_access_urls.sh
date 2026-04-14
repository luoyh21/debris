#!/usr/bin/env bash
# 打印当前机器上 Streamlit 可尝试的访问地址（IP 随 DHCP/换网会变）
echo "本机:   http://127.0.0.1:8501"
echo "本机:   http://localhost:8501"
for _if in en0 en1 en2; do
  _ip=$(ipconfig getifaddr "$_if" 2>/dev/null) || true
  if [[ -n "${_ip}" ]]; then
    echo "网卡 ${_if}: http://${_ip}:8501"
  fi
done
echo ""
echo "若旧 IP 无法打开，多半是地址已变，请用上面新地址。"
echo "从外网访问需：路由器端口转发 8501、本机防火墙放行、且非运营商严格 CGNAT。"
