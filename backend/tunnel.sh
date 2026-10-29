#!/usr/bin/env bash
# (Re)start the public Cloudflare quick tunnel to the LimbScan server and print the phone URL.
cd "$(dirname "$0")"
ps -eo pid,args | grep "[c]loudflared tunnel" | awk '{print $1}' | xargs -r kill
if ! curl -s -o /dev/null --max-time 3 http://127.0.0.1:8080/; then
  echo "LimbScan server not running - starting it"; (setsid nohup ./run.sh > data/server.log 2>&1 &); sleep 6
fi
(setsid nohup ~/.local/bin/cloudflared tunnel --no-autoupdate --url http://localhost:8080 > data/tunnel.log 2>&1 &)
for i in $(seq 1 40); do
  U=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" data/tunnel.log | head -1); [ -n "$U" ] && break; sleep 1
done
[ -z "$U" ] && { echo "Tunnel failed to start, see data/tunnel.log"; exit 1; }
sleep 5
echo; echo "  Open on your phone:  $U"; echo
