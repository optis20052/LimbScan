#!/usr/bin/env bash
# Manage the LimbScan server running on the GPU PC (ssh alias "limbpc").
#   ./pc.sh url      -> current public phone link
#   ./pc.sh status   -> service status
#   ./pc.sh restart  -> restart server + new tunnel link
#   ./pc.sh deploy   -> copy code changes from this laptop to the PC and restart the server
#   ./pc.sh logs     -> follow server logs
case "$1" in
  url)     ssh limbpc 'grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" ~/limbscan_srv/tunnel.log | tail -1' ;;
  status)  ssh limbpc 'systemctl --user --no-pager status limbscan limbscan-tunnel | grep -E "●|Active"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader' ;;
  restart) ssh limbpc 'systemctl --user restart limbscan limbscan-tunnel'; sleep 12; "$0" url ;;
  deploy)  rsync -az --exclude .venv --exclude data --exclude __pycache__ "$(dirname "$0")"/ limbpc:~/limbscan_srv/limbscan/ && ssh limbpc 'systemctl --user restart limbscan' && echo deployed ;;
  logs)    ssh limbpc 'journalctl --user -u limbscan -f' ;;
  *)       sed -n 2,7p "$0" ;;
esac
