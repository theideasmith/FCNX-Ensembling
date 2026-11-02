#!/usr/bin/env bash
# Live GPU temperature, power, utilization and memory, refreshed every INTERVAL seconds (default 2).
# Temperature turns yellow at WARN °C and red at HOT °C (the 4090 dropped off the bus at ~89 °C on 2026-10-27).
#   ./gputemp.sh            # refresh every 2 s
#   ./gputemp.sh 1          # refresh every 1 s
#   WARN=75 HOT=82 ./gputemp.sh
INTERVAL=${1:-2}
WARN=${WARN:-80}
HOT=${HOT:-85}
trap 'tput cnorm; echo; exit' INT TERM
tput civis
clear
while true; do
  tput cup 0 0
  printf '%s   (every %ss, Ctrl-C to quit)\n\n' "$(date '+%H:%M:%S')" "$INTERVAL"
  printf '%-3s %-26s %7s %9s %6s %15s\n' GPU name temp power util memory
  nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total \
             --format=csv,noheader,nounits |
  while IFS=', ' read -r idx rest; do
    IFS=',' read -r name t pw pl u mu mt <<< "$rest"
    t=${t// /}; c=$'\e[32m'
    (( t >= WARN )) && c=$'\e[33m'
    (( t >= HOT )) && c=$'\e[1;31m'
    printf '%-3s %-26s %s%5s°C\e[0m %4.0f/%-4.0fW %5s%% %6s/%-6s MiB\e[K\n' \
           "$idx" "${name# }" "$c" "$t" "$pw" "$pl" "${u// /}" "${mu// /}" "${mt// /}"
  done
  tput ed
  sleep "$INTERVAL"
done
