#!/usr/bin/env bash
# Run ray points one after another (2 chains each, both GPUs), waiting for each to finish.
#   BS="1.41421356 2 2.82842712 4" DTYPE=float32 ./queue_ray_k0p015.sh
set -uo pipefail
cd "$(dirname "$0")"
BS=${BS:-"1.41421356 2 2.82842712 4"}
DTYPE=${DTYPE:-float32}
WAIT_FOR=${WAIT_FOR:-}   # log glob to wait on before starting (e.g. a point already running)
wait_logs() {  # wait until every log matching $1 says done or failed
  local n
  until n=$(ls $1 2>/dev/null | wc -l) && [ "$n" -gt 0 ] && [ "$(grep -lE 'done in|Traceback' $1 | wc -l)" -ge "$n" ]; do sleep 20; done
}
[ -n "$WAIT_FOR" ] && wait_logs "$WAIT_FOR"
for b in $BS; do
  tag=$(echo "$b" | tr . p)
  B=$b DTYPE=$DTYPE ./launch_ray_k0p015.sh
  sleep 5
  wait_logs "runs/logs/ray_k0p015_b${tag}_${DTYPE}_*.log"
  echo "$(date +%H:%M) b=$b finished: $(grep -h 'done in' runs/logs/ray_k0p015_b${tag}_${DTYPE}_*.log | wc -l) done, $(grep -l Traceback runs/logs/ray_k0p015_b${tag}_${DTYPE}_*.log | wc -l) failed"
done
echo "queue finished"
