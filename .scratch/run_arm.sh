#!/bin/bash
# NAME GPUS PORT WORKLOADS -- launch args...
NAME=$1; GPUS=$2; PORT=$3; shift 3
cd /data/audrey/v4
export PYTHONPATH=/data/audrey/v4 HF_HOME=/data/audrey/hf_cache
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1
ulimit -n 65535
python3 /data/audrey/launch_v4.py --port $PORT "$@" > /data/audrey/m_$NAME.log 2>&1 &
SRV=$!
up=0
for i in $(seq 1 150); do
  curl -sf --max-time 5 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { up=1; break; }
  kill -0 $SRV 2>/dev/null || break
  sleep 6
done
if [ "$up" != "1" ]; then
  echo "{\"arm\":\"$NAME\",\"error\":\"never ready\"}" >> /data/audrey/matrix.jsonl
else
  KV=$(grep -a "KV Cache is allocated" /data/audrey/m_$NAME.log|grep -o "#tokens: [0-9]*"|tr "\n" " ")
  echo "{\"arm\":\"$NAME\",\"kv\":\"$KV\"}" >> /data/audrey/matrix.jsonl
  # match Melody's cells: 8128/32 at c8 and c16
  for C in 8 16; do
    OUT=$(python3 /data/audrey/closedloop.py --port $PORT --concurrency $C --n 100 \
          --prompt-tokens 8128 --max-tokens 32 --tag $NAME 2>/dev/null|tail -1)
    echo "{\"arm\":\"$NAME\",\"cell\":\"8128/32 c$C\",\"result\":$OUT}" >> /data/audrey/matrix.jsonl
  done
fi
kill $SRV 2>/dev/null; sleep 15; pkill -9 -f "launch_v4.py --port $PORT" 2>/dev/null; sleep 8
