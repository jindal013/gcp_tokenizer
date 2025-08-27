set -euo pipefail

PY_SCRIPT="ray_350bt.py" 
RUNTIME_S="5" 
RESULTS="bs_benchmark_$(date +%Y%m%d_%H%M%S).csv"
EXTRA_ARGS=""          # or "" if you don't want to resume

echo "batch_size,tokens_per_s" > "$RESULTS"

# Sweep: 25, 75, 125, ..., 725, and also include 750 explicitly
for bs in $(seq 25 50 750) 750; do
  # De-duplicate 750 if seq happened to include it (it usually won't)
  if grep -q "^$bs," "$RESULTS"; then
    continue
  fi

  echo "Testing BATCH_SIZE=$bs ..."
  # Run with a short timeout; capture tqdm (which prints on stderr)
  # BENCHMARK=1 disables uploads if you added the guard in your script
  # Run and capture BOTH stdout+stderr (tqdm is on stderr)
  LOG="$(BENCHMARK=1 BATCH_SIZE="$bs" timeout --signal=INT "${RUNTIME_S}s" \
        python "$PY_SCRIPT" $EXTRA_ARGS 2>&1 || true)"

  # Extract the last rate like 2075254.64tokens/s OR 2075254.64 tokens/s
  RATE="$(printf "%s\n" "$LOG" \
    | sed -nE 's/.*([0-9]+(\.[0-9]+)?) ?tokens\/s.*/\1/p' \
    | tail -n1)"
  [ -z "${RATE:-}" ] && RATE="NA"


  echo "$LOG"

  # Fallback if we didn’t see a rate line yet
  if [ -z "${RATE:-}" ]; then
    RATE="NA"
  fi

  echo "$bs,$RATE" | tee -a "$RESULTS"
done

echo "Done. Results saved to $RESULTS"
