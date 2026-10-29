#!/usr/bin/env bash
# Prints one progress line every INTERVAL seconds for the benchmark chain
# (sdwpf_smoke -> sdwpf_arima -> bdg2_smoke -> timing -> export) and exits when it ends.
# Usage: scripts/progress.sh [interval_seconds]
set -u
cd "$(dirname "$0")/.."
INTERVAL=${1:-300}
WIND_LOG=results/sdwpf_rerun.log
TIMING_LOG=results/timing.log
# Durations measured on previous runs, used for the phases not started yet.
ARIMA_S=600; BDG2_S=620; TIMING_S=180

elapsed() {  # seconds since the given script started, empty if not running
  ps -eo etime=,command= | grep "scripts/$1" | grep -v grep | head -1 | awk '{
    n = split($1, a, /[-:]/); s = a[n] + 60 * a[n-1];
    if (n >= 3) s += 3600 * a[n-2]; if (n == 4) s += 86400 * a[1]; print s }'
}
mins() { echo $(( ($1 + 59) / 60 )); }

while true; do
  now=$(date +%H:%M)
  if t=$(elapsed sdwpf_smoke.py) && [ -n "$t" ]; then
    done_=$(grep -c -E "^ *[0-9]+ +Timestral" "$WIND_LOG" 2>/dev/null || echo 0)
    rate=$(( done_ > 0 ? t / done_ : 20 ))
    left=$(( (142 - done_) * rate ))
    echo "[$now] Fase 1/4 eolico: $done_/142 casi (~${rate}s a caso) · fase ~$(mins $left) min · totale ~$(mins $(( left + ARIMA_S + BDG2_S + TIMING_S ))) min"
  elif t=$(elapsed sdwpf_arima.py) && [ -n "$t" ]; then
    # macOS reports the elapsed time of the whole `a && b` subshell for b, so the rate
    # comes from the per-case runtimes the script logs instead.
    done_=$(grep -c "'model': 'AutoARIMA'" "$WIND_LOG" 2>/dev/null || echo 0)
    rate=$(grep "'model': 'AutoARIMA'" "$WIND_LOG" 2>/dev/null | grep -o "'seconds': [0-9.]*" | awk '{s += $2; n++} END {print n ? int(s / n) + 1 : 30}')
    left=$(( (20 - done_) * rate ))
    echo "[$now] Fase 2/4 ARIMA eolico: $done_/20 casi · fase ~$(mins $left) min · totale ~$(mins $(( left + BDG2_S + TIMING_S ))) min"
  elif t=$(elapsed bdg2_smoke.py) && [ -n "$t" ]; then
    left=$(( BDG2_S > t ? BDG2_S - t : 60 ))
    echo "[$now] Fase 3/4 edifici: in corso da $(mins $t) min · fase ~$(mins $left) min · totale ~$(mins $(( left + TIMING_S ))) min"
  elif t=$(elapsed timing.py) && [ -n "$t" ]; then
    done_=$(grep -c "^{" "$TIMING_LOG" 2>/dev/null || echo 0)
    echo "[$now] Fase 4/4 tempi: $done_/8 misure · ~$(mins $(( TIMING_S > t ? TIMING_S - t : 30 ))) min"
  elif pgrep -f "scripts/export_frontend.py" >/dev/null; then
    echo "[$now] Export di data.js in corso"
  else
    if grep -q -E "Traceback|Error" "$WIND_LOG" results/bdg2_rerun.log "$TIMING_LOG" 2>/dev/null; then
      echo "[$now] Catena terminata CON ERRORI: controllare i log in results/"
    else
      echo "[$now] Catena terminata: data.js aggiornato alle $(date -r frontend/data.js +%H:%M)"
    fi
    exit 0
  fi
  sleep "$INTERVAL"
done
