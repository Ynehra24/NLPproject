#!/usr/bin/env bash
set -euo pipefail

# Manage the long evader training run with safe pause/resume commands.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POST_GEN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$POST_GEN_DIR/.." && pwd)"

OUTPUT_DIR_REL="${OUTPUT_DIR_REL:-HMGC-dataset/output/checkgpt/model/evader_flan_t5_base_12to15h_earlystop_v1}"
OUTPUT_DIR_ABS="$POST_GEN_DIR/$OUTPUT_DIR_REL"
OUTPUT_DIR_TAG="$(basename "$OUTPUT_DIR_REL")"

PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/.venv/bin/python}"
TRAIN_FILES_PATTERN="${TRAIN_FILES_PATTERN:-HMGC-dataset/output/checkgpt/baseline/*.jsonl}"
MODEL_NAME="${MODEL_NAME:-google/flan-t5-base}"
EVASION_EVAL_INTERVAL_SECONDS="${EVASION_EVAL_INTERVAL_SECONDS:-3600}"
EVASION_EVAL_DETECTOR_MODEL_PATH="${EVASION_EVAL_DETECTOR_MODEL_PATH:-HMGC-dataset/output/checkgpt/model/surrogate_distilroberta_base_fast}"
EVASION_EVAL_PROBE_FILES="${EVASION_EVAL_PROBE_FILES:-HMGC-dataset/output/checkgpt/baseline/Paraphrase_l40_o40.jsonl}"
EVASION_EVAL_SAMPLES="${EVASION_EVAL_SAMPLES:-256}"
EVASION_EVAL_BATCH_SIZE="${EVASION_EVAL_BATCH_SIZE:-16}"
EVASION_EVAL_DETECTOR_DEVICE="${EVASION_EVAL_DETECTOR_DEVICE:-cpu}"

find_training_pid() {
  pgrep -f "train_evader.py.*${OUTPUT_DIR_TAG}" | tail -n 1 || true
}

latest_checkpoint() {
  ls -d "$OUTPUT_DIR_ABS"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true
}

show_status() {
  local pid
  pid="$(find_training_pid)"

  if [[ -n "$pid" ]]; then
    echo "Training is RUNNING (PID: $pid)"
    ps -ww -p "$pid" -o pid= -o etime= -o command=
  else
    echo "Training is NOT running for output tag: $OUTPUT_DIR_TAG"
  fi

  local ckpt
  ckpt="$(latest_checkpoint)"
  if [[ -n "$ckpt" ]]; then
    echo "Latest checkpoint: $ckpt"
  else
    echo "No checkpoint found yet in $OUTPUT_DIR_ABS"
  fi

  local log_file="$OUTPUT_DIR_ABS/training_log.jsonl"
  if [[ -f "$log_file" ]]; then
    echo "Recent logs:"
    tail -n 5 "$log_file"
  fi

  local evasion_status_file="$OUTPUT_DIR_ABS/hourly_evasion_status.json"
  if [[ -f "$evasion_status_file" ]]; then
    echo "Hourly evasion eval status:"
    "$PYTHON_BIN" - "$evasion_status_file" <<'PY'
import datetime
import json
import sys

status_path = sys.argv[1]
with open(status_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"  enabled: {data.get('enabled')}")
print(f"  interval_seconds: {data.get('interval_seconds')}")
print(f"  probe_samples: {data.get('probe_samples')}")
print(f"  detector_model_path: {data.get('detector_model_path')}")
print(f"  detector_device: {data.get('detector_device')}")

last_eval = data.get("last_eval")
if isinstance(last_eval, dict):
    ts = last_eval.get("time")
    if ts is not None:
        print(f"  last_eval_time: {datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}")
    if "step" in last_eval:
        print(f"  last_eval_step: {last_eval['step']}")
    if "ai_evasion_accuracy" in last_eval:
        print(f"  last_ai_evasion_accuracy: {float(last_eval['ai_evasion_accuracy']):.4f}")
    if "error" in last_eval:
        print(f"  last_eval_error: {last_eval['error']}")
else:
    print("  last_eval: none yet")

next_due = data.get("next_due_unix")
if next_due is not None:
    print(f"  next_due_time: {datetime.datetime.fromtimestamp(next_due).strftime('%Y-%m-%d %H:%M:%S')}")
PY
  else
    echo "Hourly evasion status file not found yet in $OUTPUT_DIR_ABS"
  fi

  local evasion_log_file="$OUTPUT_DIR_ABS/hourly_evasion_eval.jsonl"
  if [[ -f "$evasion_log_file" ]]; then
    echo "Recent hourly evasion events:"
    tail -n 3 "$evasion_log_file"
  fi
}

pause_training() {
  local pid
  pid="$(find_training_pid)"

  if [[ -z "$pid" ]]; then
    echo "No running training process found for $OUTPUT_DIR_TAG"
    exit 1
  fi

  echo "Sending SIGINT to PID $pid for safe checkpointed pause..."
  kill -SIGINT "$pid"

  # Trainer saves at next step boundary, so wait for graceful exit.
  for _ in {1..180}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Training paused cleanly."
      break
    fi
    sleep 2
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "Process is still running. It may be finishing current step before exiting."
  fi

  local ckpt
  ckpt="$(latest_checkpoint)"
  if [[ -n "$ckpt" ]]; then
    echo "Latest checkpoint after pause: $ckpt"
  fi
}

resume_training() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found or not executable: $PYTHON_BIN"
    exit 1
  fi

  local pid
  pid="$(find_training_pid)"
  if [[ -n "$pid" ]]; then
    echo "Training already running (PID: $pid)."
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR_ABS"
  cd "$POST_GEN_DIR"

  echo "Resuming training in foreground from latest checkpoint (auto)."
  echo "Output dir: $OUTPUT_DIR_REL"

  exec "$PYTHON_BIN" train_evader.py \
    --train_files "$TRAIN_FILES_PATTERN" \
    --output_dir "$OUTPUT_DIR_REL" \
    --model_name_or_path "$MODEL_NAME" \
    --num_train_epochs 10 \
    --learning_rate 1.5e-5 \
    --warmup_ratio 0.08 \
    --weight_decay 0.02 \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_source_length 256 \
    --max_target_length 160 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 20 \
    --logging_steps 20 \
    --early_stopping_patience 12 \
    --early_stopping_threshold 0.0007 \
    --evasion_eval_interval_seconds "$EVASION_EVAL_INTERVAL_SECONDS" \
    --evasion_eval_detector_model_path "$EVASION_EVAL_DETECTOR_MODEL_PATH" \
    --evasion_eval_probe_files "$EVASION_EVAL_PROBE_FILES" \
    --evasion_eval_samples "$EVASION_EVAL_SAMPLES" \
    --evasion_eval_batch_size "$EVASION_EVAL_BATCH_SIZE" \
    --evasion_eval_detector_device "$EVASION_EVAL_DETECTOR_DEVICE" \
    --use_mps_device \
    --resume_from_checkpoint auto
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  pause    Send SIGINT for safe pause + checkpoint/state save
  resume   Resume training from latest checkpoint (foreground)
  status   Show running status, latest checkpoint, recent logs

Optional environment overrides:
  OUTPUT_DIR_REL, PYTHON_BIN, TRAIN_FILES_PATTERN, MODEL_NAME
  EVASION_EVAL_INTERVAL_SECONDS, EVASION_EVAL_DETECTOR_MODEL_PATH
  EVASION_EVAL_PROBE_FILES, EVASION_EVAL_SAMPLES
  EVASION_EVAL_BATCH_SIZE, EVASION_EVAL_DETECTOR_DEVICE
EOF
}

cmd="${1:-}"
case "$cmd" in
  pause)
    pause_training
    ;;
  resume)
    resume_training
    ;;
  status)
    show_status
    ;;
  *)
    usage
    exit 1
    ;;
esac
