#!/usr/bin/env bash
set -euo pipefail

# ====== Config ======
MODEL_BASE="deepseek"              # private.toml 里的 baseline alias
MODEL_RL="deepseek"                # 建议同一底模，只让 gate 造成差异
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8000/v1/}"
JUDGE_API_KEY="${JUDGE_API_KEY:-sk-1234}"
JUDGE_MODEL="${JUDGE_MODEL:-proactive-rm}"

# RL gate hyperparams
EPISODES="${EPISODES:-20}"
DIM="${DIM:-8192}"
SEED="${SEED:-42}"

ROOT="/home/aistudio/work/small-agent"
cd "$ROOT"

echo "[1/8] Train RL gate ..."
python eval/quick_rl.py \
  --episodes "$EPISODES" \
  --dim "$DIM" \
  --seed "$SEED" \
  --out eval/results/quick_rl_metrics_newrm.json \
  --model_out eval/results/quick_rl_model_newrm.json

echo "[2/8] Run baseline traces: ${MODEL_BASE}"
python eval/script.py --model_name "$MODEL_BASE"

echo "[3/8] Run RL-gated traces: ${MODEL_RL}"
python eval/script.py \
  --model_name "$MODEL_RL" \
  --gate_model_path eval/results/quick_rl_model_newrm.json

echo "[4/8] Clean old judged outputs ..."
rm -rf "eval/judged/${MODEL_BASE}" "eval/judged/${MODEL_RL}"

echo "[5/8] Judge via RM/API ..."
cd eval
JUDGE_BASE_URL="$JUDGE_BASE_URL" \
JUDGE_API_KEY="$JUDGE_API_KEY" \
JUDGE_MODEL="$JUDGE_MODEL" \
sh judge_result.sh

echo "[6/8] Calculate all judged folders ..."
sh calculate.sh

echo "[7/8] Print side-by-side summary ..."
python calculate_agent_performance.py "judged/${MODEL_BASE}"
python calculate_agent_performance.py "judged/${MODEL_RL}"

echo "[8/8] Done."
