#!/usr/bin/env bash
set -euo pipefail

# ====== Config ======
MODEL_BASE="deepseek"              # private.toml 里的 baseline alias
MODEL_RL="deepseek-rl"             # private.toml 里的 RL alias（建议和 baseline 同底层模型）
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-REPLACE_ME}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"

# RL gate hyperparams (fast)
EPISODES="${EPISODES:-20}"
DIM="${DIM:-8192}"

ROOT="/home/aistudio/work/small-agent"
cd "$ROOT"

echo "[1/7] Train RL gate ..."
python eval/quick_rl.py \
  --episodes "$EPISODES" \
  --dim "$DIM" \
  --out eval/results/quick_rl_metrics.json \
  --model_out eval/results/quick_rl_model.json

echo "[2/7] Run baseline traces: ${MODEL_BASE}"
python eval/script.py --model_name "$MODEL_BASE"

echo "[3/7] Run RL-gated traces: ${MODEL_RL}"
python eval/script.py \
  --model_name "$MODEL_RL" \
  --gate_model_path eval/results/quick_rl_model.json

echo "[4/7] Clean old judged outputs ..."
rm -rf eval/judged/"$MODEL_BASE" eval/judged/"$MODEL_RL"

echo "[5/7] Judge via API ..."
cd eval
JUDGE_BASE_URL="$JUDGE_BASE_URL" \
JUDGE_API_KEY="$JUDGE_API_KEY" \
JUDGE_MODEL="$JUDGE_MODEL" \
sh judge_result.sh

echo "[6/7] Calculate all judged folders ..."
sh calculate.sh

echo "[7/7] Print side-by-side summary ..."
python calculate_agent_performance.py "judged/${MODEL_BASE}"
python calculate_agent_performance.py "judged/${MODEL_RL}"

echo "Done."
