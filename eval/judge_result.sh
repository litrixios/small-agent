#!/bin/bash

cd "$(dirname "$0")"

trace_folder="traces_new"
save_folder="judged"

# Optional overrides for API-based judge model
# Example:
# JUDGE_BASE_URL=https://api.deepseek.com/v1 \
# JUDGE_API_KEY=sk-xxx \
# JUDGE_MODEL=deepseek-chat \
# sh eval/judge_result.sh
judge_base_url=${JUDGE_BASE_URL:-http://localhost:8000/v1/}
judge_api_key=${JUDGE_API_KEY:-sk-1234}
judge_model=${JUDGE_MODEL:-activellama}

mkdir -p "$save_folder"

find "$trace_folder" -type f -name "*.json" | while read json_file; do
    relative_path="${json_file#$trace_folder/}"
    mkdir -p "$save_folder/$(dirname "$relative_path")"

    destination_file="$save_folder/$relative_path"

    if [ -e "$destination_file" ]; then
        echo "Warning: File $destination_file already exists. Skipping..."
        continue
    fi

    python judge_agent_prediction.py "$json_file" -o "$destination_file" \
      --base_url "$judge_base_url" \
      --api_key "$judge_api_key" \
      --model "$judge_model"
done
