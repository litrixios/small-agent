# 奖励模型（Reward Model）部署与微调指南（基于当前仓库）

本指南基于仓库内已有的数据与脚本，给出一条可复现实操路径：

1. 检查并准备奖励模型数据集；
2. 使用 vLLM 部署一个开源模型（OpenAI 兼容 API）；
3. 用仓库脚本生成 RM 训练格式；
4. 使用 LLaMA-Factory 进行奖励模型微调；
5. 用仓库评测脚本打分。

## 1) 仓库里是否已经有训练集与训练步骤？

有，且比较完整：

- 数据说明中明确给出了奖励模型相关三份数据：
  - `dataset/reward_data/train_data.jsonl`（训练）
  - `dataset/reward_data/test_data.jsonl`（测试）
  - `dataset/reward_data/trainset_reward_llama.json`（LLaMA-Factory 可用格式）
- 数据流程文档给出了从标注转换到构建 RM 训练集的命令。
- `dataset/build_reward_trainset.py` 提供了将 `train_data.jsonl` 处理成 `trainset_reward_llama.json` 的完整逻辑。

## 2) 快速部署开源模型（vLLM）

> 用途：让仓库脚本通过 OpenAI 兼容接口访问你的模型。

你提到的这个点非常关键：**是的，通常需要先把模型权重下载到本地**（除非机器能直接联网并由 vLLM 在首次启动时自动拉取）。

### 2.1 先下载模型（推荐三种方式）

下面以 `Qwen/Qwen2.5-7B-Instruct` 为例。

**方式 A：huggingface-cli（最常用）**

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen2.5-7B-Instruct
```

下载后，本地目录 `/data/models/Qwen2.5-7B-Instruct` 就是模型路径。

**方式 B：git-lfs（适合你习惯 git 管理）**

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct /data/models/Qwen2.5-7B-Instruct
```

**方式 C：ModelScope（国内网络常更稳定）**

```bash
pip install modelscope
python - <<'PY'
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='/data/models/Qwen2.5-7B-Instruct')
PY
```

### 2.2 用本地路径启动 vLLM

如果你已经下载到本地，建议把 `--model` 写成本地目录，避免运行时二次拉取：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b-instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```

如果不提前下载，也可以直接写 HF 名称（前提是当前机器可联网并有权限拉取）：

示例（可替换模型名）：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b-instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```

启动后，仓库脚本里把 `base_url` 指向：

```text
http://localhost:8000/v1/
```

## 3) 生成/更新奖励模型训练样本

如果你有新标注数据，先执行：

```bash
cd dataset/annotation
python convert_annotations.py
```

然后回到仓库根目录运行：

```bash
python dataset/build_reward_trainset.py
```

注意先改 `dataset/build_reward_trainset.py` 里的：

- `api_key`
- `base_url`
- `model`

## 4) 用 LLaMA-Factory 微调奖励模型（RM）

仓库文档推荐使用 LLaMA-Factory。一个最小可执行思路：

1. 安装并进入 LLaMA-Factory；
2. 在其数据配置里注册 `dataset/reward_data/trainset_reward_llama.json`；
3. 选择奖励模型训练 stage（RM）；
4. 启动 LoRA 或全参训练。

示例命令（参数需按你的 GPU 资源与 LLaMA-Factory 版本调整）：

```bash
llamafactory-cli train \
  --stage rm \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --do_train true \
  --dataset proactive_rm \
  --dataset_dir /path/to/ProactiveAgent/dataset/reward_data \
  --template qwen \
  --finetuning_type lora \
  --output_dir /path/to/output/proactive-rm \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --bf16 true
```

> 核心点：让 `proactive_rm` 指向 `trainset_reward_llama.json`，并确保字段映射为多轮对话格式。

## 5) 部署微调后的奖励模型并评估

将微调后模型（或 merged 权重）再次用 vLLM 启动为 OpenAI 接口后，运行：

```bash
python eval/reward_model_scoring.py
```

然后你可以得到 RM 在 `test_data.jsonl` 上的分类统计与 F1 等指标。

## 6) 常见问题

- **Q: 没有新标注，能直接训吗？**
  可以，仓库已提供现成 `train_data.jsonl` 与 `trainset_reward_llama.json`。

- **Q: 为什么 `build_reward_trainset.py` 还要调用一个模型？**
  它在构造带“reasoning + judgement”的训练样本，并做一致性校验，属于“数据增强/蒸馏”流程。

- **Q: 我只想先验证流程通不通？**
  可以先用小模型（如 7B）+ 少量 epoch，确认训练、部署、打分三段链路打通，再扩展规模。
