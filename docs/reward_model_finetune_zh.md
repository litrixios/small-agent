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

可以，**如果你已经完成第 2 步、且不需要新增数据，就可以直接开始第 4 步微调**。

下面给你一个“从 0 到 1”的新手版流程（默认你要训练 LoRA 版 RM）：

### 4.1 安装 LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

> 若安装报错，通常是 `torch` / `cuda` 版本不匹配，先按你机器 CUDA 版本安装 PyTorch 再重试。

### 4.2 把本仓库数据链接到 LLaMA-Factory

假设你的 ProactiveAgent 仓库在：`/workspace/small-agent`。

这里我们直接使用已有数据：

- `/workspace/small-agent/dataset/reward_data/trainset_reward_llama.json`

### 4.3 在 LLaMA-Factory 注册数据集

编辑 `LLaMA-Factory/data/dataset_info.json`，添加一条（名字可自定义，这里用 `proactive_rm`）：

```json
{
  "proactive_rm": {
    "file_name": "/workspace/small-agent/dataset/reward_data/trainset_reward_llama.json"
  }
}
```

> 如果你的 `dataset_info.json` 已有很多项，只需要在最外层 JSON 里新增这一个 key。注意逗号和 JSON 格式合法。

### 4.4 准备一个最小训练命令（先跑通）

在 `LLaMA-Factory` 目录执行：

```bash
llamafactory-cli train \
  --stage rm \
  --do_train true \
  --model_name_or_path /data/models/Qwen2.5-7B-Instruct \
  --dataset proactive_rm \
  --template qwen \
  --finetuning_type lora \
  --output_dir ./saves/proactive-rm-lora \
  --overwrite_output_dir true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --cutoff_len 4096 \
  --logging_steps 10 \
  --save_steps 200 \
  --plot_loss true \
  --bf16 true
```

参数解释（你最需要关注的 6 个）：

- `--stage rm`：奖励模型训练模式。
- `--model_name_or_path`：你在第 2 步下载好的底模路径。
- `--dataset proactive_rm`：对应刚才在 `dataset_info.json` 注册的名字。
- `--template qwen`：Qwen 系列一般用这个模板。
- `--finetuning_type lora`：先用 LoRA，显存压力更小。
- `--bf16 true`：A100 / 4090 等常见新卡建议打开；如果不支持改成 `--fp16 true`。

### 4.5 训练完成后你会得到什么

- LoRA 适配器权重目录：`./saves/proactive-rm-lora`
- 你可以：
  1) 直接用支持 LoRA 的推理方式加载；或
  2) 先 merge 成完整权重，再用 vLLM 部署。

### 4.6 新手常见报错与处理

- **显存不足（OOM）**：把 `--cutoff_len` 降到 2048，`--gradient_accumulation_steps` 适当调大。
- **模板不匹配**：Qwen 用 `--template qwen`；Llama 系请改对应模板。
- **找不到数据集**：检查 `dataset_info.json` 的 key 名与 `--dataset` 是否完全一致。
- **精度参数报错**：`bf16` 不支持就改 `fp16`。

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
