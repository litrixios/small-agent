<div align= "center">
    <h1> 🧩 ProactiveBench </h1>
</div>

# 总览

ProactiveBench 是用来评估主动智能体的基准点。其包含一个数据集，一个奖励模型和评估脚本。
我们的训练集包含了三个类别的事件：编程，写作和日常生活。
当前，我们的测试集包含`227`个事件。
在数据集上训练的奖励模型在测试集上的 F1 分数达到了 `0.918`.
我们将提供所有用于评估主动智能体和奖励模型的脚本。

## 奖励模型评估
奖励模型用于评估主动智能体的性能。
你可以在此(敬请期待)下载奖励模型并且通过 [VLLM](https://github.com/vllm-project/vllm) 等框架以搭建并提供 OpenAI 风格的 API。

在此之后，你应当修改 `reward_model_scoring.py` 脚本并设置地址为自己模型的地址，运行脚本
```bash
python eval/reward_model_scoring.py
```
在该过程之后，你将会得到你的奖励模型的最终分数。


## 快速强化学习基线（升级版且仍然快）

如果你希望尽快跑出强化学习实验，可以使用升级后的 contextual bandit 脚本：

```bash
python eval/quick_rl.py --episodes 20 --dim 8192
```

相比最初的极简版，这个版本新增：
- 更丰富的哈希特征（unigram + bigram + 文本长度桶）
- REINFORCE 策略梯度 + 移动平均 baseline
- 在验证集上自动调阈值（提升 F1 稳定性）

脚本会在 `dataset/reward_data/train_data.jsonl` 上训练，在
`dataset/reward_data/test_data.jsonl` 上评估，并将结果写入
`eval/results/quick_rl_metrics.json`。

提速建议：
- 降低迭代：`--episodes 8`
- 缩小维度：`--dim 2048`
- 固定随机种子：`--seed 42`


### Qwen + 强化学习门控（推荐对照实验）

为了对比“仅 Qwen”与“Qwen + RL 的 help/no-help 决策”，可以按下面运行：

```bash
# 1) 训练 RL 门控模型
python eval/quick_rl.py --episodes 20 --dim 8192 \
  --out eval/results/quick_rl_metrics.json \
  --model_out eval/results/quick_rl_model.json

# 2) 仅 Qwen 基线（无门控）
python eval/script.py run --model_name qwen2-7b-instruct

# 3) Qwen + RL 门控
python eval/script.py run --model_name qwen2-7b-instruct \
  --gate_model_path eval/results/quick_rl_model.json
```

当设置 `gate_model_path` 后，会先预测 `p_help`：
- 若 `p_help` 低于阈值，则直接输出 `null`；
- 若 `p_help` 高于阈值，再调用 LLM 生成具体帮助内容。


### 使用 API 进行 judge（例如 DeepSeek）

如果你没有本地 reward model（`localhost:8000`），可以直接用 API 做判分：

```bash
cd eval
JUDGE_BASE_URL=https://api.deepseek.com/v1 \
JUDGE_API_KEY=YOUR_API_KEY \
JUDGE_MODEL=deepseek-chat \
sh judge_result.sh

sh calculate.sh
```

## 主动智能体评估
为了检查模型性能，你需要修改文件 `./eval/script.py` 以导入你的模型，同时运行脚本
```bash
python eval/script.py
```
该脚本会向模型输入测试数据，并且保存所有的轨迹和智能体应答于文件夹 `./eval/traces_new` 下。
在该过程之后，你可以运行
```bash
# 你应当在运行该脚本前修改 judge_agent_prediction.py 中的地址为自己的奖励模型
sh eval/judge_result.sh
```
其将让奖励模型评估来自智能体的回复是否可接受，结果将会存放于 `./eval/judged` 文件夹下。

在经过奖励模型评估之后，你可以运行
```bash
sh calculate.sh
```
获得你的模型的最终分数。
