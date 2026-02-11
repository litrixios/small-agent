"""Fast upgraded RL baseline for proactive prompting.

This keeps training lightweight while improving over the initial version:
1) richer hashed features (unigram + bigram + length bucket)
2) policy-gradient bandit with moving baseline (REINFORCE)
3) automatic decision-threshold tuning on a held-out validation split

The task is contextual bandit:
- state: recent observation text
- action: 0 (stay silent) / 1 (propose help)
- reward: +1 if action matches help-needed signal, else -1
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fire
import jsonlines

TOKEN_RE = re.compile(r"[A-Za-z_]+")


@dataclass
class Sample:
    text: str
    help_needed: int


def _iter_samples(path: str | Path) -> Iterable[Sample]:
    with jsonlines.open(path) as reader:
        for row in reader:
            obs = row.get("obs", [])
            text = " ".join(item.get("event", "") for item in obs)
            help_needed = 1 if row.get("help_needed", False) else 0
            yield Sample(text=text, help_needed=help_needed)


def _length_bucket(n_tokens: int) -> str:
    if n_tokens <= 20:
        return "len_s"
    if n_tokens <= 60:
        return "len_m"
    return "len_l"


def _hash_features(text: str, dim: int) -> list[int]:
    """Feature indices with multiplicity for a tiny bag-of-n-grams encoder."""
    tokens = TOKEN_RE.findall(text.lower())
    feats: list[int] = []

    # unigram features
    for tok in tokens:
        feats.append(hash(f"u:{tok}") % dim)

    # bigram features add local context but remain cheap
    for i in range(len(tokens) - 1):
        feats.append(hash(f"b:{tokens[i]}_{tokens[i+1]}") % dim)

    # coarse length bucket often helps proactive triggering
    feats.append(hash(_length_bucket(len(tokens))) % dim)

    if not feats:
        feats = [0]
    return feats


def _reward(action: int, help_needed: int) -> float:
    return 1.0 if action == help_needed else -1.0


class PolicyBandit:
    """Two-action linear policy trained with REINFORCE + moving baseline."""

    def __init__(self, dim: int = 8192, lr: float = 0.05):
        self.dim = dim
        self.lr = lr
        # single score function for action=1; action=0 is implicit
        self.w = [0.0] * dim
        self.baseline = 0.0

    def logit(self, feats: list[int]) -> float:
        return sum(self.w[i] for i in feats) / max(1, len(feats))

    def prob_help(self, feats: list[int]) -> float:
        z = max(-20.0, min(20.0, self.logit(feats)))
        return 1.0 / (1.0 + math.exp(-z))

    def sample_action(self, feats: list[int], epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, 1)
        p = self.prob_help(feats)
        return 1 if random.random() < p else 0

    def predict(self, feats: list[int], threshold: float = 0.5) -> int:
        return 1 if self.prob_help(feats) >= threshold else 0

    def update(self, feats: list[int], action: int, reward: float, baseline_beta: float = 0.95):
        """REINFORCE gradient for Bernoulli policy with scalar baseline."""
        p = self.prob_help(feats)
        advantage = reward - self.baseline
        # grad(log pi(a|s)) for Bernoulli(logit): (a - p) * x
        coeff = self.lr * advantage * (action - p) / max(1, len(feats))
        for i in feats:
            self.w[i] += coeff
        self.baseline = baseline_beta * self.baseline + (1.0 - baseline_beta) * reward


def _compute_metrics(preds: list[int], labels: list[int]) -> dict:
    tp = fp = tn = fn = 0
    total_reward = 0.0
    for pred, y in zip(preds, labels):
        total_reward += _reward(pred, y)
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_reward": total_reward / max(1, len(preds)),
    }


def _split_train_val(data: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    data = data[:]
    rnd = random.Random(seed)
    rnd.shuffle(data)
    n_val = max(1, int(len(data) * val_ratio))
    val_data = data[:n_val]
    train_data = data[n_val:]
    if not train_data:
        train_data, val_data = data, data[:]
    return train_data, val_data


def _tune_threshold(model: PolicyBandit, val_data: list[Sample], dim: int) -> float:
    best_t = 0.5
    best_f1 = -1.0
    labels = [s.help_needed for s in val_data]
    val_feats = [_hash_features(s.text, dim) for s in val_data]
    for t in [i / 100 for i in range(20, 81, 2)]:
        preds = [model.predict(f, threshold=t) for f in val_feats]
        f1 = _compute_metrics(preds, labels)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


def train(
    train_path: str = "dataset/reward_data/train_data.jsonl",
    test_path: str = "dataset/reward_data/test_data.jsonl",
    episodes: int = 20,
    dim: int = 8192,
    lr: float = 0.05,
    epsilon_start: float = 0.20,
    epsilon_end: float = 0.02,
    val_ratio: float = 0.15,
    seed: int = 42,
    out: str = "eval/results/quick_rl_metrics.json",
):
    random.seed(seed)
    all_train = list(_iter_samples(train_path))
    test_data = list(_iter_samples(test_path))
    train_data, val_data = _split_train_val(all_train, val_ratio=val_ratio, seed=seed)

    model = PolicyBandit(dim=dim, lr=lr)

    # train
    for ep in range(episodes):
        random.shuffle(train_data)
        ratio = ep / max(1, episodes - 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * ratio
        for s in train_data:
            feats = _hash_features(s.text, dim)
            action = model.sample_action(feats, epsilon=epsilon)
            reward = _reward(action, s.help_needed)
            model.update(feats, action, reward)

    # threshold calibration on held-out validation
    threshold = _tune_threshold(model, val_data, dim=dim)

    # evaluate on test set
    test_feats = [_hash_features(s.text, dim) for s in test_data]
    test_labels = [s.help_needed for s in test_data]
    test_preds = [model.predict(feats, threshold=threshold) for feats in test_feats]
    metrics = _compute_metrics(test_preds, test_labels)
    metrics.update(
        {
            "threshold": threshold,
            "episodes": episodes,
            "dim": dim,
            "lr": lr,
            "val_ratio": val_ratio,
            "seed": seed,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
        }
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {out}")


if __name__ == "__main__":
    fire.Fire(train)
