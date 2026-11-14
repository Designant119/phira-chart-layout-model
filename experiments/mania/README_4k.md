# 4K mania 全量实验

## 数据

数据不入库（体积大），只保留构建脚本和权重。把 `osu/*.osu` 放到 `full/osu/`，然后：

```bash
cd experiments/mania/full
python3 build_4k_full.py          # -> mania_4k_full.npz + split_4k_full.json
python3 train_4k_full.py --epochs 20   # -> ckpt_mania4k_full.pt
python3 finetune_4k_ss.py --epochs 5   # -> ckpt_mania4k_full_ss.pt
```

全量数据：734 张 4K 谱、984,157 个音，按谱 8:1:1 切成 587/73/74 张；
训练/验证/测试音数 787,769 / 103,456 / 92,932。

## 特征（18 维）

`dt, chord, dens, hold, dur, streak, prev one-hot(5), prev2 one-hot(5), lastL, lastR`

- `streak` 是“进入当前音之前的连打长度”，**不能**用当前音列号算，否则标签泄漏；
- 4K 左手列 0/1，右手列 2/3；
- 生成列映射到 Phira x = `-495 / -165 / 165 / 495`。

## 权重

| 文件 | 说明 | val / test |
|---|---|---|
| `weights/ckpt_mania4k_full_honest.pt` | 全量 734 张，诚实 streak 口径，当前主权重 | 69.85% / 67.62% |
| `weights/ckpt_mania4k_full_ss.pt` | scheduled sampling 微调 5 epoch，未采用 | 69.62% / 66.92% |
| `weights/ckpt_mania4k_legacy.pt` | 早期 165 张 4K，旧 streak 口径，仅作对照 | 67.35% / 67.33% |
| `weights/mlp_4k.pth` | MLP 基线，旧口径 | 66.25% / 66.42% |
| `weights/ckpt_4k_early.pth` | 早期 4K Transformer 基线 | val 59.77% |

多数类基线约 25.5%。测试谱 `5960548` 的 80–120s 段生成 AR：诚实版权重约 35.6%，
旧权重约 40%，但旧权重训练特征有泄漏，数字不可直接比。

## 已知问题

- 逐音自回归仍会误差滚雪球，AR 只有 35% 左右；
- 生成器和训练窗口已对齐（历史放头部），生成器的 streak 状态也已改成“进入当前音前”的口径；
- 和弦仍用贪心互斥，不是联合预测。
