# Phira / RPE 谱面自动排键模型

根据谱面节奏信息，预测每个音符在判定线上的水平位置（x 坐标），用于自动生成排键布局。
项目经历了两代范式：早期 GRU 位移分类采样，当前为 **Transformer 轨迹回归（直接输出 x 坐标）**。

> 本仓库仅包含代码脚本；训练谱面数据、模型权重、推理产物与排键图均不入库（见末尾说明）。

## 模型现状

### 当前主线：Transformer 轨迹回归

- 输入：128 音窗口 × 8 维特征（`dt, phase, density, pos, ty, durs, chord_x, level_norm`）
- 结构：`TransformerEncoder(d_model=128, nhead=4, layers=3)`，一次前向并行输出 128 个 x
- 损失：`MSE(谱师 x) + λ_smooth·相邻位移平滑 + λ_bound·边界惩罚`（选模/早停用 val 纯 MSE）
- 推理：滑窗（128 / stride 64）重叠平均 + 尾窗补覆盖，两遍推理（第一遍粗排 → 第二遍用同刻粗排 x 均值精排）
- 特征口径：训练与推理共用 `traj_common.py`（整线计算后切窗），音符类型仅 Tap(1)/Hold(2)
- 推理输出每音符带 `"src": "model" | "fallback"`（fallback=线内 <4 音沿用原谱 x），评估据此统计模型真实覆盖率

### 历史路线（保留作对照）

- GRU 位移分类：`train_layout*.py` / `infer_layout*.py`（位移分桶 + softmax 采样）
- 双押上下文：`train_chord.py` / `infer_chord.py`（加入同刻双押 x 特征）

## 目录与脚本分组

| 分组 | 脚本 | 说明 |
| --- | --- | --- |
| 谱面采集 | `phira_fetch.py` `phira_retry.py` `phira_refetch.py` `phira_level_fetch.py` `batch_fetch.py` `patch_fetch.py` `scan_list.py` | 拉取 phira 谱面列表 / 下载 zip / 解析 RPE JSON / 校验重抓 / 难度缓存 |
| 数据构建 | `build_traj_npz.py` `build_chord_npz.py` `build_train_npz*.py` `build_level_cache.py` `run_bpm_extract.py` `pec_parse.py` | 解析谱面 → 特征/标签 → npz 数据集 |
| 训练 | `train_traj.py` `train_chord.py` `train_layout*.py` `ft_train_*.py` `ft_exp_ab.py` | 轨迹 / 分类 / 双押 / 微调各版本训练 |
| 推理 | `infer_traj.py` `infer_chord.py` `infer_layout*.py` | 各模型推理输出排键 JSON |
| 规则排键 | `rule_charter.py` `rule_charter2.py` | 固定对称键位的确定性规则排键（模型对照基线） |
| 评估 | `eval_traj_*.py` `eval_layout_*.py` `chord_eval.py` `chord_gap_stats.py` `density_gap_corr.py` `verify_*.py` | MAE / 双押间距 / 分布相关性评估与校验 |
| 可视化 | `viz_plain.py` `viz_editor.py` `viz_real_speed.py` | 排键布局图渲染 |
| 工具 | `easing_lib.py` `finalize.py` `density_gap_corr.py` | 曲线插值、收尾处理、相关性统计 |

## 环境依赖

- Python 3.10+
- `numpy`、`torch`（训练/推理）、`pyyaml`（解析 info.yml）、`Pillow`（出图）、`matplotlib`（viz 出图）

```bash
pip install numpy torch pyyaml pillow matplotlib
```

## 快速开始

1. **修改路径**：各脚本顶部的 `BASE` 常量改为你本地数据目录（`parsed/`、`bpm/`、`level_cache.json` 所在处）；产物路径（`--out`/`--data`/`--ckpt`）相对路径一律相对当前工作目录，绝对路径原样。
2. **采集谱面**：运行 `phira_fetch.py` 拉取并解析谱面到 `parsed/` 与 `bpm/`。
3. **构建数据集**：

```bash
python3 build_traj_npz.py --out train_data_traj --aug-blank 0.25
```

4. **训练**：

```bash
python3 train_traj.py --data train_data_traj --ckpt ckpt_traj --epochs 40
```

5. **推理**：

```bash
python3 infer_traj.py <fid> --ckpt ckpt_traj --out infer_out.json
```

6. **出图**：

```bash
python3 viz_plain.py <fid> infer_out.json <tag>
```

## 归一化口径

- 水平坐标：`X_SCALE = 675.0`，标签 `xnorm = clip(x / 675, ±1.2)`
- 难度归一化：`level_norm = level / 25.0`
- 音符类型过滤：`type ∈ {1, 2}`（Tap / Hold）

## 已知局限

- **空白谱展开**：训练样本均为谱师排好的完整轨迹，模型对原谱 x 全 0 的空白谱倾向输出窄带、不会向两侧铺开；`--aug-blank` 可做从 x=0 起步的增强缓解。
- **连续值 vs 离散键位**：谱师排键通常落在少数固定键位上，回归模型输出连续 x，必要时可在推理后处理做键位量化。
- 社区自制谱质量参差，混入低质量谱面会稀释训练信号。

## 不入库内容

以下目录 / 文件体积大或含本地数据，均已在 `.gitignore` 中排除：

- 谱面数据：`parsed/` `bpm/` `phira_full/` `subsets/` `parsed_bak_*/`
- 数据集：`train_data*/` `*.npz`
- 模型权重：`layout_ckpt*/` `ckpt_*/`
- 产物：`infer_*.json` `*.png` `*.log` `*.html`

## 版权与免责

代码仅供学习研究。谱面版权归原作者与谱师所有；使用他人谱面时请遵守对应平台与作者的授权条款。
