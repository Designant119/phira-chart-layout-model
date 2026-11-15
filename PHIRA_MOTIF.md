# Phira 高质量谱面与 motif 库

## 抓谱

```bash
python3 phira_corpus.py --outdir /path/corpus --min-diff 11 --max-diff 16.5 --max-charts 100
```

只从 Phira 的 `ranked` + `stable` 池取，按难度和 `rating` 排序；输出 `parsed/` 和 `bpm/`，
不下载音频。谱面文件不入库。

本次抓取：ranked 589 + stable 633，去重 636，难度 11–16.5 候选 570，取前 100 张。

## motif 索引

```bash
python3 build_motif_index.py \
  --roots /path/corpus,/path/data_fixed \
  --out /path/motif_index
```

每条判定线切 12 音、stride 4。索引里两套东西分开：

- `Q`：节奏 query（dt / 同刻和弦 / 密度 / 类型 one-hot / Hold 时长），96 维；
- `X/R`：人类谱的绝对 x 和相对首音的 x pattern。

当前索引：270 张谱、80,978 个 motif；jack 率 13.8%、Hold 率 11.0%、
Drag 5.3%、Flick 23.8%。

## 检索

```bash
python3 query_motif.py --index /path/motif_index --fid 6882 --line 0 --topk 5
```

按节奏相似度找最像的人类 pattern，输出 `x` 和音符类型。生成时可以拿它当先验：
先检索相似节奏段，再让模型做局部修改。
