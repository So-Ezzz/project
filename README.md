```bash
project/
├── data/                  # 数据相关
│   ├── raw/               # 原始音频数据
│   ├── processed/         # 预处理后的特征数据
├── src/                   # 核心代码
│   ├── preprocess/        # 数据预处理代码
│   │   ├── fft.py         # 实现FFT算法
│   │   ├── stft.py        # 实现STFT算法
│   │   ├── mfcc.py        # 实现MFCC提取
│   ├── model/             # 打分模型
│   │   ├── scoring.py     # 实现打分逻辑
│   ├── evaluation/        # 评估代码
│   │   ├── evaluate.py    # 精度评估逻辑
│   ├── utils/             # 工具代码
│   │   ├── helpers.py     # 数据加载、文件处理等工具函数
├── results/               # 结果存放
│   ├── evaluation/        # 精度评估结果
│   ├── plots/             # 可视化结果
├── experiments/           # 不同帧移、帧长超参数实验代码
│   ├── param_test.py      # 不同参数实验的运行脚本
├── README.md              # 项目说明

```

```
(SoEzzz) PS D:\SoEzz\24down\HW\数字信号处理\大作业\project> tree
├───data
│   ├───processed
│   │   ├───fft
│   │   ├───mfcc
│   │   └───stft
│   └───raw
│       └───meta
├───experiments
├───results
│   ├───evaluation
│   └───plots
└───src
    ├───evaluation
    ├───model
    ├───preprocess
    └───utils
        └───__pycache__

```

# 项目流程文档

## 1. 数据加载

**目标**: 从 `data/raw/` 加载原始音频文件和元数据。**实现**:

- 调用 `src/utils/helpers.py` 中的函数加载音频和元数据。
- 音频数据存储为字典，元数据存储为 DataFrame。**关键输出**:
- **音频数据**: `{filename: audio_array}`
- **元数据**: 包含音频类别、fold 分组等信息的 `DataFrame`。

---

## 2. 特征提取

**目标**: 从音频数据中提取特征（FFT、STFT、MFCC）。**实现**:

- 调用 `src/preprocess/fft.py`、`stft.py` 和 `mfcc.py` 中的函数。
- 提取的特征存储到 `data/processed/` 中，供后续步骤使用。**关键输出**:
- **特征数据**: 以 `{filename: feature_array}` 的形式存储。

---

## 3. 数据划分

**目标**: 将数据按 fold 分组，最后一个 fold 作为查询数据，前 4 个 fold 作为候选数据库。**实现**:

- 使用元数据中的 `fold` 列划分音频文件。**关键输出**:
- **查询数据**: 一个包含音频文件名和特征的字典。
- **候选数据库**: 一个包含音频文件名和特征的字典。

---

## 4. 匹配与打分

**目标**: 为查询音频生成推荐列表，并为每个查询音频计算匹配分数。**实现**:

- 调用 `src/model/scoring.py` 中的函数，使用欧氏距离或余弦相似度计算匹配分数。**关键输出**:
- **推荐列表**: 包含每个查询音频的 Top10 和 Top20 匹配结果。

---

## 5. 精度评估

**目标**: 计算推荐列表中相同类别音频的匹配精度。**实现**:

- 调用 `src/evaluation/evaluate.py`，比较推荐列表与真实类别，计算 Top10 和 Top20 的精度。**关键输出**:
- **精度结果**: 包含 Top10 和 Top20 精度的字典或 `DataFrame`。

---

## 6. 超参数实验（可选）

**目标**: 测试不同帧长和帧移设置对模型精度的影响。**实现**:

- 调用 `experiments/param_test.py`，以不同参数重新提取特征，重复步骤 4 和 5。**关键输出**:
- **实验结果**: 不同参数下的精度结果。

---

## 7. 结果可视化

**目标**: 将推荐精度和超参数实验结果可视化。**实现**:

- 使用 `matplotlib` 或其他可视化工具绘制折线图或柱状图。**关键输出**:
- **图表文件**: 存储在 `results/plots/` 中。

```
array([1.37047729e+01, 4.88058402e+00, 7.93867680e+00, ...,
       1.60291178e-02, 8.83011920e-03, 9.16488209e-03])
```
