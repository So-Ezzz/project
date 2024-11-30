```bash
.
├── README.md
├── data
│   ├── processed
│   │   ├── fft
│   │   ├── mel
│   │   ├── mfcc
│   │   └── stft
│   └── raw
│       ├── meta
│       │   └── esc50.csv
│       ├── train_set              # 训练数据集
│       └── val_set  			   # 验证数据集
├── experiments
│   └── param_test.py
├── main.py
├── results
│   ├── evaluation
│   └── plots
├── src
│   ├── evaluation
│   │   └── evaluate.py
│   ├── model
│   │   ├── fft_classifier.py
│   │   ├── mfcc_classifier.py
│   │   ├── scoring.py
│   │   └── stft_classifier.py
│   ├── preprocess
│   │   ├── extract_features.py
│   │   ├── fft.py
│   │   ├── mel.py
│   │   ├── mfcc.py
│   │   └── stft.py
│   └── utils
│       ├── audio_data.py
│       ├── compute_features_single.py
│       ├── deal_pkl.py
│       ├── load_intermediate_data.py
│       ├── plot_graphs.py
│       └── project_global.py
└── test.ipynb
```

# 项目流程文档

## 1. 数据加载

**目标**: 从 `data/raw/` 加载原始音频文件和元数据。**实现**:

- 
- 
- 
- 

---

## 2. 特征提取

**目标**: 从音频数据中提取特征（FFT、STFT、MFCC）。**实现**:

- 
- 
- 

---

## 3. 数据划分

**目标**: 将数据按 fold 分组，最后一个 fold 作为查询数据，前 4 个 fold 作为候选数据库。**实现**:

- 
- 
- 

---

## 4. 匹配与打分

**目标**: 为查询音频生成推荐列表，并为每个查询音频计算匹配分数。**实现**:

- 
- 

---

## 5. 精度评估

**目标**: 计算推荐列表中相同类别音频的匹配精度。**实现**:

- 
- 

---

## 6. 超参数实验（可选）

**目标**: 测试不同帧长和帧移设置对模型精度的影响。**实现**:

- 
- 

---

## 7. 结果可视化

**目标**: 将推荐精度和超参数实验结果可视化。**实现**:

- 
- 

