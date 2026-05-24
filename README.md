# fraud_detection

## 项目结构

```
credit-fraud-detection/
├── main.py            # 主程序：训练所有模型并对比评估
├── ablation.py        # 消融实验：验证各模块的贡献
├── requirements.txt   # 依赖列表
├── data/              # 数据目录（需手动下载数据集）
│   └── creditcard.csv
└── outputs/           # 运行后自动生成，存放结果和图表
    ├── results.csv
    ├── ablation_results.csv
    ├── pr_curves.png
    ├── metrics_comparison.png
    ├── ablation.png
    └── confusion_matrix_CS-Stack.png
```

---

## 数据集获取

本项目使用 Kaggle 公开数据集：**Credit Card Fraud Detection**

1. 访问：https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. 点击 **Download** 下载 `creditcard.csv`（约 150 MB）
3. 将文件放入项目根目录下的 `data/` 文件夹：
   ```
   mkdir data
   mv creditcard.csv data/
   ```

**数据集说明**：
- 共 284,807 条交易记录，其中欺诈 492 条（占比 0.172%）
- 特征：Time、V1-V28（PCA 匿名化）、Amount、Class（标签）
- 来源：欧洲持卡人 2013 年 9 月真实交易数据

---

## 环境配置

### 方式一：pip 直接安装

```bash
# Python 3.9 ~ 3.11 均可
pip install -r requirements.txt
```

### 方式二：conda 虚拟环境

```bash
conda create -n fraud python=3.10
conda activate fraud
pip install -r requirements.txt
```

**主要依赖版本**：

| 库 | 版本 |
|----|------|
| scikit-learn | ≥ 1.3.0 |
| xgboost | ≥ 2.0.0 |
| lightgbm | ≥ 4.1.0 |
| imbalanced-learn | ≥ 0.11.0 |
| pandas | ≥ 2.0.0 |
| matplotlib | ≥ 3.7.0 |

---

## 运行说明

### 第一步：主实验（9 种算法对比）

```bash
python main.py
```

**输出内容**：
- 控制台打印所有算法的 Precision / Recall / F1 / PR-AUC / ROC-AUC / MCC
- `outputs/results.csv`：结果汇总表
- `outputs/pr_curves.png`：PR 曲线对比图
- `outputs/metrics_comparison.png`：各指标柱状图
- `outputs/confusion_matrix_CS-Stack.png`：混淆矩阵

预计运行时间：**15～30 分钟**（取决于 CPU 核数）

### 第二步：消融实验

```bash
python ablation.py
```

**输出内容**：
- 控制台打印各消融配置的结果
- `outputs/ablation_results.csv`：消融结果表
- `outputs/ablation.png`：消融实验对比图

---

## 方法简介

### CS-Stack 框架（核心创新）

```
训练数据（含 SMOTE-Tomek 过采样）
        │
        ├─── CS-LR   (代价敏感逻辑回归)  ──┐
        ├─── CS-RF   (代价敏感随机森林)  ──┤
        ├─── CS-XGB  (代价敏感 XGBoost)  ──┤─→ 元特征（OOF 预测概率）
        └─── CS-LGBM (代价敏感 LightGBM) ──┘
                                            │
                                      元学习器（逻辑回归）
                                            │
                                        最终预测
```
三层创新组合：
1. **数据层**：SMOTE-Tomek 过采样 + 清噪声
2. **算法层**：代价敏感权重（scale_pos_weight ≈ 577）
3. **模型层**：5 折 OOF Stacking 集成

### 对比算法

| 算法 | 说明 |
|------|------|
| LR（基线） | 普通逻辑回归，无任何不平衡处理 |
| CS-LR | 代价敏感逻辑回归（class_weight='balanced'）|
| CS-RF | 代价敏感随机森林 |
| ST-XGB | SMOTE-Tomek + XGBoost |
| CS-LGBM | 代价敏感 LightGBM |
| **CS-Stack** | 本文提出的融合框架 |

---

## 主要实验结果

| 算法 | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
|------|-----------|--------|----------|--------|---------|
| LR（基线） | 0.843 | 0.716 | 0.774 | 0.762 | 0.941 |
| CS-LGBM | 0.874 | 0.883 | 0.878 | 0.871 | 0.978 |
| **CS-Stack** | **0.854** | **0.913** | **0.882** | **0.891** | **0.982** |

---

## 参考文献

1. Dal Pozzolo et al. (2015). Calibrating probability with undersampling for unbalanced classification.
2. Chawla et al. (2002). SMOTE: Synthetic minority over-sampling technique.
3. Chen & Guestrin (2016). XGBoost: A scalable tree boosting system.
4. Ke et al. (2017). LightGBM: A highly efficient gradient boosting decision tree.
5. ai-cases.com. [Fraud Detection for Banking](https://ai-cases.com/banking/fraud-detection/)
