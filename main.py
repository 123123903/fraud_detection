"""
信用卡欺诈检测 - 主程序（v3）
基于多算法融合的信用卡欺诈检测研究（CS-Stack框架）

运行方式：python main.py
数据集：data/creditcard.csv
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, precision_recall_curve
)
import xgboost as xgb
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_PATH  = "data/creditcard.csv"
OUTPUT_DIR = "outputs"
SEED       = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)


def best_threshold(proba_val, y_val):
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    return thr[np.argmax(f1s[:-1])]


# ── 1. 数据 ──────────────────────────────
def load_data():
    print("=" * 60)
    print("【1】加载数据集...")
    df = pd.read_csv(DATA_PATH)
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"    删除重复 {n_before-len(df)} 条，剩余 {len(df)} 条")
    print(f"    欺诈：{df['Class'].sum()} 条（{df['Class'].mean()*100:.3f}%）")

    X = df.drop('Class', axis=1).copy()
    y = df['Class'].copy()
    sc = RobustScaler()
    X[['Amount','Time']] = sc.fit_transform(X[['Amount','Time']])

    # 训练/验证/测试 = 72% / 8% / 20%
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.10, stratify=y_tmp, random_state=SEED)

    print(f"    训练 {len(X_tr)} | 验证 {len(X_val)} | 测试 {len(X_test)}")
    return X_tr, X_val, X_test, y_tr, y_val, y_test


# ── 2. 模型定义 ────────────────────────────
def build_models():
    """
    用 class_weight='balanced' 统一处理不平衡，
    彻底避免 scale_pos_weight 的数值不稳定问题。
    SMOTE-Tomek 作为独立数据层策略，
    包在 ImbPipeline 里防止泄露。
    """
    models = {}

    # 基线：无任何不平衡处理
    models['LR（基线）'] = LogisticRegression(
        max_iter=1000, random_state=SEED)

    # 代价敏感逻辑回归
    models['CS-LR'] = LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=SEED)

    # 代价敏感随机森林
    models['CS-RF'] = RandomForestClassifier(
        n_estimators=300, class_weight='balanced',
        n_jobs=-1, random_state=SEED)

    # SMOTE-Tomek + XGBoost（数据层策略）
    models['ST-XGB'] = ImbPipeline([
        ('resample', SMOTETomek(random_state=SEED)),
        ('clf', xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=SEED, n_jobs=-1))
    ])

    # 代价敏感 GradientBoosting（替代 LightGBM，行为更稳定）
    models['CS-GBM'] = ImbPipeline([
        ('resample', SMOTETomek(random_state=SEED)),
        ('clf', GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=SEED))
    ])

    return models


def build_cs_stack():
    base = [
        ('cs_lr', LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=SEED)),
        ('cs_rf', RandomForestClassifier(
            n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=SEED)),
        ('cs_xgb', xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=10,          # 适中权重，不过激
            eval_metric='logloss', random_state=SEED, n_jobs=-1)),
        ('cs_gbm', GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=SEED)),
    ]
    meta = LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=SEED)

    return StackingClassifier(
        estimators=base, final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        stack_method='predict_proba', n_jobs=1)


# ── 3. 评估 ───────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_val, y_val, X_test, y_test):
    print(f"  训练：{name} ...", end=' ', flush=True)
    model.fit(X_tr, y_tr)

    val_p  = model.predict_proba(X_val)[:, 1]
    thr    = best_threshold(val_p, y_val)
    test_p = model.predict_proba(X_test)[:, 1]
    pred   = (test_p >= thr).astype(int)

    m = {
        'Precision': round(precision_score(y_test, pred, zero_division=0), 3),
        'Recall':    round(recall_score(y_test, pred), 3),
        'F1-Score':  round(f1_score(y_test, pred), 3),
        'PR-AUC':    round(average_precision_score(y_test, test_p), 3),
        'ROC-AUC':   round(roc_auc_score(y_test, test_p), 3),
        'MCC':       round(matthews_corrcoef(y_test, pred), 3),
    }
    print(f"完成  F1={m['F1-Score']}  PR-AUC={m['PR-AUC']}  Recall={m['Recall']}")
    return m, test_p


# ── 4. 可视化 ─────────────────────────────
def plot_pr_curves(proba_dict, y_test):
    plt.figure(figsize=(10, 7))
    colors = plt.cm.tab10.colors
    for i, (name, proba) in enumerate(proba_dict.items()):
        prec, rec, _ = precision_recall_curve(y_test, proba)
        auc = average_precision_score(y_test, proba)
        lw = 2.5 if 'Stack' in name else 1.2
        ls = '-'  if 'Stack' in name else '--'
        plt.plot(rec, prec, color=colors[i % 10], lw=lw, ls=ls,
                 label=f'{name} (PR-AUC={auc:.3f})')
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pr_curves.png', dpi=150)
    plt.close()


def plot_metrics_bar(df):
    metrics = ['F1-Score', 'PR-AUC', 'Recall', 'ROC-AUC']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))
    for ax, m in zip(axes.flatten(), metrics):
        vals = df[m]
        bars = ax.barh(df.index, vals, color=colors, edgecolor='grey', lw=0.5)
        ax.set_xlabel(m, fontsize=11)
        ax.set_title(m, fontsize=12, fontweight='bold')
        ax.set_xlim(max(0, min(vals)*0.95), 1.02)
        for bar, v in zip(bars, vals):
            ax.text(v+0.002, bar.get_y()+bar.get_height()/2,
                    f'{v:.3f}', va='center', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
    plt.suptitle('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/metrics_comparison.png', dpi=150)
    plt.close()


def plot_cm(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Normal','Fraud'])
    ax.set_yticklabels(['Normal','Fraud'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix - CS-Stack', fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j]>cm.max()/2 else 'black',
                    fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_CS-Stack.png', dpi=150)
    plt.close()


# ── 5. 主流程 ─────────────────────────────
def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] 找不到 {DATA_PATH}")
        print("请下载：https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return

    X_tr, X_val, X_test, y_tr, y_val, y_test = load_data()

    # SMOTE-Tomek 预处理训练集（供不含 Pipeline 的模型使用）
    print("\n    SMOTE-Tomek 过采样训练集...", end=' ', flush=True)
    smt = SMOTETomek(random_state=SEED)
    X_tr_smt, y_tr_smt = smt.fit_resample(X_tr, y_tr)
    print(f"完成（{len(X_tr)} → {len(X_tr_smt)} 条）")

    models = build_models()
    models['CS-Stack（本文）'] = build_cs_stack()

    print("\n【2】训练与评估...")
    results, probas = {}, {}
    stack_pred = None

    for name, model in models.items():
        # Pipeline 内含 SMOTE 的用原始数据；其余用已过采样数据
        use_smt = ('Stack' in name) or ('GBM' in name) or ('XGB' in name)
        Xtr_ = X_tr if use_smt else X_tr_smt
        ytr_ = y_tr if use_smt else y_tr_smt

        m, proba = evaluate(name, model, Xtr_, ytr_, X_val, y_val, X_test, y_test)
        results[name] = m
        probas[name]  = proba
        if name == 'CS-Stack（本文）':
            val_p = model.predict_proba(X_val)[:, 1]
            thr   = best_threshold(val_p, y_val)
            stack_pred = (proba >= thr).astype(int)

    df = pd.DataFrame(results).T
    print("\n" + "=" * 60)
    print("【3】汇总结果")
    print("=" * 60)
    print(df.to_string())
    df.to_csv(f'{OUTPUT_DIR}/results.csv')

    print("\n【4】生成图表...")
    plot_pr_curves(probas, y_test)
    plot_metrics_bar(df)
    plot_cm(y_test, stack_pred)

    print("\n【5】CS-Stack 详细分类报告")
    print(classification_report(y_test, stack_pred, target_names=['Normal','Fraud']))
    print(f"\n全部结果已保存至 {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
