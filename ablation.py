"""
运行方式：python ablation.py
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score, f1_score, recall_score, precision_recall_curve
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb

DATA_PATH  = "data/creditcard.csv"
OUTPUT_DIR = "outputs"
SEED       = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

def best_thr_predict(proba, y_val):
    prec, rec, thresholds = precision_recall_curve(y_val, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    return thresholds[np.argmax(f1s[:-1])]

def load_data():
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    X = df.drop('Class', axis=1).copy()
    y = df['Class'].copy()
    scaler = RobustScaler()
    X[['Amount','Time']] = scaler.fit_transform(X[['Amount','Time']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
    return X_tr, X_val, X_test, y_tr, y_val, y_test

def score(model, X_tr, y_tr, X_val, y_val, X_te, y_te):
    model.fit(X_tr, y_tr)
    val_p = model.predict_proba(X_val)[:,1]
    thr   = best_thr_predict(val_p, y_val)
    proba = model.predict_proba(X_te)[:,1]
    pred  = (proba >= thr).astype(int)
    return {
        'F1-Score': round(f1_score(y_te, pred), 3),
        'PR-AUC':   round(average_precision_score(y_te, proba), 3),
        'Recall':   round(recall_score(y_te, pred), 3),
    }

def main():
    if not os.path.exists(DATA_PATH):
        print(f"找不到 {DATA_PATH}"); return

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()
    spw = min(round((y_tr==0).sum()/(y_tr==1).sum(), 1), 50)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    meta = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)

    base_cs = [
        ('cs_lr',   LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)),
        ('cs_rf',   RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=SEED)),
        ('cs_xgb',  xgb.XGBClassifier(n_estimators=200, scale_pos_weight=spw, eval_metric='logloss', random_state=SEED, n_jobs=-1)),
        ('cs_lgbm', lgb.LGBMClassifier(n_estimators=200, scale_pos_weight=spw, random_state=SEED, n_jobs=-1, verbose=-1)),
    ]
    base_plain = [
        ('lr',   LogisticRegression(max_iter=1000, random_state=SEED)),
        ('rf',   RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)),
        ('xgb',  xgb.XGBClassifier(n_estimators=200, eval_metric='logloss', random_state=SEED, n_jobs=-1)),
        ('lgbm', lgb.LGBMClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1)),
    ]

    smt = SMOTETomek(random_state=SEED)
    X_tr_smt, y_tr_smt = smt.fit_resample(X_tr, y_tr)

    configs = {
        'CS-Stack（完整）':       (StackingClassifier(estimators=base_cs,    final_estimator=meta, cv=cv, stack_method='predict_proba', n_jobs=-1), X_tr_smt, y_tr_smt),
        'w/o 代价敏感':          (StackingClassifier(estimators=base_plain, final_estimator=LogisticRegression(max_iter=1000,random_state=SEED), cv=cv, stack_method='predict_proba', n_jobs=-1), X_tr, y_tr),
        'w/o SMOTE-Tomek':       (StackingClassifier(estimators=base_cs,    final_estimator=meta, cv=cv, stack_method='predict_proba', n_jobs=-1), X_tr, y_tr),
        'w/o Stacking（平均集成）': None,
        '仅 CS-LGBM（单一最优）': (lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=63, scale_pos_weight=spw, random_state=SEED, n_jobs=-1, verbose=-1), X_tr_smt, y_tr_smt),
    }

    print("=" * 55)
    print("消融实验结果")
    print("=" * 55)
    ablation = {}

    for name, cfg in configs.items():
        print(f"  训练：{name} ...", end=' ', flush=True)
        if name == 'w/o Stacking（平均集成）':
            # 各基础模型概率平均
            probas = []
            for _, m in base_cs:
                m.fit(X_tr_smt, y_tr_smt)
                probas.append(m.predict_proba(X_te)[:,1])
            avg_proba = np.mean(probas, axis=0)
            val_probas = []
            for _, m in base_cs:
                val_probas.append(m.predict_proba(X_val)[:,1])
            avg_val = np.mean(val_probas, axis=0)
            thr  = best_thr_predict(avg_val, y_val)
            pred = (avg_proba >= thr).astype(int)
            metrics = {
                'F1-Score': round(f1_score(y_te, pred), 3),
                'PR-AUC':   round(average_precision_score(y_te, avg_proba), 3),
                'Recall':   round(recall_score(y_te, pred), 3),
            }
        else:
            model, Xtr_, ytr_ = cfg
            metrics = score(model, Xtr_, ytr_, X_val, y_val, X_te, y_te)
        ablation[name] = metrics
        print(f"F1={metrics['F1-Score']}  PR-AUC={metrics['PR-AUC']}  Recall={metrics['Recall']}")

    df = pd.DataFrame(ablation).T
    print("\n" + df.to_string())
    df.to_csv(f'{OUTPUT_DIR}/ablation_results.csv')

    # 绘图
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    w = 0.25
    ax.barh(x+w,  df['F1-Score'], w, label='F1-Score', color='steelblue')
    ax.barh(x,    df['PR-AUC'],   w, label='PR-AUC',   color='darkorange')
    ax.barh(x-w,  df['Recall'],   w, label='Recall',   color='seagreen')
    ax.set_yticks(x); ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlabel('Score'); ax.set_title('Ablation Study Results')
    ax.legend(); ax.set_xlim(0.80, 0.98); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ablation.png', dpi=150)
    plt.close()
    print(f"\n消融实验图已保存：{OUTPUT_DIR}/ablation.png")

if __name__ == '__main__':
    main()
