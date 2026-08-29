import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

# 配置
RANDOM_STATE = 42
CV_FOLDS = 5
N_TRIALS_LGB = 30
N_TRIALS_XGB = 30
N_TRIALS_RF = 30
CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

class ManualVoting:
    def __init__(self, models_list):
        self.models_list = models_list
    def predict_proba(self, X):
        probs = np.array([model.predict_proba(X) for model in self.models_list])
        return np.mean(probs, axis=0)
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
# 数据加载
def load_data(train_csv='X_train_fe.csv', val_csv='X_val_fe.csv', test_csv='X_test_fe.csv',
              y_train_csv='y_train.csv', y_val_csv='y_val.csv', y_test_csv='y_test.csv',
              header_features=0, header_labels=None):
    X_train = pd.read_csv(train_csv, header=header_features)
    X_val = pd.read_csv(val_csv, header=header_features)
    X_test = pd.read_csv(test_csv, header=header_features)
    y_train = pd.read_csv(y_train_csv, header=header_labels).values.ravel()
    y_val = pd.read_csv(y_val_csv, header=header_labels).values.ravel()
    y_test = pd.read_csv(y_test_csv, header=header_labels).values.ravel()
    return X_train, X_val, X_test, y_train, y_val, y_test

# 参数空间定义
def lgb_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

def xgb_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }

def rf_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

# 目标函数：最大化PR-AUC（average_precision）
def objective_generic(trial, model_class, param_func, X_train, y_train):
    model = model_class(**param_func(trial))
    # 使用交叉验证最大化 PR-AUC
    scores = cross_val_score(model, X_train, y_train, cv=CV, scoring='average_precision', n_jobs=-1)
    return scores.mean()

def objective_lgb(trial, X_train, y_train):
    return objective_generic(trial, lgb.LGBMClassifier, lgb_params, X_train, y_train)

def objective_xgb(trial, X_train, y_train):
    return objective_generic(trial, xgb.XGBClassifier, xgb_params, X_train, y_train)

def objective_rf(trial, X_train, y_train):
    return objective_generic(trial, RandomForestClassifier, rf_params, X_train, y_train)

# 超参数优化函数
def optimize_model(objective_func, model_name, X_train, y_train, n_trials):
    print(f"开始优化 {model_name} (试验次数: {n_trials})")
    study = optuna.create_study(direction='maximize', study_name=model_name)
    study.optimize(
        lambda trial: objective_func(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )
    print(f"\n最佳 {model_name} PR-AUC: {study.best_value:.4f}")
    print(f"\n最佳参数: {study.best_params}")
    return study.best_params

# 训练与集成
def train_models(X_train, y_train, best_params):
    """
    基模型只在Stacking训练一次，Voting复用
    """
    print("训练Stacking集成")
    
    # 1. 创建基模型实例
    lgb_base = lgb.LGBMClassifier(**best_params['LightGBM'], random_state=RANDOM_STATE, n_jobs=-1)
    xgb_base = xgb.XGBClassifier(**best_params['XGBoost'], random_state=RANDOM_STATE, n_jobs=-1,
                                 eval_metric='logloss', use_label_encoder=False)
    rf_base = RandomForestClassifier(**best_params['RandomForest'], random_state=RANDOM_STATE, n_jobs=-1)

    # 2. 训练 Stacking
    meta_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    stacking_clf = StackingClassifier(
        estimators=[('lgb', lgb_base), ('xgb', xgb_base), ('rf', rf_base)],
        final_estimator=meta_model,
        cv=CV_FOLDS,
        stack_method='predict_proba'
    )
    stacking_clf.fit(X_train, y_train)

    # 3. 提取 Stacking 最终拟合的基模型
    fitted_models = stacking_clf.named_estimators_ 

    # 4. 构建 Voting
    voting_model = ManualVoting(list(fitted_models.values()))

    # 5. 返回模型字典
    models = {
        'LightGBM': fitted_models['lgb'],
        'XGBoost': fitted_models['xgb'],
        'RandomForest': fitted_models['rf'],
        'Voting': voting_model,
        'Stacking': stacking_clf
    }
    print("集成模型构建完成")
    return models

# 评估函数
def evaluate_model(model, X, y, name, set_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        '模型': name,
        '数据集': set_name,
        '准确率': accuracy_score(y, y_pred),
        '精确率': precision_score(y, y_pred),
        '召回率': recall_score(y, y_pred),
        'F1分数': f1_score(y, y_pred),
        'AUC-ROC': roc_auc_score(y, y_prob)
    }

def evaluate_models(models, X_val, y_val, X_test, y_test):
    results = []
    for name, model in models.items():
        results.append(evaluate_model(model, X_val, y_val, name, 'Validation'))
        results.append(evaluate_model(model, X_test, y_test, name, 'Test'))
    return pd.DataFrame(results)

# 阈值优化
def find_best_threshold(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
    # 避免除以零
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1

def evaluate_with_threshold(model, X, y, threshold, name):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        '模型': name,
        '阈值': threshold,
        '准确率': accuracy_score(y, y_pred),
        '精确率': precision_score(y, y_pred),
        '召回率': recall_score(y, y_pred),
        'F1分数': f1_score(y, y_pred),
        'AUC-ROC': roc_auc_score(y, y_prob)
    }

# 绘图函数
def plot_curves(models, X_val, y_val, title_suffix='Validation', save_path=None):
    plt.figure(figsize=(12, 5))
    # ROC
    plt.subplot(1, 2, 1)
    for name, model in models.items():
        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc_val = roc_auc_score(y_val, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel('假阳性率')
    plt.ylabel('真正例率')
    plt.title(f'ROC曲线 ({title_suffix})')
    plt.legend()

    # PR
    plt.subplot(1, 2, 2)
    for name, model in models.items():
        y_prob = model.predict_proba(X_val)[:, 1]
        prec, rec, _ = precision_recall_curve(y_val, y_prob)
        plt.plot(rec, prec, label=name)
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title(f'PR曲线 ({title_suffix})')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return plt.gcf()

def plot_confusion_matrices(models, X_val, y_val, title_suffix='Validation', save_path=None):
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name} ({title_suffix})')
        ax.set_xlabel('预测值')
        ax.set_ylabel('真实值')

    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def run_secondary_optimization(
    do_optimize=True,
    n_trials_lgb=N_TRIALS_LGB,
    n_trials_xgb=N_TRIALS_XGB,
    n_trials_rf=N_TRIALS_RF,
    load_auc_models=True,
    auc_models_path='best_models.pkl',
    auc_params_path='best_params.json',
    f1_params_save_path='best_params_f1.json',
    f1_models_save_path='best_models_f1.pkl',
    results_save_path='comparison_results.csv'
):


    print("——"*30)
    print("——"*30)

    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 2. 优化 PR-AUC
    if do_optimize:
        print("\n优化 PR-AUC")
        best_lgb = optimize_model(objective_lgb, 'LightGBM_F1', X_train, y_train, n_trials_lgb)
        best_xgb = optimize_model(objective_xgb, 'XGBoost_F1', X_train, y_train, n_trials_xgb)
        best_rf = optimize_model(objective_rf, 'RandomForest_F1', X_train, y_train, n_trials_rf)
        best_params_f1 = {
            'LightGBM': best_lgb,
            'XGBoost': best_xgb,
            'RandomForest': best_rf
        }
        # 保存F1参数
        with open(f1_params_save_path, 'w') as f:
            json.dump(best_params_f1, f, indent=2)
        print(f"F1参数已保存至 {f1_params_save_path}")
    else:
        print("\n加载已有F1参数")
        with open(f1_params_save_path, 'r') as f:
            best_params_f1 = json.load(f)

    # 3. 训练F1模型
    print("\n训练F1优化模型")
    models_f1 = train_models(X_train, y_train, best_params_f1)
    # 保存F1模型
    joblib.dump(models_f1, f1_models_save_path)
    print(f"F1模型已保存至 {f1_models_save_path}")

    # 4. 评估F1模型在验证集和测试集（默认阈值0.5）
    df_results_f1 = evaluate_models(models_f1, X_val, y_val, X_test, y_test)
    df_results_f1.to_csv('optimized_performance_f1.csv', index=False)
    print("\nF1模型默认阈值性能:")
    print(df_results_f1[df_results_f1['数据集']=='Test'])

    # 5. 在验证集上进行阈值优化
    print("\n在验证集上寻找最优阈值")
    thresholds_f1 = {}
    for name, model in models_f1.items():
        th, f1_best = find_best_threshold(model, X_val, y_val)
        thresholds_f1[name] = th
        print(f"{name}: 最优阈值 = {th:.4f}, 对应F1 = {f1_best:.4f}")

    # 6. 用最优阈值评估测试集
    results_threshold_f1 = []
    for name, model in models_f1.items():
        th = thresholds_f1[name]
        results_threshold_f1.append(evaluate_with_threshold(model, X_test, y_test, th, name))
    df_threshold_f1 = pd.DataFrame(results_threshold_f1)
    df_threshold_f1.to_csv('threshold_performance_f1.csv', index=False)
    print("\nF1模型优化阈值后测试集性能:")
    print(df_threshold_f1)

    # 7. 加载AUC模型作为baseline
    if load_auc_models:
        print("\n加载AUC优化模型(baseline) ")
        try:
            models_auc = joblib.load(auc_models_path)
            print("AUC模型加载成功")
        except FileNotFoundError:
            print("AUC模型文件未找到，重新训练")
            with open(auc_params_path, 'r') as f:
                best_params_auc = json.load(f)
            models_auc = train_models(X_train, y_train, best_params_auc)
            print("AUC模型重新训练完成")
        # 评估AUC模型的最优阈值（使用验证集）
        thresholds_auc = {}
        for name, model in models_auc.items():
            th, _ = find_best_threshold(model, X_val, y_val)
            thresholds_auc[name] = th
        results_threshold_auc = []
        for name, model in models_auc.items():
            th = thresholds_auc[name]
            results_threshold_auc.append(evaluate_with_threshold(model, X_test, y_test, th, name))
        df_threshold_auc = pd.DataFrame(results_threshold_auc)

        # 8. 对比表格
        print("\n对比结果(测试集, 阈值优化后)")
        # 合并AUC和F1的优化阈值结果
        df_auc = df_threshold_auc.set_index('模型')
        df_f1 = df_threshold_f1.set_index('模型')
        # 添加基线（AUC调优）和F1调优的差异
        comparison = pd.DataFrame({
            'AUC_调优_精确率': df_auc['精确率'],
            'AUC_调优_召回率': df_auc['召回率'],
            'AUC_调优_F1分数': df_auc['F1分数'],
            'AUC_调优_AUC': df_auc['AUC-ROC'],
            'F1_调优_精确率': df_f1['精确率'],
            'F1_调优_召回率': df_f1['召回率'],
            'F1_调优_F1分数': df_f1['F1分数'],
            'F1_调优_AUC': df_f1['AUC-ROC'],
            'F1_提升_F1分数': df_f1['F1分数'] - df_auc['F1分数'],
            'F1_提升_召回率': df_f1['召回率'] - df_auc['召回率']
        })
        print(comparison)
        comparison.to_csv(results_save_path)
        print(f"对比结果已保存至 {results_save_path}")

        # 9. 可视化对比
        # 绘制F1模型的ROC和PR
        plot_curves(models_f1, X_val, y_val, title_suffix='F1调优_验证集', save_path='roc_pr_f1.png')
        plot_confusion_matrices(models_f1, X_val, y_val, title_suffix='F1调优_验证集', save_path='cm_f1.png')

        # 绘制AUC模型与F1模型的性能对比条形图
        plt.figure(figsize=(10, 6))
        metrics = ['精确率', '召回率', 'F1分数']
        x = np.arange(len(comparison))
        width = 0.35
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i+1)
            auc_vals = comparison[f'AUC_调优_{metric}']
            f1_vals = comparison[f'F1_调优_{metric}']
            plt.bar(x - width/2, auc_vals, width, label='AUC调优')
            plt.bar(x + width/2, f1_vals, width, label='F1调优')
            plt.xticks(x, comparison.index, rotation=45)
            plt.title(metric)
            plt.legend()
        plt.tight_layout()
        plt.savefig('comparison_bars.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("对比图已保存为 comparison_bars.png")

        return comparison
    else:
        print("未加载AUC模型，仅输出F1模型结果")
        return df_threshold_f1