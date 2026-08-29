import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'codes'))

import model_sec_optimization

import zipfile
import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 页面配置
st.set_page_config(
    page_title="信贷违约预测 - 模型仪表板",
    page_icon="O",
    layout="wide"
)

# 标题
st.title("信贷违约预测模型仪表板")
st.markdown("\n")

# 加载模型和数据
@st.cache_resource
def load_models():
    """加载已训练的模型，并修复 LightGBM _Booster"""
    import lightgbm as lgb
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    pkl_path = os.path.join(base_path, 'best_models_f1_optimized.pkl')
    zip_path = os.path.join(base_path, 'best_models_f1_optimized.zip')
    data_zip_path = os.path.join(base_path, 'data.zip')
    
    # 解压模型
    if not os.path.exists(pkl_path) and os.path.exists(zip_path):
        with st.spinner("解压模型文件中"):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_path)
        st.success("模型解压完成")
    
    # 解压数据（包含 best_thresholds.json 等）
    data_extracted = False
    if os.path.exists(data_zip_path):
        # 检查关键文件是否存在，如果不存在则解压
        if not os.path.exists(os.path.join(base_path, 'best_thresholds.json')):
            with st.spinner("解压数据文件中"):
                with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(base_path)
            data_extracted = True
            st.success("数据解压完成")
    
    models = joblib.load(pkl_path)
    
    def fix_lightgbm_booster(obj):
        """递归修复 LightGBM 模型：用 model_to_string 重建 _Booster"""
        if isinstance(obj, lgb.LGBMClassifier):
            if obj._Booster is not None:
                try:
                    # 提取模型字符串，重建 Booster
                    model_str = obj._Booster.model_to_string()
                    obj._Booster = lgb.Booster(model_str=model_str)
                except Exception as e:
                    obj._Booster = None
        elif isinstance(obj, dict):
            for v in obj.values():
                fix_lightgbm_booster(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                fix_lightgbm_booster(item)
        elif hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                fix_lightgbm_booster(attr_value)
        return obj
    
    for name, model in models.items():
        models[name] = fix_lightgbm_booster(model)
    
    return models

@st.cache_data
def load_data():
    """加载验证集和测试集数据"""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    X_val = pd.read_csv(os.path.join(base_path, 'X_val_fe.csv'))
    y_val = pd.read_csv(os.path.join(base_path, 'y_val.csv'), header=None).values.ravel()
    X_test = pd.read_csv(os.path.join(base_path, 'X_test_fe.csv'))
    y_test = pd.read_csv(os.path.join(base_path, 'y_test.csv'), header=None).values.ravel()
    return X_val, y_val, X_test, y_test

@st.cache_data
def load_thresholds():
    """加载最优阈值"""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    with open(os.path.join(base_path, 'best_thresholds.json'), 'r') as f:
        thresholds = json.load(f)
    return thresholds

@st.cache_data
def load_performance():
    """加载模型性能对比"""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    return pd.read_csv(os.path.join(base_path, 'final_model_performance_comparison.csv'))

# 加载数据
models = load_models()
X_val, y_val, X_test, y_test = load_data()
thresholds = load_thresholds()
perf_df = load_performance()

@st.cache_data
def get_all_predictions(X_val, X_test):
    """
    预先计算所有模型在验证集和测试集上的预测概率，
    结果以字典形式缓存，键为 "模型名_val" 或 "模型名_test"
    """
    all_probs = {}
    for name, model in models.items():
        all_probs[f"{name}_val"] = model.predict_proba(X_val)[:, 1]
        all_probs[f"{name}_test"] = model.predict_proba(X_test)[:, 1]
    return all_probs

all_probs = get_all_predictions(X_val, X_test)

# 侧边栏：模型选择
st.sidebar.header("控制面板")

model_names = list(models.keys())
selected_model = st.sidebar.selectbox(
    "选择模型",
    model_names,
    index=model_names.index('Stacking')
)

dataset_choice = st.sidebar.radio(
    "选择数据集",
    ["验证集 (Validation)", "测试集 (Test)"]
)

# 阈值调整
use_optimal = st.sidebar.checkbox("使用最优阈值", value=True)
if not use_optimal:
    custom_threshold = st.sidebar.slider(
        "自定义阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

st.sidebar.markdown("\n")

# 主区域：KPI 卡片
st.subheader(f"{selected_model} 模型性能概览")

# 选择数据集
if dataset_choice == "验证集 (Validation)":
    X, y = X_val, y_val
    suffix = "val"
else:
    X, y = X_test, y_test
    suffix = "test"

# 获取预测
key = f"{selected_model}_{suffix}"
y_prob = all_probs[key]

# 应用阈值
if use_optimal:
    threshold = thresholds.get(selected_model, 0.5)
else:
    threshold = custom_threshold

y_pred = (y_prob >= threshold).astype(int)

# 计算指标
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)

# 显示 KPI
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("准确率", f"{acc:.4f}")
col2.metric("精确率", f"{prec:.4f}")
col3.metric("召回率", f"{rec:.4f}")
col4.metric("F1 分数", f"{f1:.4f}")
col5.metric("AUC-ROC", f"{auc:.4f}")

st.markdown(f"当前阈值: {threshold:.4f}")

# 混淆矩阵
st.subheader("混淆矩阵")

cm = confusion_matrix(y, y_pred)
fig_cm = px.imshow(
    cm,
    text_auto=True,
    labels=dict(x="预测值", y="真实值", color="数量"),
    x=["非违约 (0)", "违约 (1)"],
    y=["非违约 (0)", "违约 (1)"],
    color_continuous_scale="Blues",
    aspect="auto"
)
fig_cm.update_layout(height=400)
st.plotly_chart(fig_cm, use_container_width=True)

# ROC曲线 和 PR曲线
st.subheader("ROC曲线 与 PR曲线")

from sklearn.metrics import roc_curve, precision_recall_curve

# ROC
fpr, tpr, _ = roc_curve(y, y_prob)
# PR
precisions, recalls, _ = precision_recall_curve(y, y_prob)

# 双图布局
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("ROC 曲线", "PR 曲线")
)

# ROC
fig.add_trace(
    go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{selected_model} (AUC={auc:.4f})'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=[0,1], y=[0,1], mode='lines', name='随机猜测', line=dict(dash='dash')),
    row=1, col=1
)
fig.update_xaxes(title_text="假阳性率", row=1, col=1)
fig.update_yaxes(title_text="真正例率", row=1, col=1)

# PR
fig.add_trace(
    go.Scatter(x=recalls, y=precisions, mode='lines', name=selected_model),
    row=1, col=2
)
fig.update_xaxes(title_text="召回率", row=1, col=2)
fig.update_yaxes(title_text="精确率", row=1, col=2)

fig.update_layout(height=400, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# 特征重要性
st.subheader("树模型特征重要性")

# 检查是否有 feature_importances_ 属性
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_names = X.columns.tolist()
    
    # 创建 DataFrame 并排序
    imp_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importances
    }).sort_values('重要性', ascending=True)
    
    # 取 Top 10
    top_n = st.slider("显示 Top N 特征", min_value=5, max_value=20, value=10)
    imp_df_top = imp_df.tail(top_n)
    
    fig_imp = px.bar(
        imp_df_top,
        x='重要性',
        y='特征',
        orientation='h',
        title=f'{selected_model} Top {top_n} 特征重要性',
        color='重要性',
        color_continuous_scale='Viridis'
    )
    fig_imp.update_layout(height=400)
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info(f"！！ {selected_model} 不支持特征重要性分析")

# 模型对比
st.subheader("所有模型性能对比")

metrics = st.multiselect(
    "选择显示的指标",
    ["精确率", "召回率", "F1分数", "AUC-ROC"],
    default=["精确率", "召回率", "F1分数"]
)

# 过滤数据
if not perf_df.empty:
    # 根据实际 DataFrame 结构调整
    # 将数据从宽格式转为长格式
    perf_long = perf_df.melt(id_vars=['模型'], value_vars=metrics, 
                         var_name='指标', value_name='数值')
    # 去除缺失值
    perf_long = perf_long.dropna()
    complete_models = perf_long.groupby('模型')['数值'].count()
    complete_models = complete_models[complete_models == len(metrics)].index
    perf_long = perf_long[perf_long['模型'].isin(complete_models)]
    
    fig_compare = px.bar(
        perf_long,
        x='模型',
        y='数值',
        color='指标',
        barmode='group',
        title="模型性能对比"
        )
    st.plotly_chart(fig_compare, use_container_width=True)
else:
    st.info("暂无对比数据，检查performance文件")

# 预测概率分布
st.subheader("预测概率分布")

fig_dist = px.histogram(
    x=y_prob,
    color=y.astype(str),
    nbins=50,
    labels={'x': '预测概率', 'color': '真实标签'},
    title=f'{selected_model} 预测概率分布（{dataset_choice}）',
    color_discrete_map={'0': 'lightskyblue', '1': 'lightcoral'}
)
fig_dist.update_layout(height=400)
st.plotly_chart(fig_dist, use_container_width=True)


st.markdown("\n")
st.caption(f"数据来源: 信贷数据集 | 模型版本: {selected_model} | 阈值: {threshold:.4f}")