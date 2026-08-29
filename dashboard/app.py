import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'codes'))

import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 页面配置
st.set_page_config(
    page_title="信贷违约预测 - 模型仪表板",
    page_icon="O",
    layout="wide"
)

# 标题
st.title("信贷违约预测模型仪表板")
st.markdown("\n")

# 加载所有数据（缓存）
@st.cache_resource
def load_models():
    """加载已训练的模型，并修复 LightGBM _Booster"""
    import lightgbm as lgb
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    pkl_path = os.path.join(base_path, 'best_models_f1_optimized.pkl')
    zip_path = os.path.join(base_path, 'best_models_f1_optimized.zip')
    
    if not os.path.exists(pkl_path) and os.path.exists(zip_path):
        import zipfile
        with st.spinner("解压模型文件中"):
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(base_path)
        st.success("模型解压完成")
    
    models = joblib.load(pkl_path)
    
    def fix_booster(obj):
        if isinstance(obj, lgb.LGBMClassifier):
            if obj._Booster is not None:
                try:
                    model_str = obj._Booster.model_to_string()
                    obj._Booster = lgb.Booster(model_str=model_str)
                except Exception:
                    obj._Booster = None
        elif isinstance(obj, dict):
            for v in obj.values():
                fix_booster(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                fix_booster(item)
        elif hasattr(obj, '__dict__'):
            for attr_value in obj.__dict__.values():
                fix_booster(attr_value)
        return obj
    
    for name, model in models.items():
        models[name] = fix_booster(model)
    
    return models

@st.cache_data
def load_data():
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    X_val = pd.read_csv(os.path.join(base_path, 'X_val_fe.csv'))
    y_val = pd.read_csv(os.path.join(base_path, 'y_val.csv'), header=None).values.ravel()
    X_test = pd.read_csv(os.path.join(base_path, 'X_test_fe.csv'))
    y_test = pd.read_csv(os.path.join(base_path, 'y_test.csv'), header=None).values.ravel()
    return X_val, y_val, X_test, y_test

@st.cache_data
def load_thresholds():
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    with open(os.path.join(base_path, 'best_thresholds.json'), 'r') as f:
        thresholds = json.load(f)
    return thresholds

@st.cache_data
def load_performance():
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    return pd.read_csv(os.path.join(base_path, 'final_model_performance_comparison.csv'))

@st.cache_data
def load_shap_importance():
    """加载预计算的 SHAP 特征重要性"""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    return pd.read_csv(os.path.join(base_path, 'shap_importance.csv'))

@st.cache_data
def load_shap_dependence():
    """加载预计算的 SHAP 依赖图数据"""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    with open(os.path.join(base_path, 'shap_dependence.json'), 'r') as f:
        data = json.load(f)
    return data

@st.cache_data
def load_shap_features():
    base_path = os.path.join(os.path.dirname(__file__), '..', 'codes')
    with open(os.path.join(base_path, 'shap_features.json'), 'r') as f:
        return json.load(f)

# 主加载
models = load_models()
X_val, y_val, X_test, y_test = load_data()
thresholds = load_thresholds()
perf_df = load_performance()

# 加载 SHAP 数据
try:
    shap_importance = load_shap_importance()
    shap_dependence = load_shap_dependence()
    shap_features = load_shap_features()
    shap_available = True
except FileNotFoundError:
    shap_available = False
    st.warning("SHAP 预计算数据未找到")

# 缓存预测概率
@st.cache_data
def get_all_predictions():
    all_probs = {}
    for name, model in models.items():
        all_probs[f"{name}_val"] = model.predict_proba(X_val)[:, 1]
        all_probs[f"{name}_test"] = model.predict_proba(X_test)[:, 1]
    return all_probs

all_probs = get_all_predictions()

# 缓存曲线数据
@st.cache_data
def get_curve_data(y_true, y_prob, label):
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    
    n = len(fpr)
    if n > 1000:
        idx = np.linspace(0, n-1, 1000, dtype=int)
        fpr = fpr[idx]
        tpr = tpr[idx]
    n_pr = len(recalls)
    if n_pr > 1000:
        idx_pr = np.linspace(0, n_pr-1, 1000, dtype=int)
        recalls = recalls[idx_pr]
        precisions = precisions[idx_pr]
    
    return fpr, tpr, precisions, recalls

# 侧边栏
st.sidebar.header("控制面板")

# 视图切换
view_mode = st.sidebar.radio(
    "选择视图",
    ["模型性能", "SHAP 分析"],
    index=0
)

if view_mode == "模型性能":
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

    use_optimal = st.sidebar.checkbox("使用最优阈值", value=True)
    if not use_optimal:
        custom_threshold = st.sidebar.slider(
            "自定义阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
else:
    # SHAP 分析模式下，只保留模型选择（用于说明）
    st.sidebar.info("SHAP 分析基于 LightGBM 模型（验证集 5000 样本）")

st.sidebar.markdown("\n")

# 主体内容
if view_mode == "模型性能":
    # 原有的模型性能展示
    if dataset_choice == "验证集 (Validation)":
        X, y = X_val, y_val
        suffix = "val"
    else:
        X, y = X_test, y_test
        suffix = "test"

    model = models[selected_model]
    key = f"{selected_model}_{suffix}"
    y_prob = all_probs[key]

    if use_optimal:
        threshold = thresholds.get(selected_model, 0.5)
    else:
        threshold = custom_threshold

    y_pred = (y_prob >= threshold).astype(int)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    st.subheader(f"{selected_model} 模型性能概览")

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

    # ROC/PR 曲线
    st.subheader("ROC曲线 与 PR曲线")
    fpr, tpr, precisions, recalls = get_curve_data(y, y_prob, f"{selected_model}_{suffix}")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC 曲线", "PR 曲线"))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{selected_model} (AUC={auc:.4f})'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='随机猜测', line=dict(dash='dash')), row=1, col=1)
    fig.update_xaxes(title_text="假阳性率", row=1, col=1)
    fig.update_yaxes(title_text="真正例率", row=1, col=1)
    fig.add_trace(go.Scatter(x=recalls, y=precisions, mode='lines', name=selected_model), row=1, col=2)
    fig.update_xaxes(title_text="召回率", row=1, col=2)
    fig.update_yaxes(title_text="精确率", row=1, col=2)
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 特征重要性（树模型）
    st.subheader("树模型特征重要性")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X.columns.tolist()
        imp_df = pd.DataFrame({'特征': feature_names, '重要性': importances}).sort_values('重要性', ascending=True)
        top_n = st.slider("显示 Top N 特征", min_value=5, max_value=20, value=10, key="feature_top_n")
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
        default=["精确率", "召回率", "F1分数"],
        key="metrics_select"
    )
    if not perf_df.empty:
        perf_long = perf_df.melt(id_vars=['模型'], value_vars=metrics, 
                             var_name='指标', value_name='数值')
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

else:
    # SHAP 分析视图
    st.subheader("SHAP 特征重要性分析（LightGBM）")
    st.caption("基于验证集样本计算，展示各特征对违约概率的平均影响强度")

    if not shap_available:
        st.warning("SHAP 数据未加载")
    else:
        # 1. 特征重要性条形图（交互式）
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_importance = px.bar(
                shap_importance,
                x='SHAP_均值',
                y='特征',
                orientation='h',
                title="SHAP 特征重要性（平均绝对值）",
                color='SHAP_均值',
                color_continuous_scale='Viridis',
                text='SHAP_均值'
            )
            fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_importance.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_importance, use_container_width=True)

        with col2:
            st.markdown("""
            - **条形越长** = 该特征对违约概率的影响越大
            - **SHAP 均值** = 该特征平均改变模型输出的幅度
            - 正负号不影响长度，只影响方向
            
            **Top 3 驱动因子：**
            1. **subGrade** — 贷款细分级
            2. **dti_sq** — 负债收入比的平方
            3. **dti** — 负债收入比
            """)

        # 2. 依赖图（选择特征，绘制散点图）
        st.markdown("\n")
        st.subheader("特征依赖图（查看影响方向与拐点）")

        selected_feature = st.selectbox(
            "选择特征查看依赖图",
            shap_features,
            index=0
        )

        if selected_feature in shap_dependence:
            dep_data = pd.DataFrame(shap_dependence[selected_feature])
            
            fig_dep = px.scatter(
                dep_data,
                x='特征值',
                y='SHAP值',
                color='color_by',
                color_continuous_scale='RdBu_r',
                title=f'{selected_feature} 的 SHAP 依赖图',
                labels={
                    '特征值': selected_feature,
                    'SHAP值': 'SHAP 值（影响方向）',
                    'color_by': 'fico_interest'
                },
                opacity=0.7
            )
            # 添加 y=0 参考线
            fig_dep.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_dep.update_layout(height=450)
            st.plotly_chart(fig_dep, use_container_width=True)

            # 显示业务解读
            st.info(f"""
            **{selected_feature} 的 SHAP 解读：**
            - SHAP 值 > 0：该特征值会 **提高** 违约概率
            - SHAP 值 < 0：该特征值会 **降低** 违约概率
            - 颜色越红：fico_interest 组合值越高
            """)

st.markdown("\n")
st.caption(f"数据来源: 信贷数据集 | 模型版本: 5 个 F1 调优模型 | SHAP 基于 LightGBM")