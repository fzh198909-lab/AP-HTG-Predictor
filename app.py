import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 网页全局设置 (打造高级学术感)
# ==========================================
st.set_page_config(page_title="AP-HTG Subphenotype Predictor", layout="wide", initial_sidebar_state="expanded")

# 移除默认的顶部空白，使布局更紧凑
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🖥️ AP-HTG Dynamic Trajectory Predictor (Multi-Class)")
st.markdown("#### Machine Learning-Derived Calculator for 4 Distinct Clinical Subphenotypes")
st.markdown("<span style='color:gray'>**A real-time, explainable clinical decision support tool based on admission-day indicators.**</span>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 2. 侧边栏：临床指标输入区
# ==========================================
st.sidebar.markdown("### 📝 Patient Data Input (Day 1)")
st.sidebar.caption("Please enter the routine laboratory parameters:")
st.sidebar.markdown("---")

wbc = st.sidebar.number_input("WBC (×10^9/L)", value=11.22, step=0.1)
ca = st.sidebar.number_input("Serum Calcium (mmol/L)", value=2.02, step=0.01)
lymph = st.sidebar.number_input("Lymphocyte count", value=15.66, step=0.1)
alb = st.sidebar.number_input("Albumin (g/L)", value=32.53, step=0.1)
plt_count = st.sidebar.number_input("Platelet (×10^9/L)", value=260.26, step=1.0)
hct = st.sidebar.number_input("Hematocrit (%)", value=32.24, step=0.1)

# ==========================================
# 3. 核心算法：提取的 4分类模型参数
# ==========================================
classes = ['C1', 'C2', 'C3', 'C4']
intercepts = {'C1': 0.98343, 'C2': 1.53403, 'C3': -1.90701, 'C4': -0.61045}

means_sds = {
    'WBC': {'mean': 11.21977, 'sd': 6.04654},
    'Serum Calcium': {'mean': 2.02116, 'sd': 0.32009},
    'Lymphocyte': {'mean': 15.66425, 'sd': 9.38016},
    'Albumin': {'mean': 32.52759, 'sd': 6.96843},
    'Platelet': {'mean': 260.25505, 'sd': 137.14878},
    'Hematocrit': {'mean': 32.23712, 'sd': 5.42257}
}

coefs = {
    'WBC': {'C1': 0.61190, 'C2': -2.51316, 'C3': 1.98007, 'C4': -0.07881},
    'Serum Calcium': {'C1': 0.99034, 'C2': 0.84533, 'C3': 0.16057, 'C4': -1.99625},
    'Lymphocyte': {'C1': 0.08573, 'C2': 0.36105, 'C3': -0.32577, 'C4': -0.12101},
    'Albumin': {'C1': 0.17778, 'C2': 0.56119, 'C3': -0.79003, 'C4': 0.05107},
    'Platelet': {'C1': 0.23163, 'C2': -0.54278, 'C3': 0.39587, 'C4': -0.08472},
    'Hematocrit': {'C1': 0.23191, 'C2': 0.27175, 'C3': -0.15064, 'C4': -0.35302}
}

input_vals = {
    'WBC': wbc, 'Serum Calcium': ca, 'Lymphocyte': lymph, 
    'Albumin': alb, 'Platelet': plt_count, 'Hematocrit': hct
}

# 亚型图例颜色字典 (Nature/Lancet 配色)
color_map = {'C1': '#4DBBD5', 'C2': '#00A087', 'C3': '#F39B7F', 'C4': '#E64B35'}

# ==========================================
# 严格按照文章同步的亚型全称
# ==========================================
class_names_full = [
    'C1 (Moderate-Inflammatory)', 
    'C2 (Mild-Steady)', 
    'C3 (Severe-Hyperinflammatory)', 
    'C4 (Lipotoxic-Hypocalcemic)'
]

class_name_dict = dict(zip(classes, class_names_full))

# ==========================================
# 4. 预测与结果展示逻辑 (多分类 Softmax 版)
# ==========================================
st.sidebar.markdown("---")
if st.sidebar.button("🚀 Predict Subphenotype", use_container_width=True, type="primary"):
    
    # 步骤 1: 数据标准化 & 计算各个类别的 Z-score
    z_scores = {c: intercepts[c] for c in classes}
    feature_contributions_per_class = {c: {} for c in classes}
    
    for feature in means_sds.keys():
        val = input_vals[feature]
        mean = means_sds[feature]['mean']
        sd = means_sds[feature]['sd']
        std_val = (val - mean) / sd
        
        for c in classes:
            contribution = std_val * coefs[feature][c]
            feature_contributions_per_class[c][feature] = contribution
            z_scores[c] += contribution

    # 步骤 2: Softmax 函数转换为概率分布
    exp_z = {c: np.exp(z_scores[c]) for c in classes}
    sum_exp_z = sum(exp_z.values())
    probabilities = {c: exp_z[c] / sum_exp_z for c in classes}
    
    # 获取最高概率的预测结果
    predicted_class = max(probabilities, key=probabilities.get)
    max_prob = probabilities[predicted_class]

    # ================= UI 展现区 =================
    col1, col_space, col2 = st.columns([1.2, 0.1, 1.5])
    
    with col1:
        st.markdown("### 📊 Probability Distribution")
        
        # 绘制四分类概率水平条形图 (纯白背景 + 全名)
        prob_values = [probabilities[c] * 100 for c in classes]
        colors = [color_map[c] for c in classes]
        
        fig_probs = go.Figure(go.Bar(
            x=prob_values,
            y=class_names_full,  # 使用具象化的全名
            orientation='h',
            marker_color=colors,
            marker_line_color='black',
            marker_line_width=1,
            text=[f"{val:.1f}%" for val in prob_values],
            textposition='auto',
            textfont=dict(color='white', size=14, family="Arial", weight="bold")
        ))
        
        fig_probs.update_layout(
            xaxis_title="Probability (%)",
            yaxis_title="",
            xaxis=dict(range=[0, 100], showgrid=True, gridcolor='#E5E5E5'),
            plot_bgcolor='rgba(0,0,0,0)', # 纯透明/纯白背景
            margin=dict(l=10, r=20, t=20, b=40),
            height=260,
            font=dict(family="Arial")
        )
        st.plotly_chart(fig_probs, use_container_width=True)
        
        # 动态结论警报卡片 (极简干净版：只显示亚型和概率)
        st.markdown(f"#### 🎯 Dominant Trajectory: <span style='color:{color_map[predicted_class]}'>{predicted_class}</span>", unsafe_allow_html=True)
        
        if predicted_class == 'C4':
            st.error(f"### {class_name_dict['C4']}\n\n**Probability: {max_prob*100:.1f}%**")
        elif predicted_class == 'C3':
            st.warning(f"### {class_name_dict['C3']}\n\n**Probability: {max_prob*100:.1f}%**")
        elif predicted_class == 'C2':
            st.success(f"### {class_name_dict['C2']}\n\n**Probability: {max_prob*100:.1f}%**")
        elif predicted_class == 'C1':
            st.info(f"### {class_name_dict['C1']}\n\n**Probability: {max_prob*100:.1f}%**")

    with col2:
        st.markdown(f"### 🧬 Feature Impact on {predicted_class}")
        st.caption(f"How individual parameters drove the model towards the **{predicted_class}** classification.")
        
        # 提取主导类别的特征贡献
        contributions = feature_contributions_per_class[predicted_class]
        features = list(contributions.keys())
        impacts = list(contributions.values())
        
        sorted_indices = np.argsort(np.abs(impacts))
        sorted_features = [features[i] for i in sorted_indices]
        sorted_impacts = [impacts[i] for i in sorted_indices]
        
        # 颜色逻辑：推动主导类别（红色系），拉扯离主导类别（蓝色系）
        bar_colors = ['#E64B35' if val > 0 else '#4DBBD5' for val in sorted_impacts]
        
        # 绘制高级条形图 (无裁剪，防遮挡)
        fig_bar = go.Figure(go.Bar(
            x=sorted_impacts,
            y=sorted_features,
            orientation='h',
            marker_color=bar_colors,
            marker_line_color='black',
            marker_line_width=1,
            text=[f"{val:+.3f}" for val in sorted_impacts],
            textposition='outside',
            textfont=dict(family="Arial", size=12),
            cliponaxis=False  # 防止文字被裁剪
        ))
        
        # 动态 X 轴留白防遮挡
        max_val, min_val = max(sorted_impacts), min(sorted_impacts)
        padding = (max_val - min_val) * 0.3 if (max_val - min_val) != 0 else 1
        
        fig_bar.update_layout(
            xaxis_title=f"Impact on Log-odds (towards {predicted_class})",
            yaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)', # 纯透明/纯白背景
            margin=dict(l=10, r=60, t=20, b=40), # 加大右侧留白
            height=360,
            font=dict(family="Arial"),
            xaxis=dict(showgrid=True, gridcolor='#E5E5E5', zeroline=True, zerolinecolor='black', zerolinewidth=1.5, range=[min_val - padding, max_val + padding])
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        st.info(f"💡 **Interpretation:** Red bars pushed the model to classify the patient as **{predicted_class}**, while blue bars pushed the decision away from it.")

else:
    st.info("👈 Please adjust the clinical parameters in the sidebar and click **'Predict Subphenotype'** to generate the analysis.")

# ==========================================
# 5. 底部：免责声明
# ==========================================
st.markdown("---")
with st.expander("📖 Disclaimer", expanded=False):
    st.markdown("""
    *This machine learning-derived web tool is intended strictly for academic research and adjunctive clinical decision support. It does not replace professional medical judgment. Always correlate with dynamic clinical assessments.*
    """)