import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 페이지 설정
st.set_page_config(page_title="용접 데이터 대시보드", layout="wide")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 데이터 로드 ====================
@st.cache_data
def load_and_preprocess_data():
    """데이터 로드 및 전처리"""
    df_raw = pd.read_excel('Welding Data Set_01.xlsx', sheet_name='Raw data')
    df_res = pd.read_excel('Welding Data Set_01.xlsx', sheet_name='result')
    df = pd.merge(df_raw, df_res, on=['idx', 'Machine_Name', 'Item No', 'working time'], how='inner')
    columns_to_drop = ['Machine_Name', 'Item No', 'working time', 'Unnamed: 6', 'idx']
    df = df.drop(columns=columns_to_drop)
    df = df.dropna()
    return df

# ==================== 모델 학습 (캐싱) ====================
@st.cache_resource
def train_model(df):
    """모델 학습 - 앱 실행 시 한 번만 수행"""
    X = df[['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']]
    y = df['defect']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X, accuracy, y_test, y_pred

# 데이터 로드
df = load_and_preprocess_data()

# 모델 학습
model, X, model_accuracy, y_test, y_pred = train_model(df)

# ==================== 페이지 상단 ====================
st.title("⚙️ 용접 공정 품질 분석 대시보드")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("총 데이터 행 수", len(df))
with col2:
    st.metric("총 데이터 컬럼 수", len(df.columns))
with col3:
    st.metric("결측치 수", df.isnull().sum().sum())

st.subheader("📊 원본 데이터프레임 상단 5행")
st.dataframe(df.head(5), use_container_width=True)

# ==================== 섹션 1: 상관계수 히트맵 ====================
st.markdown("---")
st.subheader("📈 공정 변수 간의 상관계수 히트맵")

numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
numerical_cols_for_corr = [col for col in numerical_cols if col not in ['defect', 'defect type']]

fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
sns.heatmap(df[numerical_cols_for_corr].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax_heatmap)
ax_heatmap.set_title('공정 변수 간의 상관계수 히트맵', fontsize=16, pad=20)
st.pyplot(fig_heatmap)

# ==================== 섹션 2: 품질 등급별 분포 비교 ====================
st.markdown("---")
st.subheader("🎯 품질 등급(Defect)별 분포 비교")

fig_dist, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(data=df, x='weld force(bar)', hue='defect', kde=True, palette='viridis', alpha=0.6, multiple='stack', ax=axes[0])
axes[0].set_title('품질 등급(defect)별 용접 가압력(weld force) 분포', fontsize=14)
axes[0].set_xlabel('용접 가압력 (bar)', fontsize=12)
axes[0].set_ylabel('빈도', fontsize=12)
axes[0].legend(title='Defect Grade', labels=sorted(df['defect'].unique()))

sns.histplot(data=df, x='weld current(kA)', hue='defect', kde=True, palette='viridis', alpha=0.6, multiple='stack', ax=axes[1])
axes[1].set_title('품질 등급(defect)별 용접 전류(weld current) 분포', fontsize=14)
axes[1].set_xlabel('용접 전류 (kA)', fontsize=12)
axes[1].set_ylabel('빈도', fontsize=12)
axes[1].legend(title='Defect Grade', labels=sorted(df['defect'].unique()))

plt.tight_layout()
st.pyplot(fig_dist)

# ==================== 섹션 3: 머신러닝 모델 학습 및 평가 ====================
st.markdown("---")
st.subheader("🤖 RandomForestClassifier 모델 학습 및 평가")

col_model1, col_model2 = st.columns(2)
with col_model1:
    st.metric("모델 정확도 (Accuracy)", f"{model_accuracy:.4f}")
with col_model2:
    st.info(f"테스트 데이터 샘플 수: {len(y_test)}")

st.write("##### 분류 보고서 (Classification Report)")
classification_rep = classification_report(y_test, y_pred, zero_division=0)
st.code(classification_rep, language='text')

st.subheader("🔍 모델 특성 중요도")
fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax_importance)
ax_importance.set_title('Random Forest 모델의 특성 중요도', fontsize=14)
ax_importance.set_xlabel('중요도', fontsize=12)
st.pyplot(fig_importance)

# ==================== 섹션 4: 실시간 공정 수치 예측 (NEW) ====================
st.markdown("---")
st.subheader("⚡ 실시간 공정 수치 예측")

st.info("💡 공정 변수를 직접 입력하면 품질 등급을 실시간으로 예측합니다.")

# 원본 데이터 통계값 계산
stats = df[['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']].describe()

# 입력 UI 섹션
col_input1, col_input2, col_input3, col_input4 = st.columns(4)

with col_input1:
    weld_force = st.number_input(
        label='용접 가압력 (bar)',
        min_value=float(stats.loc['min', 'weld force(bar)']),
        max_value=float(stats.loc['max', 'weld force(bar)']),
        value=float(stats.loc['mean', 'weld force(bar)']),
        step=0.1
    )

with col_input2:
    weld_current = st.number_input(
        label='용접 전류 (kA)',
        min_value=float(stats.loc['min', 'weld current(kA)']),
        max_value=float(stats.loc['max', 'weld current(kA)']),
        value=float(stats.loc['mean', 'weld current(kA)']),
        step=0.1
    )

with col_input3:
    weld_voltage = st.number_input(
        label='용접 전압 (v)',
        min_value=float(stats.loc['min', 'weld Voltage(v)']),
        max_value=float(stats.loc['max', 'weld Voltage(v)']),
        value=float(stats.loc['mean', 'weld Voltage(v)']),
        step=0.1
    )

with col_input4:
    weld_time = st.number_input(
        label='용접 시간 (ms)',
        min_value=float(stats.loc['min', 'weld time(ms)']),
        max_value=float(stats.loc['max', 'weld time(ms)']),
        value=float(stats.loc['mean', 'weld time(ms)']),
        step=1.0
    )

# ==================== AI 인퍼런스 섹션 ====================
# 입력값을 모델이 인식할 수 있는 데이터프레임으로 변환
input_data = pd.DataFrame({
    'weld force(bar)': [weld_force],
    'weld current(kA)': [weld_current],
    'weld Voltage(v)': [weld_voltage],
    'weld time(ms)': [weld_time]
})

# 모델 예측
predicted_defect = model.predict(input_data)[0]
predicted_defect_proba = model.predict_proba(input_data)[0]

# 예측 결과 표시
st.markdown("---")
st.subheader("🎯 예측 결과")

col_result1, col_result2 = st.columns(2)

with col_result1:
    color_map = {'Good': '🟢', 'Bad': '🔴', 'Normal': '🟡'}
    emoji = color_map.get(predicted_defect, '⚪')
    st.success(f"{emoji} **예측 품질 등급: {predicted_defect}**")

with col_result2:
    max_proba = max(predicted_defect_proba)
    st.info(f"📊 **신뢰도: {max_proba:.2%}**")

# 각 클래스별 확률 시각화
st.write("**각 품질 등급별 예측 확률**")
defect_classes = sorted(model.classes_)
proba_df = pd.DataFrame({
    '품질 등급': defect_classes,
    '예측 확률': predicted_defect_proba
})

fig_proba, ax_proba = plt.subplots(figsize=(10, 5))
bars = ax_proba.bar(proba_df['품질 등급'], proba_df['예측 확률'], color=['#FF6B6B', '#FFE66D', '#4ECDC4'])
ax_proba.set_ylabel('확률', fontsize=12)
ax_proba.set_title('각 품질 등급별 예측 확률', fontsize=14)
ax_proba.set_ylim([0, 1])

for bar in bars:
    height = bar.get_height()
    ax_proba.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1%}', ha='center', va='bottom')

st.pyplot(fig_proba)

st.markdown("---")
st.success("✅ 대시보드 로딩 완료!")
