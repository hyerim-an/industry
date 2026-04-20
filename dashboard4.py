import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==================== 페이지 설정 ====================
st.set_page_config(page_title="용접 데이터 대시보드", layout="wide")

# ==================== 한글 폰트 설정 (Streamlit Cloud 리눅스 환경 대응) ====================
# 리눅스 서버의 나눔고딕 폰트 절대 경로
NANUM_GOTHIC_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

# 로컬 환경(Windows) 또는 리눅스 환경 구분
if os.path.exists(NANUM_GOTHIC_PATH):
    # Streamlit Cloud 리눅스 환경: 나눔고딕 폰트 사용
    fm.fontManager.addfont(NANUM_GOTHIC_PATH)
    plt.rcParams['font.family'] = 'NanumGothic'
else:
    # 로컬 환경(Windows): 맑은 고딕 사용
    plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스(-) 기호 깨짐 방지
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

# 데이터 로드 및 모델 학습
df = load_and_preprocess_data()
model, X, model_accuracy, y_test, y_pred = train_model(df)

# ==================== 페이지 상단 ====================
st.title("⚙️ 용접 공정 품질 분석 대시보드")
st.markdown("---")

# 데이터 통계값 미리 계산 (좌측 입력 UI용)
stats = df[['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']].describe()

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

sns.histplot(data=df, x='weld current(kA)', hue='defect', kde=True, palette='viridis', alpha=0.6, multiple='stack', ax=axes[1])
axes[1].set_title('품질 등급(defect)별 용접 전류(weld current) 분포', fontsize=14)
axes[1].set_xlabel('용접 전류 (kA)', fontsize=12)
axes[1].set_ylabel('빈도', fontsize=12)

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

# ==================== 섹션 4: 실시간 공정 수치 예측 ====================
st.markdown("---")
st.subheader("⚡ 실시간 공정 수치 예측")

st.info("💡 공정 변수를 직접 입력하면 품질 등급을 실시간으로 예측합니다. (학습 범위 외 입력도 가능)")

# 좌측 사이드바: 사용자 입력 (무제한 range 지원)
st.sidebar.title("⚙️ 공정 수치 입력")
st.sidebar.markdown("---")

with st.sidebar:
    st.write("**공정 변수 설정** (학습 범위 외 입력 가능)")
    
    # 용접 가압력 입력: st.number_input 사용 (무제한 범위)
    weld_force = st.number_input(
        label='🔧 용접 가압력 (bar)',
        min_value=0.0,
        value=float(stats.loc['mean', 'weld force(bar)']),
        step=0.1,
        help=f"학습 범위: {float(stats.loc['min', 'weld force(bar)']):.2f} ~ {float(stats.loc['max', 'weld force(bar)']):.2f} bar"
    )
    
    # 용접 전류 입력: st.number_input 사용 (무제한 범위)
    weld_current = st.number_input(
        label='⚡ 용접 전류 (kA)',
        min_value=0.0,
        value=float(stats.loc['mean', 'weld current(kA)']),
        step=0.1,
        help=f"학습 범위: {float(stats.loc['min', 'weld current(kA)']):.2f} ~ {float(stats.loc['max', 'weld current(kA)']):.2f} kA"
    )
    
    # 용접 전압 입력: st.number_input 사용 (무제한 범위)
    weld_voltage = st.number_input(
        label='⚡ 용접 전압 (v)',
        min_value=0.0,
        value=float(stats.loc['mean', 'weld Voltage(v)']),
        step=0.1,
        help=f"학습 범위: {float(stats.loc['min', 'weld Voltage(v)']):.2f} ~ {float(stats.loc['max', 'weld Voltage(v)']):.2f} v"
    )
    
    # 용접 시간 입력: st.number_input 사용 (무제한 범위)
    weld_time = st.number_input(
        label='⏱️ 용접 시간 (ms)',
        min_value=0.0,
        value=float(stats.loc['mean', 'weld time(ms)']),
        step=1.0,
        help=f"학습 범위: {float(stats.loc['min', 'weld time(ms)']):.2f} ~ {float(stats.loc['max', 'weld time(ms)']):.2f} ms"
    )

# ==================== AI 인퍼런스: 모델 예측 수행 ====================
# 사용자 입력값을 모델 입력 데이터프레임으로 변환
input_data = pd.DataFrame({
    'weld force(bar)': [weld_force],
    'weld current(kA)': [weld_current],
    'weld Voltage(v)': [weld_voltage],
    'weld time(ms)': [weld_time]
})

# 모델 추론 수행
predicted_defect = model.predict(input_data)[0]
predicted_defect_proba = model.predict_proba(input_data)[0]
max_proba = max(predicted_defect_proba)

# ==================== 예측 결과 및 상태 표시 ====================
st.markdown("---")
st.subheader("🎯 예측 결과 및 상태")

# 좌측: 예측 결과, 우측: 특성 중요도 시각화
col_result_left, col_result_right = st.columns([1, 1])

# 왼쪽: 예측 결과 + 품질 알림
with col_result_left:
    st.write("#### 📊 예측 결과")
    
    # 품질 등급에 따른 시각적 경고 (정의된 등급에 따라 아이콘과 메시지 선택)
    if predicted_defect == 3:
        # 불량(Bad): 사이렌 아이콘 + 에러 메시지
        st.error(f"🚨 **불량(Bad)** - 품질 등급: {predicted_defect}")
        st.error("⚠️ **설비 점검 필수** - 공정 매개변수를 즉시 검토하고 설비를 점검하세요.")
    elif predicted_defect in [1, 2]:
        # 정상: 체크 아이콘 + 성공 메시지
        st.success(f"✅ **정상** - 품질 등급: {predicted_defect}")
        st.success("현재 공정 설정을 유지하세요.")
    
    # 예측 신뢰도 표시
    st.metric("예측 신뢰도", f"{max_proba:.2%}")

# 오른쪽: 특성 중요도 (변수 중요도 기반 근거 제시)
with col_result_right:
    st.write("#### 🔍 모델 특성 중요도")
    
    # 특성 중요도 데이터프레임 생성
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # 특성 중요도 바 차트: hue='Feature' 설정으로 palette 경고 해결
    fig_importance, ax_importance = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=feature_importance,
        x='Importance',
        y='Feature',
        hue='Feature',  # hue 설정으로 "Passing palette without assigning hue" 경고 방지
        palette='viridis',
        ax=ax_importance,
        legend=False  # 범례 비활성화
    )
    ax_importance.set_title('예측 근거: 특성 중요도', fontsize=12)
    ax_importance.set_xlabel('중요도', fontsize=10)
    st.pyplot(fig_importance)

# ==================== 각 클래스별 예측 확률 시각화 ====================
st.markdown("---")
st.write("#### 각 품질 등급별 예측 확률")

defect_classes = sorted(model.classes_)
proba_df = pd.DataFrame({
    '품질 등급': defect_classes,
    '예측 확률': predicted_defect_proba
})

fig_proba, ax_proba = plt.subplots(figsize=(10, 5))
bars = ax_proba.bar(proba_df['품질 등급'], proba_df['예측 확률'], color=['#FF6B6B', '#FFE66D', '#4ECDC4'])
ax_proba.set_ylabel('확률', fontsize=12)
ax_proba.set_xlabel('품질 등급 (Defect)', fontsize=12)
ax_proba.set_title('각 품질 등급별 예측 확률', fontsize=14)
ax_proba.set_ylim([0, 1])

# 각 바에 확률값을 백분율로 표시
for bar in bars:
    height = bar.get_height()
    ax_proba.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1%}', ha='center', va='bottom')

st.pyplot(fig_proba)

# ==================== 맞춤형 공정 인사이트 (3가지 핵심) ====================
st.markdown("---")
with st.expander("📋 맞춤형 공정 인사이트 (핵심 3가지)", expanded=True):
    
    # 인사이트 1: 용접 가압력(Force) 안정성 평가
    force_mean = float(stats.loc['mean', 'weld force(bar)'])
    force_std = float(stats.loc['std', 'weld force(bar)'])
    force_deviation = ((weld_force - force_mean) / force_std) if force_std > 0 else 0
    
    st.write("**1️⃣ 용접 가압력(Force) 안정성**")
    if abs(force_deviation) < 1:
        st.write(f"   ✅ 현재 **{weld_force:.2f} bar**는 최적 범위 내")
        st.write(f"   📊 참고 (평균 {force_mean:.2f}±{force_std:.2f} bar) | 편차: {force_deviation:.2f}σ")
        st.write(f"   💡 권고: 현재 설정 유지")
    elif force_deviation > 1:
        st.write(f"   ⚠️ 현재 **{weld_force:.2f} bar**는 평균보다 높음")
        st.write(f"   📊 참고 (평균 {force_mean:.2f}±{force_std:.2f} bar) | 편차: {force_deviation:.2f}σ")
        st.write(f"   💡 권고: {force_mean:.2f}±{force_std:.2f} bar 범위로 감소 조정")
    else:
        st.write(f"   ⚠️ 현재 **{weld_force:.2f} bar**는 평균보다 낮음")
        st.write(f"   📊 참고 (평균 {force_mean:.2f}±{force_std:.2f} bar) | 편차: {force_deviation:.2f}σ")
        st.write(f"   💡 권고: {force_mean:.2f}±{force_std:.2f} bar 범위로 증가 조정")
    
    # 인사이트 2: 용접 전류(Current) 안정성 평가
    current_mean = float(stats.loc['mean', 'weld current(kA)'])
    current_std = float(stats.loc['std', 'weld current(kA)'])
    current_deviation = ((weld_current - current_mean) / current_std) if current_std > 0 else 0
    
    st.write("**2️⃣ 용접 전류(Current) 안정성**")
    if abs(current_deviation) < 1:
        st.write(f"   ✅ 현재 **{weld_current:.2f} kA**는 최적 범위 내")
        st.write(f"   📊 참고 (평균 {current_mean:.2f}±{current_std:.2f} kA) | 편차: {current_deviation:.2f}σ")
        st.write(f"   💡 권고: 현재 설정 유지")
    elif current_deviation > 1:
        st.write(f"   ⚠️ 현재 **{weld_current:.2f} kA**는 평균보다 높음")
        st.write(f"   📊 참고 (평균 {current_mean:.2f}±{current_std:.2f} kA) | 편차: {current_deviation:.2f}σ")
        st.write(f"   💡 권고: {current_mean:.2f}±{current_std:.2f} kA 범위로 감소 조정")
    else:
        st.write(f"   ⚠️ 현재 **{weld_current:.2f} kA**는 평균보다 낮음")
        st.write(f"   📊 참고 (평균 {current_mean:.2f}±{current_std:.2f} kA) | 편차: {current_deviation:.2f}σ")
        st.write(f"   💡 권고: {current_mean:.2f}±{current_std:.2f} kA 범위로 증가 조정")
    
    # 인사이트 3: 모델 신뢰도 및 다음 액션 추천
    max_proba_pct = max_proba * 100
    st.write("**3️⃣ 모델 신뢰도 및 실행 액션**")
    
    if max_proba_pct >= 90:
        st.write(f"   ✅ 신뢰도: **{max_proba_pct:.1f}%** (높음 - 매우 신뢰성 있음)")
        st.write(f"   💡 액션: 예측을 신뢰하고 해당 공정 파라미터로 운영 진행")
        st.write(f"   📌 추가 조치: 특별한 모니터링 불필요")
    elif max_proba_pct >= 70:
        st.write(f"   ⚠️ 신뢰도: **{max_proba_pct:.1f}%** (중간 - 추가 검증 권장)")
        st.write(f"   💡 액션: 추가 모니터링 필수, 실시간 센서 데이터와 함께 검증")
        st.write(f"   📌 추가 조치: 5~10개 샘플 추가 수집 후 재검증")
    else:
        st.write(f"   ❌ 신뢰도: **{max_proba_pct:.1f}%** (낮음 - 신뢰성 낮음)")
        st.write(f"   💡 액션: 설비 점검 및 공정 파라미터 튜닝 필수")
        st.write(f"   📌 추가 조치: 엔지니어 확인 후 운영 재개 필요")

st.markdown("---")
st.success("✅ 대시보드 로딩 완료!")
