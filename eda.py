import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib to use Malgun Gothic (Korean font for Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 'Welding Data Set_01.xlsx' 파일에서 'Raw data' 시트를 불러와 df_raw에 저장
df_raw = pd.read_excel('Welding Data Set_01.xlsx', sheet_name='Raw data')

# 'Welding Data Set_01.xlsx' 파일에서 'result' 시트를 불러와 df_res에 저장
df_res = pd.read_excel('Welding Data Set_01.xlsx', sheet_name='result')

# df_raw와 df_res를 'idx' 컬럼 기준으로 병합하여 df 생성
df = pd.merge(df_raw, df_res, on=['idx', 'Machine_Name', 'Item No', 'working time'], how='inner')

# 분석에 불필요한 컬럼 제거
columns_to_drop = ['Machine_Name', 'Item No', 'working time', 'Unnamed: 6', 'idx']
df = df.drop(columns=columns_to_drop)

# 결측치가 있는 행 모두 제거하여 학습용 데이터셋 완성
df = df.dropna()

# 1. 상관계수 히트맵 생성
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
numerical_cols_for_corr = [col for col in numerical_cols if col not in ['defect', 'defect type']]

plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols_for_corr].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('공정 변수 간의 상관계수 히트맵', fontsize=16)
plt.show()

# 2. 품질 등급(defect)별 분포 비교
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data=df, x='weld force(bar)', hue='defect', kde=True, palette='viridis', alpha=0.6, multiple='stack')
plt.title('품질 등급(defect)별 용접 가압력(weld force) 분포', fontsize=14)
plt.xlabel('용접 가압력 (bar)', fontsize=12)
plt.ylabel('빈도', fontsize=12)
plt.legend(title='Defect Grade', labels=df['defect'].unique())

plt.subplot(1, 2, 2)
sns.histplot(data=df, x='weld current(kA)', hue='defect', kde=True, palette='viridis', alpha=0.6, multiple='stack')
plt.title('품질 등급(defect)별 용접 전류(weld current) 분포', fontsize=14)
plt.xlabel('용접 전류 (kA)', fontsize=12)
plt.ylabel('빈도', fontsize=12)
plt.legend(title='Defect Grade', labels=df['defect'].unique())

plt.tight_layout()
plt.show()

# 모델 학습 및 평가
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = df[['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']]
y = df['defect']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("RandomForestClassifier 모델 학습 완료.\n")
print(f"테스트 데이터에 대한 예측 결과 (일부):\n{y_pred[:5]}\n")
print(f"모델 정확도: {accuracy:.4f}\n")
print("분류 보고서:\n")
print(classification_report(y_test, y_pred, zero_division=0))
