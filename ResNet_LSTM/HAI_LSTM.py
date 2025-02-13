import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# CSV 파일 경로 (실제 파일 위치)
file_path = r"C:\Users\조성찬\OneDrive - UOS\바탕 화면\hai-23.05\hai-test1.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 데이터 확인
print(df.head())
print(df.columns)  # 컬럼 이름 확인

# 타겟 컬럼 선택 (예: P1_FCV01D)
data = df['P1_FCV01D'].values

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# 시퀀스 데이터 생성 함수
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 20  # 입력 시퀀스 길이
X, y = create_sequences(data_scaled, seq_length)

# 학습 데이터와 테스트 데이터 분리
split_ratio = 0.8
split_idx = int(len(X) * split_ratio)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# LSTM 입력 형태로 변경 (batch_size, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 모델 생성
model = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, activation='tanh'),
    Dense(1)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 모델 학습
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# 예측 수행
y_pred = model.predict(X_test)

# 원래 스케일로 변환
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_rescaled = scaler.inverse_transform(y_pred)

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.plot(y_test_rescaled, label="Actual", linestyle="dashed")
plt.plot(y_pred_rescaled, label="Predicted")
plt.legend()
plt.title("LSTM 시계열 예측")
plt.show()
