import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanSquaredError
import numpy as np

# 加载数据
# 请将 'your_data.csv' 替换为你实际的数据文件名
data = pd.read_csv('../002Data/preprocessing data.csv')

# 找出所有的分类特征列
categorical_columns = data.select_dtypes(include=['object']).columns

# 对所有分类特征进行独热编码
for col in categorical_columns:
    encoded_col = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data.drop(col, axis=1), encoded_col], axis=1)

# 2. 数据预处理
X = data.drop('对数总价', axis=1).values
y = data['对数总价'].values

# 特征缩放
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建人工神经网络
def build_ann_model(input_shape):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1)  # 输出层
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
    return model

# 定义模型评估函数
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

# 人工神经网络模型
ann_model = build_ann_model(X_train.shape[1])
history = ann_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
y_pred_ann = ann_model.predict(X_test).flatten()
mae_ann = mean_absolute_error(y_test, y_pred_ann)
mse_ann = mean_squared_error(y_test, y_pred_ann)
r2_ann = r2_score(y_test, y_pred_ann)
print("人工神经网络模型:")
print(f"MAE: {mae_ann:.4f}, MSE: {mse_ann:.4f}, R²: {r2_ann:.4f}")

# 随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
mae_rf, mse_rf, r2_rf = evaluate_model(rf_model, X_train, y_train, X_test, y_test)
print("随机森林模型:")
print(f"MAE: {mae_rf:.4f}, MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}")

# 决策树模型
dt_model = DecisionTreeRegressor(random_state=42)
mae_dt, mse_dt, r2_dt = evaluate_model(dt_model, X_train, y_train, X_test, y_test)
print("决策树模型:")
print(f"MAE: {mae_dt:.4f}, MSE: {mse_dt:.4f}, R²: {r2_dt:.4f}")

# 梯度提升树模型
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
mae_gb, mse_gb, r2_gb = evaluate_model(gb_model, X_train, y_train, X_test, y_test)
print("梯度提升树模型:")
print(f"MAE: {mae_gb:.4f}, MSE: {mse_gb:.4f}, R²: {r2_gb:.4f}")
