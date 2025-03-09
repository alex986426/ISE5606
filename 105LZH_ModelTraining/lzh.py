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

# 定义特征
X = data.drop(['对数总价', '对数均价/平方米每元'], axis=1).values

# 总价目标变量
y_total = data['对数总价'].values
# 均价目标变量
y_avg = data['对数均价/平方米每元'].values

# 特征缩放
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集（总价预测）
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(
    X_scaled, y_total, test_size=0.2, random_state=42)

# 划分训练集和测试集（均价预测）
X_train_avg, X_test_avg, y_train_avg, y_test_avg = train_test_split(
    X_scaled, y_avg, test_size=0.2, random_state=42)

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

# 人工神经网络模型（总价预测）
ann_total = build_ann_model(X_train_total.shape[1])
history_total = ann_total.fit(X_train_total, y_train_total, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
y_pred_ann_total = ann_total.predict(X_test_total).flatten()
mae_ann_total = mean_absolute_error(y_test_total, y_pred_ann_total)
mse_ann_total = mean_squared_error(y_test_total, y_pred_ann_total)
r2_ann_total = r2_score(y_test_total, y_pred_ann_total)
print("人工神经网络模型（总价预测）:")
print(f"MAE: {mae_ann_total:.4f}, MSE: {mse_ann_total:.4f}, R²: {r2_ann_total:.4f}")

# 人工神经网络模型（均价预测）
ann_avg = build_ann_model(X_train_avg.shape[1])
history_avg = ann_avg.fit(X_train_avg, y_train_avg, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
y_pred_ann_avg = ann_avg.predict(X_test_avg).flatten()
mae_ann_avg = mean_absolute_error(y_test_avg, y_pred_ann_avg)
mse_ann_avg = mean_squared_error(y_test_avg, y_pred_ann_avg)
r2_ann_avg = r2_score(y_test_avg, y_pred_ann_avg)
print("人工神经网络模型（均价预测）:")
print(f"MAE: {mae_ann_avg:.4f}, MSE: {mse_ann_avg:.4f}, R²: {r2_ann_avg:.4f}")

# 随机森林模型（总价预测）
rf_total = RandomForestRegressor(n_estimators=100, random_state=42)
mae_rf_total, mse_rf_total, r2_rf_total = evaluate_model(rf_total, X_train_total, y_train_total, X_test_total, y_test_total)
print("随机森林模型（总价预测）:")
print(f"MAE: {mae_rf_total:.4f}, MSE: {mse_rf_total:.4f}, R²: {r2_rf_total:.4f}")

# 随机森林模型（均价预测）
rf_avg = RandomForestRegressor(n_estimators=100, random_state=42)
mae_rf_avg, mse_rf_avg, r2_rf_avg = evaluate_model(rf_avg, X_train_avg, y_train_avg, X_test_avg, y_test_avg)
print("随机森林模型（均价预测）:")
print(f"MAE: {mae_rf_avg:.4f}, MSE: {mse_rf_avg:.4f}, R²: {r2_rf_avg:.4f}")

# 决策树模型（总价预测）
dt_total = DecisionTreeRegressor(random_state=42)
mae_dt_total, mse_dt_total, r2_dt_total = evaluate_model(dt_total, X_train_total, y_train_total, X_test_total, y_test_total)
print("决策树模型（总价预测）:")
print(f"MAE: {mae_dt_total:.4f}, MSE: {mse_dt_total:.4f}, R²: {r2_dt_total:.4f}")

# 决策树模型（均价预测）
dt_avg = DecisionTreeRegressor(random_state=42)
mae_dt_avg, mse_dt_avg, r2_dt_avg = evaluate_model(dt_avg, X_train_avg, y_train_avg, X_test_avg, y_test_avg)
print("决策树模型（均价预测）:")
print(f"MAE: {mae_dt_avg:.4f}, MSE: {mse_dt_avg:.4f}, R²: {r2_dt_avg:.4f}")

# 梯度提升树模型（总价预测）
gb_total = GradientBoostingRegressor(n_estimators=100, random_state=42)
mae_gb_total, mse_gb_total, r2_gb_total = evaluate_model(gb_total, X_train_total, y_train_total, X_test_total, y_test_total)
print("梯度提升树模型（总价预测）:")
print(f"MAE: {mae_gb_total:.4f}, MSE: {mse_gb_total:.4f}, R²: {r2_gb_total:.4f}")

# 梯度提升树模型（均价预测）
gb_avg = GradientBoostingRegressor(n_estimators=100, random_state=42)
mae_gb_avg, mse_gb_avg, r2_gb_avg = evaluate_model(gb_avg, X_train_avg, y_train_avg, X_test_avg, y_test_avg)
print("梯度提升树模型（均价预测）:")
print(f"MAE: {mae_gb_avg:.4f}, MSE: {mse_gb_avg:.4f}, R²: {r2_gb_avg:.4f}")
