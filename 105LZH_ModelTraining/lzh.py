import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanSquaredError

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 加载数据
data = pd.read_csv('../002Data/preprocessing data.csv')  # 替换为你的数据文件路径

# 定义特征和目标变量
# 这里根据你实际的列名进行修改
X = data[['对数面积', '是否是人车分流', '是否是车位充足', '是否是品牌房企', '是否是绿化率高',
          '是否是低总价', '是否是国央企', '是否是佛山楼盘', '是否是低单价']]
Y_total_price = data['对数总价']
Y_avg_price = data['对数均价/平方米每元']

# 划分训练集和测试集
X_train, X_test, Y_total_train, Y_total_test, Y_avg_train, Y_avg_test = train_test_split(
    X, Y_total_price, Y_avg_price, test_size=0.3, random_state=42)

# 特征缩放（仅用于人工神经网络）
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 人工神经网络模型
model_ann = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model_ann.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])
history = model_ann.fit(X_train_scaled, Y_total_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# 总价预测
y_total_pred_ann = model_ann.predict(X_test_scaled).flatten()
mae_total_ann = mean_absolute_error(Y_total_test, y_total_pred_ann)
mse_total_ann = mean_squared_error(Y_total_test, y_total_pred_ann)
r2_total_ann = r2_score(Y_total_test, y_total_pred_ann)

# 均价预测
y_avg_pred_ann = model_ann.predict(scaler.transform(X_test)).flatten()
mae_avg_ann = mean_absolute_error(Y_avg_test, y_avg_pred_ann)
mse_avg_ann = mean_squared_error(Y_avg_test, y_avg_pred_ann)
r2_avg_ann = r2_score(Y_avg_test, y_avg_pred_ann)

print('人工神经网络模型（总价预测）:')
print('mae:', mae_total_ann)
print('mse:', mse_total_ann)
print('r2:', r2_total_ann)
print('人工神经网络模型（均价预测）:')
print('mae:', mae_avg_ann)
print('mse:', mse_avg_ann)
print('r2:', r2_avg_ann)

# 随机森林模型
model_rf = RandomForestRegressor(n_estimators=50, random_state=42)

# 总价预测
model_rf.fit(X_train, Y_total_train)
y_total_pred_rf = model_rf.predict(X_test)
mae_total_rf = mean_absolute_error(Y_total_test, y_total_pred_rf)
mse_total_rf = mean_squared_error(Y_total_test, y_total_pred_rf)
r2_total_rf = r2_score(Y_total_test, y_total_pred_rf)

# 均价预测
model_rf.fit(X_train, Y_avg_train)  # 这里使用 Y_avg_train 而不是 Y_avg_test
y_avg_pred_rf = model_rf.predict(X_test)
mae_avg_rf = mean_absolute_error(Y_avg_test, y_avg_pred_rf)
mse_avg_rf = mean_squared_error(Y_avg_test, y_avg_pred_rf)
r2_avg_rf = r2_score(Y_avg_test, y_avg_pred_rf)

print('随机森林模型（总价预测）:')
print('mae:', mae_total_rf)
print('mse:', mse_total_rf)
print('r2:', r2_total_rf)
print('随机森林模型（均价预测）:')
print('mae:', mae_avg_rf)
print('mse:', mse_avg_rf)
print('r2:', r2_avg_rf)

# 决策树模型
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, Y_total_train)
y_total_pred_dt = model_dt.predict(X_test)
mae_total_dt = mean_absolute_error(Y_total_test, y_total_pred_dt)
mse_total_dt = mean_squared_error(Y_total_test, y_total_pred_dt)
r2_total_dt = r2_score(Y_total_test, y_total_pred_dt)

model_dt.fit(X_train, Y_avg_train)
y_avg_pred_dt = model_dt.predict(X_test)
mae_avg_dt = mean_absolute_error(Y_avg_test, y_avg_pred_dt)
mse_avg_dt = mean_squared_error(Y_avg_test, y_avg_pred_dt)
r2_avg_dt = r2_score(Y_avg_test, y_avg_pred_dt)

print('决策树模型（总价预测）:')
print('mae:', mae_total_dt)
print('mse:', mse_total_dt)
print('r2:', r2_total_dt)
print('决策树模型（均价预测）:')
print('mae:', mae_avg_dt)
print('mse:', mse_avg_dt)
print('r2:', r2_avg_dt)


# 梯度提升树模型
model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_gb.fit(X_train, Y_total_train)
y_total_pred_gb = model_gb.predict(X_test)
mae_total_gb = mean_absolute_error(Y_total_test, y_total_pred_gb)
mse_total_gb = mean_squared_error(Y_total_test, y_total_pred_gb)
r2_total_gb = r2_score(Y_total_test, y_total_pred_gb)

model_gb.fit(X_train, Y_avg_train)
y_avg_pred_gb = model_gb.predict(X_test)
mae_avg_gb = mean_absolute_error(Y_avg_test, y_avg_pred_gb)
mse_avg_gb = mean_squared_error(Y_avg_test, y_avg_pred_gb)
r2_avg_gb = r2_score(Y_avg_test, y_avg_pred_gb)

print('梯度提升树模型（总价预测）:')
print('mae:', mae_total_gb)
print('mse:', mse_total_gb)
print('r2:', r2_total_gb)
print('梯度提升树模型（均价预测）:')
print('mae:', mae_avg_gb)
print('mse:', mse_avg_gb)
print('r2:', r2_avg_gb)

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model_rf, X, Y_total_price, cv=5, scoring='r2')
print("交叉验证平均 $R^2$ 值:", cv_scores.mean())
