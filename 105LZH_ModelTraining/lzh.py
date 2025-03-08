import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from wordcloud import WordCloud
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanSquaredError
import re

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'SimHei'

# 导入数据文件 数据预览
filename = 'C:/Users/12521/PycharmProjects/pythonProject1/final_data.csv'
data = pd.read_csv(filename, encoding='utf-8')
data.info()
print('___________')
print(data.sample(5))
print('___________1')
# 修改列名
data = data[['标题', '地区', '具体位置', '类型', '均价/平方米每元', '室厅数', '面积', '标签', '总价']]
# 数据清洗
print(data.isnull().sum())
print('___________')
summary = {
    '标题': [data['标题'].nunique()],
    '类型': list(data['类型'].unique()),
    '具体位置': [data['具体位置'].nunique()],
    '地区': list(data['地区'].unique()),
    '室厅数': list(data['室厅数'].unique()),
}
df = pd.DataFrame.from_dict(summary, orient='index').T
df = df.fillna('')
print(df)
print('___________2')

# 缺失值用0填充
data['均价/平方米每元'] = data['均价/平方米每元'].replace('价格待定', 0)

# 中位数填充
data['均价/平方米每元'] = data['均价/平方米每元'].astype(float)
medians = data['均价/平方米每元'].median()
data['均价/平方米每元'] = data['均价/平方米每元'].replace(0, medians)

# 众数填充
data['室厅数'].value_counts()
data['室厅数'].fillna('3室,4室', inplace=True)

# 提取小区标签并统计出现次数
tags = data['标签'].str.split(',').explode()
tag_counts = Counter(tags)
print(tag_counts)
print('___________')

# 查找可用的中文字体文件路径
try:
    font_path = findfont(FontProperties(family='SimHei'))
except:
    # 如果找不到 SimHei 字体，尝试其他字体
    font_path = findfont(FontProperties())

# 生成词云图
wordcloud = WordCloud(font_path=font_path, background_color='white', width=800, height=400).generate_from_frequencies(tag_counts)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('小区标签词云图')
plt.show()

# 提取总价列中的数值
def extract_price(row):
    match = re.findall(r'\d+', row)
    if match:
        prices = [int(num) for num in match]
        return np.mean(prices)
    else:
        return np.nan

data['总价数值'] = data['总价'].apply(extract_price)
data = data.dropna(subset=['总价数值'])

# 通过以上的描述分析可知【均价，总价，面积】呈现偏态分布
# 进行对数变换
data['对数均价/平方米每元'] = np.log(data['均价/平方米每元'])
data['对数总价'] = np.log(data['总价数值'])
# 面积列也需要先转换为数值类型
data['面积数值'] = data['面积'].str.extract(r'(\d+\.?\d*)').astype(float)
data['对数面积'] = np.log(data['面积数值'])

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.hist(data['对数均价/平方米每元'])
plt.title('均价分布')

plt.subplot(2, 2, 2)
plt.hist(data['室厅数'])
plt.title('室厅数分布')

plt.subplot(2, 2, 3)
plt.hist(data['对数面积'])
plt.title('面积分布')

print(data.sample())
print('___________')

# 对室厅数进行独热编码
room_hall_encoded = pd.get_dummies(data['室厅数'], prefix='室厅数')
data = pd.concat([data, room_hall_encoded], axis=1)
data.drop('室厅数', axis=1, inplace=True)

# 定义特征和目标变量
X = data[['对数均价/平方米每元', '对数面积'] + list(room_hall_encoded.columns)]
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
model_rf.fit(X_train, Y_total_train)
y_total_pred_rf = model_rf.predict(X_test)
mae_total_rf = mean_absolute_error(Y_total_test, y_total_pred_rf)
mse_total_rf = mean_squared_error(Y_total_test, y_total_pred_rf)
r2_total_rf = r2_score(Y_total_test, y_total_pred_rf)

model_rf.fit(X_train, Y_avg_train)
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