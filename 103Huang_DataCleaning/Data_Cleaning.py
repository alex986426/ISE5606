import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'SimHei'

# 导入数据 查看基本信息
filename = '/Users/alexlee/文件/ISE5606/ISE5606/103Huang_DataCleaning/Original_Data.csv'
data = pd.read_csv(filename, encoding='utf-8')
data.info()
print('初始数据规模：', data.shape)
ds1 = data.shape[0]
ds2 = data.shape[1]

# 列名重命名
data.rename(columns={'resblock-tag1 resblock-tag_resblock-type_resblock-tag2_resblock-tag3': '标签'}, inplace=True)

# 重复项识别删除
data = data.drop_duplicates(subset=['标题', '地区', '位置', '具体位置', '类型', '均价/平方米每元', '室厅数', '面积', '标签', '总价'])
print('重复项：已删除行数', ds1-data.shape[0], '清洗后数据规模', data.shape)
ds1 = data.shape[0]

# 选择子集 删除无用属性（具体位置）
data = data[['标题', '地区', '位置', '类型', '均价/平方米每元', '室厅数', '面积', '标签', '总价']]
print('删除无用属性：已删除列数', ds2-data.shape[1], '清洗后数据规模', data.shape)
ds2 = data.shape[1]


# 两端空格 识别处理
def check_space(s):
    if isinstance(s, str):
        has_space = s.startswith(' ') or s.endswith(' ')
        return has_space
    return False


rows_to_delete = []
columns_to_skip = ['标签', '室厅数', '面积']
for col in data.columns:
    if col in columns_to_skip:
        continue
    for index, value in data[col].items():
        has_space = check_space(value)
        if has_space:
            print(f"列名: {col}, 行索引: {index}, 值: {value}, 是否有空格: {has_space}")
            if index not in rows_to_delete:
                rows_to_delete.append(index)
data = data.drop(rows_to_delete)
print('两端空格：已删除行数', ds1-data.shape[0], '清洗后数据规模', data.shape)
ds1 = data.shape[0]

# 特殊字符 识别处理
def check_specialchar(s):
    if isinstance(s, str):
        pattern = r'[@;?!@#$%^&*【】]'
        has_special_char = bool(re.search(pattern, s))
        return has_special_char
    return False


rows_to_delete = []
for col in data.columns:
    for index, value in data[col].items():
        has_special_char = check_specialchar(value)
        if has_special_char:
            print(f"列名: {col}, 行索引: {index}, 值: {value}, 是否有特殊字符: {has_special_char}")
            if index not in rows_to_delete:
                rows_to_delete.append(index)
data = data.drop(rows_to_delete)
print('特殊字符：已删除行数', ds1-data.shape[0], '清洗后数据规模', data.shape)
ds1 = data.shape[0]

# 字段拆分 挑选标签
label_series = data['标签'].str.split(',')
label_counts = Counter(tag.strip() for tags in label_series for tag in tags)
label_counts = pd.Series(label_counts).sort_values(ascending=False)
print(label_counts)

data['标签列表'] = data['标签'].str.split(',')
data['标签列表'] = data['标签列表'].apply(lambda tags: [tag.strip() for tag in tags])

频繁标签 = label_counts[label_counts > 100].index.tolist()
print("频繁标签:", 频繁标签)
for 标签 in 频繁标签:
    data[f'是否是{标签}'] = data['标签列表'].apply(lambda tags: 1 if 标签 in tags else 0)
data = data.drop(columns=['标签列表', '标签'])
print('标签拆分：已增加列数', data.shape[1]-ds2, '预处理后数据规模', data.shape)
ds2 = data.shape[1]

# 删除没有因变量（总价）的数据
data = data.dropna(subset=['总价'])
print('因变量缺失：已删除行数', ds1-data.shape[0], '清洗后数据规模', data.shape)
ds1 = data.shape[0]

# 异常值（每平方均价与常识不符）
sns.boxplot(x=data['均价/平方米每元'])
plt.title('均价/平方米每元的箱型图')
plt.xlabel('均价/平方米每元')
plt.show()
data['均价/平方米每元'] = pd.to_numeric(data['均价/平方米每元'], errors='coerce')
data = data.dropna(subset=['均价/平方米每元'])
q1 = data['均价/平方米每元'].quantile(0.25)
q3 = data['均价/平方米每元'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outlier_index = data[(data['均价/平方米每元'] < lower_bound) | (data['均价/平方米每元'] > upper_bound)].index
data = data.drop(outlier_index)
print('均价异常值：已删除行数', ds1-data.shape[0], '清洗后数据规模', data.shape)
ds1 = data.shape[0]

# 缺失值识别
data.info()

# 位置 不填充直接删除
data = data.dropna(subset=['位置'])
print('位置空缺值：已删除行数', ds1-data.shape[0], '清洗后数据规模', data.shape)
ds1 = data.shape[0]

# 室厅数 众数填充
print('室厅数空缺值：', data['室厅数'].isnull().sum())
mode_value = data['室厅数'].mode()[0]
data['室厅数'].fillna(mode_value, inplace=True)
missing_count = data['室厅数'].isnull().sum()
print('室厅数空缺值（填充后）：', data['室厅数'].isnull().sum())

# 面积替换（中位数） 空缺值填充（平均数）
print('面积空缺值：', data['面积'].isnull().sum())
data['面积'] = data['面积'].astype(str)


def process_area(area):
    area = area.replace('建面', '').strip()
    area = area.replace('㎡', '').strip()
    if "-" in area:
        area_min, area_max = map(float, area.split('-'))
        return (area_min + area_max) / 2
    return float(area)


data['面积'] = data['面积'].apply(process_area)
area_average = int(data['面积'].median())
data['面积'].fillna(area_average, inplace=True)
print('面积空缺值（填充后）：', data['面积'].isnull().sum(), '填充数字', area_average)

# 总价替换 中位数
data['总价'] = data['总价'].astype(str)


def process_total(total):
    total = total.replace('总价', '').strip()
    total = total.replace("(万/套)", '').strip()
    if '-' in total:
        total_min, total_max = map(float, total.split('-'))
        return (total_min + total_max) / 2
    return float(total)


data['总价'] = data['总价'].apply(process_total)

# 检查清洗后数据
data.info()
print(data.sample(5))

# ALex:将清洗后的数据保存为CSV文件
output_filename = '/Users/alexlee/文件/ISE5606/ISE5606/002Data/Cleaned_Data.csv'
data.to_csv(output_filename, index=False, encoding='utf-8')
print(f"清洗后的数据已保存到 {output_filename}")


