import  pandas as pd
import  numpy as np

df = pd.read_csv('new1.csv')
df.head

# 定义新的列名列表，确保新列名数量和原列名数量一致
new_column_names = ['标题', '类型', '地区','位置','具体位置', '室厅数1','室厅数2', '室厅数3', '面积','标签1','标签2','标签3','标签4','均价','平方米每元/单位','总价',]  # 这里根据实际列数修改

# 重命名列
df.columns = new_column_names

# 打印修改列名后的 DataFrame （可选，用于查看结果）
print(df)
#%%
# 定义新的列顺序
new_column_order = ['标题','地区','位置','具体位置', '标签1','标签2','标签3','标签4','均价','平方米每元/单位','总价','类型','室厅数1','室厅数2', '室厅数3','面积',]

# 重新排列列
df = df[new_column_order]

# 打印重新排列列后的 DataFrame（可选，用于查看结果）
print(df)
#%%
# 假设要合并的列名为 'col1', 'col2', 'col3'
columns_to_merge = ['标签1','标签2','标签3','标签4',]

# 将指定列的内容合并到新的一列，并用逗号隔开
df.loc[:, '标签'] = df[columns_to_merge].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1)

# 假设要合并的列名为 'col1', 'col2', 'col3'
columns_to_merge = ['室厅数1','室厅数2', '室厅数3',]

# 链式操作完成列合并
df = df.assign(室厅数=df[columns_to_merge].apply(lambda row: ', '.join(row.dropna().astype(str)), axis=1))

# 打印处理后的数据（可选，用于查看结果）
print(df)
#%%
# 定义新文件的路径
new_file_path = 'new2.csv'

# 将修改后的数据保存为新的 CSV 文件
df.to_csv(new_file_path, index=False)

print(f"修改后的数据已保存到 {new_file_path}")
#%%
column_to_delete = '标签1'
if column_to_delete in df.columns:  # 这里有意外的缩进
    df = df.drop(column_to_delete, axis=1)
    print(f"成功删除列 {column_to_delete}")
else:
    print(f"列 {column_to_delete} 不存在于文件中。")
#%%
column_to_delete = '标签2'
if column_to_delete in df.columns:  # 这里有意外的缩进
    df = df.drop(column_to_delete, axis=1)
    print(f"成功删除列 {column_to_delete}")
else:
    print(f"列 {column_to_delete} 不存在于文件中。")
#%%
column_to_delete = '标签3'
if column_to_delete in df.columns:  # 这里有意外的缩进
    df = df.drop(column_to_delete, axis=1)
    print(f"成功删除列 {column_to_delete}")
else:
    print(f"列 {column_to_delete} 不存在于文件中。")
#%%
column_to_delete = '标签4'
if column_to_delete in df.columns:  # 这里有意外的缩进
    df = df.drop(column_to_delete, axis=1)
    print(f"成功删除列 {column_to_delete}")
else:
    print(f"列 {column_to_delete} 不存在于文件中。")
#%%
column_to_delete = '室厅数1'
if column_to_delete in df.columns:  # 这里有意外的缩进
    df = df.drop(column_to_delete, axis=1)
    print(f"成功删除列 {column_to_delete}")
else:
    print(f"列 {column_to_delete} 不存在于文件中。")
#%%
column_to_delete = '室厅数2'
if column_to_delete in df.columns:  # 这里有意外的缩进
    df = df.drop(column_to_delete, axis=1)
    print(f"成功删除列 {column_to_delete}")
else:
    print(f"列 {column_to_delete} 不存在于文件中。")
#%%
column_to_delete = '室厅数3'
if column_to_delete in df.columns:  # 这里有意外的缩进
    df = df.drop(column_to_delete, axis=1)
    print(f"成功删除列 {column_to_delete}")
else:
    print(f"列 {column_to_delete} 不存在于文件中。")
#%%
df.head
# 定义新的列名列表，确保新列名数量和原列名数量一致
new_column_names = ['标题', '类型', '地区', '位置', '具体位置', '室厅数', '面积', '标签', '均价', '平方米每元/单位', '总价', ]  # 这里根据实际列数修改

# 重命名列
df.columns = new_column_names

# 打印修改列名后的 DataFrame （可选，用于查看结果）
print(df)
#%%
# 定义新文件的路径
new_file_path = 'new3.csv'

# 将修改后的数据保存为新的 CSV 文件
df.to_csv(new_file_path, index=False)

print(f"修改后的数据已保存到 {new_file_path}")
df = pd.read_csv('new3.csv')
df.head

column_to_delete = '平方米每元/单位'
if column_to_delete in df.columns:  # 这里有意外的缩进
    df = df.drop(column_to_delete, axis=1)
    print(f"成功删除列 {column_to_delete}")
else:
    print(f"列 {column_to_delete} 不存在于文件中。")

print(df)
#%%
# 定义新的列顺序
new_column_order = ['标题','地区','位置','具体位置', '标签','均价','总价','类型','室厅数','面积',]

# 重新排列列
df = df[new_column_order]

# 打印重新排列列后的 DataFrame（可选，用于查看结果）
print(df)
#%%
# 定义新文件的路径
new_file_path = 'new5.csv'

# 将修改后的数据保存为新的 CSV 文件
df.to_csv(new_file_path, index=False)

print(f"修改后的数据已保存到 {new_file_path}")
df = pd.read_csv('../../fangjia/new5.csv')
df.head
#%%
# 要更改的旧列名
old_column_name = '标签'
# 新的列名
new_column_name = 'resblock-tag1 resblock-tag_resblock-type_resblock-tag2_resblock-tag3'
#%%
# 要更改的旧列名
old_column_name = '均价'
# 新的列名
new_column_name = '均价/平方米每元'
#%%
# 要更改的旧列名
old_column_name = '均价'
# 新的列名
new_column_name = '均价/平方米每元'
#%%
file_path = '../../fangjia/new5.csv'  # 检查旧列名是否存在

if old_column_name in df.columns:
    # 使用 rename 方法更改列名
    df = df.rename(columns={old_column_name: new_column_name})
    # 将修改后的 DataFrame 保存回原文件
    df.to_csv(file_path, index=False)
    print(f"成功将列 {old_column_name} 更改为 {new_column_name} 并保存到文件。")
else:
    print(f"列 {old_column_name} 不存在于文件中，无法进行更改。")
#%%
# 定义新文件的路径
new_file_path = '../../fangjia/original data.csv'

# 将修改后的数据保存为新的 CSV 文件
df.to_csv(new_file_path, index=False)

print(f"修改后的数据已保存到 {new_file_path}")
