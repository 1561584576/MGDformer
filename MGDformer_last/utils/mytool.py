# # 康汝兵 哈哈哈
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
# # 读取数据
# df = pd.read_csv(r'../dataset/weather/weather.csv')
# df=df.iloc[:1000,1:]
# # 初始化 MinMaxScaler
# scaler = MinMaxScaler()
# # 对每列数据进行归一化
# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# num=[]
# count=0
# df1=pd.DataFrame()
# while df:
#     first_col=df.columns[0]
#     corr_matrix = df.corr()
#     high_corr_cols=corr_matrix[first_col][corr_matrix[first_col].abs()>0.7].index
#     df1[high_corr_cols]=df[high_corr_cols]
#     df=df.drop(high_corr_cols)
#     df=df.drop(first_col)
#     count=count+1
#     num.append(len(high_corr_cols))
#
# print(count)
# print(num)
# print("sssssssssssssssss")


#版本2
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
# # 读取数据
# df = pd.read_csv(r'../dataset/weather/weather.csv')
# # df = pd.read_csv(r'../dataset/exchange_rate/exchange_rate.csv')
# time=df.iloc[:,0]
# df = df.iloc[:, 1:]
#
# # 初始化 MinMaxScaler
# scaler = MinMaxScaler()
# # 对每列数据进行归一化
# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#
# num = []
# count = 0
# df1 = pd.DataFrame()
#
# while not df.empty:  # 检查 DataFrame 是否为空
#     first_col = df.columns[0]
#     corr_matrix = df.corr()
#
#     # 获取与first_col相关性绝对值大于0.7的列
#     high_corr_cols = corr_matrix[first_col][corr_matrix[first_col].abs() > 0.9].index.tolist()
#     # print(high_corr_cols)
#     co=df[high_corr_cols].corr()
#
#     # print(co)
#     # df0=df[high_corr_cols]
#
#     storage=[]
#
#     if len(high_corr_cols)==1:
#         df1[high_corr_cols] = df[high_corr_cols]
#
#         df = df.drop(high_corr_cols,axis=1)
#
#         count = count + 1
#         num.append(1)
#         continue
#
#     for i in range(1,len(high_corr_cols)):
#         for j in range(i) :
#             if abs(co.iloc[i,j]) < 0.9 and high_corr_cols[i] not in storage:
#                 print("delete")
#                 print(co.iloc[i,j])
#                 storage.append(high_corr_cols[i])
#
#     df1[high_corr_cols]=df[high_corr_cols]
#     print("剔除的列")
#     print(storage)
#
#     df1=df1.drop(storage,axis=1)
#     df=df.drop(list(set(high_corr_cols)-set(storage)),axis=1)
#
#     count=count+1
#     num.append(len(list(set(high_corr_cols)-set(storage))))
#
#
#
#
#     # if len(high_corr_cols) == 0:  # 如果没有高相关列，退出循环
#     #
#     #     df = df.drop(high_corr_cols, axis=1)  # 删除高相关列
#     #
#     #     count += 1
#     #     num.append(len(high_corr_cols))
#     #     break
#     #
#     # # 将相关列添加到df1
#     # df1[high_corr_cols] = df[high_corr_cols]
#     # df = df.drop(high_corr_cols, axis=1)  # 删除高相关列
#     #
#     # count += 1
#     # num.append(len(high_corr_cols))
#
# # 输出结果
# print("提取的高相关特征数量:", num)
# print("分组数量:", count)
# print("新 DataFrame 的形状:", df1.shape)
# df1=pd.concat([time, df1], axis=1)
# # df1.to_csv('../dataset/weather/weather_test.csv', index=False)
# # df1.to_csv('../dataset/exchange_rate/exchange_rate_70.csv', index=False)
# df1.to_csv('../dataset/weather/weather_90.csv', index=False)
# # df1.to_csv('../dataset/exchange_rate/exchange_rate_test_90.csv', index=False)
#版本1
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
# # 读取数据
# df = pd.read_csv(r'../dataset/weather/weather.csv')
# # df = pd.read_csv(r'../dataset/exchange_rate/exchange_rate.csv')
# time=df.iloc[:,0]
# df = df.iloc[:, 1:]
#
# # 初始化 MinMaxScaler
# scaler = MinMaxScaler()
# # 对每列数据进行归一化
# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#
# num = []
# count = 0
# df1 = pd.DataFrame()
#
# while not df.empty:  # 检查 DataFrame 是否为空
#     first_col = df.columns[0]
#     corr_matrix = df.corr()
#
#     # 获取与first_col相关性绝对值大于0.7的列
#     high_corr_cols = corr_matrix[first_col][corr_matrix[first_col].abs() > 0.9].index.tolist()
#
#     # 将相关列添加到df1
#     df1[high_corr_cols] = df[high_corr_cols]
#     df = df.drop(high_corr_cols, axis=1)  # 删除高相关列
#
#     count += 1
#     num.append(len(high_corr_cols))
#
# # 输出结果
# print("提取的高相关特征数量:", num)
# print("分组数量:", count)
# print("新 DataFrame 的形状:", df1.shape)
# df1=pd.concat([time, df1], axis=1)
# # df1.to_csv('../dataset/weather/weather_test.csv', index=False)
# # df1.to_csv('../dataset/exchange_rate/exchange_rate_70.csv', index=False)
# # df1.to_csv('../dataset/weather/weather_90.csv', index=False)
# # df1.to_csv('../dataset/exchange_rate/exchange_rate_test_90.csv', index=False)

# 版本三
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据
df = pd.read_csv(r'../dataset/electricity/electricity.csv')
# df = pd.read_csv(r'../dataset/exchange_rate/exchange_rate.csv')
time=df.iloc[:,0]
df = df.iloc[:, 1:]

# 初始化 MinMaxScaler
scaler = MinMaxScaler()
# 对每列数据进行归一化
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

num = []
count = 0
df1 = pd.DataFrame()

while not df.empty:  # 检查 DataFrame 是否为空
    first_col = df.columns[0]
    corr_matrix = df.corr()

    # 获取与first_col相关性绝对值大于0.7的列
    high_corr_cols = corr_matrix[first_col][corr_matrix[first_col].abs() > 0.9].index.tolist()
    # print(df[high_corr_cols].corr())
    cor_col=high_corr_cols


    df_test=df.drop(high_corr_cols,axis=1)

    for i in df_test.columns:
        for j in high_corr_cols:
            if abs(corr_matrix[i][j])>0.9:
                cor_col.append(i)
                continue

    # 将相关列添加到df1
    df1[cor_col] = df[cor_col]
    print(cor_col)
    df = df.drop(cor_col, axis=1)  # 删除高相关列

    count += 1
    num.append(len(cor_col))

# 反归一化
denormalized_data = scaler.inverse_transform(df1)

# 将反归一化后的数据转换为 DataFrame
df1 = pd.DataFrame(denormalized_data, columns=df1.columns)

# 输出结果
print("提取的高相关特征数量:", num)
print("分组数量:", count)
print("新 DataFrame 的形状:", df1.shape)
df1=pd.concat([time, df1], axis=1)
# df1.to_csv('../dataset/weather/weather_test.csv', index=False)
# df1.to_csv('../dataset/exchange_rate/exchange_rate_70.csv', index=False)
# df1.to_csv('../dataset/exchange_rate/exchange_rate_900.csv', index=False)
# df1.to_csv('../dataset/exchange_rate/exchange_rate_test_90.csv', index=False)
df1.to_csv('../dataset/electricity/electricity_900.csv', index=False)
























# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('your_data.csv')  # 替换为你的CSV文件路径
# df1=pd.DataFrame()
# num=[]
# count=0
#
#
# while not df.empty:
#     # 计算相关性矩阵
#     corr_matrix = df.corr()
#
#     # 初始化一个空列表来存储满足条件的列名
#     selected_columns = []
#
#     # 遍历相关性矩阵的上三角（或下三角），排除对角线
#     for i in range(corr_matrix.shape[0]):
#         for j in range(i + 1, corr_matrix.shape[1]):
#             if corr_matrix.iloc[i, j] > 0.9:
#                 # 如果列j尚未被添加到列表中，则添加它
#                 if df.columns[j] not in selected_columns:
#                     selected_columns.append(df.columns[j])
#                     # 通常，我们不需要将i也添加到列表中，除非有特别需要
#                 # 但如果你想要保留所有相关列，也可以添加df.columns[i]
#     df1[selected_columns]=df[selected_columns]
#     df=df.drop(selected_columns,axis=1)
#     num.append(len(selected_columns))
#     count=count+1
#
#     # 打印结果
#     print("Selected columns:", selected_columns)
#
# # 如果需要，可以基于这些列名重新创建DataFrame
# df_filtered = df[selected_columns]
# print(df_filtered.head())
#

