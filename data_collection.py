import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

fruits = ['apple', 'lime', 'lemon', 'orange', 'pear', 'tomato']
types = ["A0", "A1", "r", "g", "b"]
rgbmax_length = 16  # 最短长度是16
fsrmax_length = 220 # 去0后估计差不多230左右

data = {}

for fruit in fruits:
    data[fruit] = {}
    for tp in types:
        data[fruit][tp] = []
        for i in range(1, 11):  # 10个文件
            file_name = f'{fruit}{i}.txt'
            try:
                with open(f"real_data/{tp}/{file_name}", 'r') as file:
                    content = file.readlines()
                    # 将每行转换为整数，并移除空行
                    num_list = [int(x.strip()) for x in content if x.strip()]
                    # 移除列表前面的连续零值
                    first_non_zero_index = next((index for index, value in enumerate(num_list) if value != 0), None)
                    if first_non_zero_index is not None:
                        cleaned_data = num_list[first_non_zero_index:]
                    else:
                        cleaned_data = num_list  # 如果全部是零，则保留原样
                    # 如果清理后的数据长度超过最大长度，进行截断
                    if tp in ['r', 'g', 'b']: 
                        if len(cleaned_data) > rgbmax_length:
                            cleaned_data = cleaned_data[:rgbmax_length]
                    else:
                        if len(cleaned_data) > fsrmax_length:
                            cleaned_data = cleaned_data[:fsrmax_length]                        
                    # 存储到相应的分类下
                    data[fruit][tp].append(cleaned_data)
            except FileNotFoundError:
                print(f"文件未找到: {file_name}")

pdb.set_trace()

# 进行数据处理
processed_data = {}

for fruit in fruits:
    processed_data[fruit] = {}
    for tp in types:
        # 同步处理rgb和fsr
        time_series_length = 16 if tp in ['r', 'g', 'b'] else 220
        # 初始化新的时间序列列表
        new_time_series = []
        for time_point in range(time_series_length):
            # 收集该时间点在10组数据中的所有值
            values_at_time_point = [data[fruit][tp][trial][time_point] for trial in range(10) if len(data[fruit][tp][trial]) > time_point]
            # 去除最大值和最小值
            if len(values_at_time_point) > 2:  # 确保有足够的数据去除
                values_at_time_point.remove(max(values_at_time_point))
                values_at_time_point.remove(min(values_at_time_point))
                # 计算平均值
                avg_value = sum(values_at_time_point) / len(values_at_time_point)
            else:
                avg_value = None  # 应该不会发生，因为已经确定了16(rgb)和220(A0,A1)，所以每个时间点的每种物体，肯定都是10组数据
            new_time_series.append(avg_value)
        
        # 存储处理后的新时间序列
        processed_data[fruit][tp] = new_time_series

with open("processed_data.pkl", "wb") as f:
    pickle.dump(processed_data, f)


# rgb: 每种物体8个数据作为训练，剩下2个测试，方法是KNN
# fsr: zero-mean，然后思路1：lstm 思路2：PCA降维