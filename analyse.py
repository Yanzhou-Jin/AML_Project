import matplotlib.pyplot as plt
import numpy as np

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


# 进行数据处理
processed_data = {}

for fruit in fruits:
    processed_data[fruit] = {}
    for tp in types:
        # 确定时间序列的长度
        time_series_length = 16 if tp in ['r', 'g', 'b'] else 220
        # 初始化新的时间序列列表
        new_time_series = []
        for time_point in range(time_series_length):
            # 收集该时间点在10组数据中的所有值
            values_at_time_point = [data[fruit][tp][group][time_point] for group in range(10) if len(data[fruit][tp][group]) > time_point]
            # 去除最大值和最小值
            if len(values_at_time_point) > 2:  # 确保有足够的数据去除
                values_at_time_point.remove(max(values_at_time_point))
                values_at_time_point.remove(min(values_at_time_point))
                # 计算平均值
                avg_value = sum(values_at_time_point) / len(values_at_time_point)
            else:
                avg_value = None  # 根据需要处理不足三个数据的情况
            new_time_series.append(avg_value)
        
        # 存储处理后的新时间序列
        processed_data[fruit][tp] = new_time_series


# 假设time_series_length是根据之前处理后的数据确定的时间序列的长度
time_series_length = 220
time_points = range(time_series_length)

# 绘制A0数据的图表
plt.figure(figsize=(12, 6))
for fruit in fruits:
    if 'A0' in processed_data[fruit]:  # 确保数据存在
        plt.plot(time_points, processed_data[fruit]['A0'], label=fruit, marker='o')
plt.title('A0 Time Series for Different Fruits')
plt.xlabel('Time Point')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 绘制A1数据的图表
plt.figure(figsize=(12, 6))
for fruit in fruits:
    if 'A1' in processed_data[fruit]:  # 确保数据存在
        plt.plot(time_points, processed_data[fruit]['A1'], label=fruit, marker='o')
plt.title('A1 Time Series for Different Fruits')
plt.xlabel('Time Point')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


###############################
# train
###############################

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# 假设processed_data已正确处理并填充数据
# fruits和types变量继续使用之前定义

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 为不同的水果指定不同的颜色
fruit_colors = {
    'apple': 'red',
    'lime': 'green',
    'lemon': 'yellow',
    'orange': 'orange',
    'pear': 'lightgreen',
    'tomato': 'tomato',
}

for fruit, color in fruit_colors.items():
    # 假设processed_data[fruit]['r'], ['g'], ['b']是平均化后的数据列表
    xs = processed_data[fruit]['r']  # R值
    ys = processed_data[fruit]['g']  # G值
    zs = processed_data[fruit]['b']  # B值
    ax.scatter(xs, ys, zs, c=color, marker='o', label=fruit)

ax.set_xlabel('R Value')
ax.set_ylabel('G Value')
ax.set_zlabel('B Value')
plt.title('3D Scatter Plot of RGB Features for Different Fruits')
plt.legend()

plt.show()




