import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
from sklearn.neighbors import KNeighborsClassifier

with open('processed_data.pkl', 'rb') as handle:
    data = pickle.load(handle)

fruits = ['apple', 'lime', 'lemon', 'orange', 'pear', 'tomato']
for item in fruits:
    data[f'{item}'].pop("r")
    data[f'{item}'].pop("g")
    data[f'{item}'].pop("b")
fsr = data

# zero-mean

for index,item in enumerate(fruits):
    if index == 0:
        a0 = fsr[f"{item}"]['A0']
        a1 = fsr[f"{item}"]['A1']
    else:
        a0 = [x + y for x, y in zip(fsr[f"{item}"]['A0'],a0)]
        a1 = [x + y for x, y in zip(fsr[f"{item}"]['A1'],a1)]
a0 = [x/6 for x in a0]
a1 = [x/6 for x in a1]

for index,item in enumerate(fruits):
    # fsr[f"{item}"]['A0'] , fsr[f"{item}"]['A1'] 
    # a0, a1
    fsr[f"{item}"]['A0']  = [x - y for x, y in zip(fsr[f"{item}"]['A0'],a0)]
    fsr[f"{item}"]['A1']  = [x - y for x, y in zip(fsr[f"{item}"]['A1'],a1)]

processed_data = fsr

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



# lstm

# 先主成分，再去用LSTM
# 必须要记住每一个fsr类别的平均值，例如apple在A0上的平均值
