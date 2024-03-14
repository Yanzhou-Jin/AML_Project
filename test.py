from joblib import dump, load
import serial
import numpy as np
import time
import pdb
import copy

# rf = load('knn_final.joblib')
rf = load('rf_final.joblib')

# fruits = ["garlic", "lemon", "lime", "orange", "tomato", "whitefoam", "yellowsponge"]

ser = serial.Serial(r'COM4', 9600, timeout=1)
time.sleep(1)

def read():
    all_data = []
    save_data = []

    time_interval = 0

    start_time = time.time()

    while(time_interval < 20): # 单组数据默认采集5秒钟
        single_data = ser.readline().decode('utf-8').rstrip()
        if single_data != '':
            all_data.append(single_data)
        print(single_data)
        end_time = time.time()
        time_interval = end_time - start_time

    all_data = np.array(all_data)

    processed_data = [item.split('=') for item in all_data if '=' in item]

    # save_data = np.empty((0, 5), dtype=np.float64)
    save_data_a0 = []
    save_data_a1 = []
    save_data_r = []
    save_data_g = []
    save_data_b = []
    
    
    for item in processed_data:
        if item[0] == "A0":
            save_data_a0.append(item[1])
    
        elif item[0] == "A1":
            save_data_a1.append(item[1])
    
        elif item[0] == "red":
            save_data_r.append(item[1])

        elif item[0] == "green":
            save_data_g.append(item[1])

        elif item[0] == "blue":
            save_data_b.append(item[1])

    save_data_a0 = np.array(save_data_a0)
    save_data_a1 = np.array(save_data_a1)
    save_data_r = np.array(save_data_r)
    save_data_g = np.array(save_data_g)
    save_data_b = np.array(save_data_b)

    save_data = [save_data_a0, save_data_a1, save_data_r, save_data_g, save_data_b]

    return save_data

data = read()

'''
test data:
1. filter all zero
2. count average value at the end
3. count going_up time
'''
types = ["A0", "A1", "r", "g", "b"]

fsr_going_up_time = []
fsr_end_avg = []
all_fsr_data =[]
rgb_datas = []

#把所有数据改为float32
data = [np.float32(item) for item in data]
test_data = []

data_copy = copy.deepcopy(data)

data = data_copy[0]
filtered_fsr = [x for x in data if x != 0]
filtered_fsr = np.array(filtered_fsr)

# count avg value at the end
num = int(0.1*len(filtered_fsr))
tmp = np.mean(np.array(filtered_fsr[-num:]))
test_data.append(tmp)

# count going_up time
rise_amount = np.max(filtered_fsr) - np.min(filtered_fsr)
ninety_percent_rise = 0.9 * rise_amount
index = np.argmax(filtered_fsr >= np.min(filtered_fsr) + ninety_percent_rise)
test_data.append(index)

r_mean = np.mean(data_copy[2], axis=0); test_data.append(r_mean)
g_mean = np.mean(data_copy[3], axis=0); test_data.append(g_mean)
b_mean = np.mean(data_copy[4], axis=0); test_data.append(b_mean)

# test_data = np.array(test_data[-3:]).reshape(1,3)

# normaliztion
test_data = test_data[-3:]
sum_rgb = np.sum(test_data, axis=0, keepdims=True)
test_data = test_data / sum_rgb

tmp = np.array(tmp).reshape(-1,1)
index = np.array(index).reshape(-1,1)

fsr_data = np.concatenate([tmp, index]).reshape(2)
test_data = np.concatenate((fsr_data, test_data))

#pdb.set_trace()
test_data = test_data.reshape(1,5)

y_pred = rf.predict(test_data)

print("y_pred: ", y_pred)

#ser.write((y_pred + '\n').encode('utf-8'))