import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv  # 导入csv模块
import pdb
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

# first fsr, then rgb

fruits = ["yellowsponge", "lime", "lemon", "tomato", "whitefoam"]
types = ["A0", "A1", "r", "g", "b"]

data = {}
rgb = {}

fsr_going_up_time = []
fsr_end_avg = []
all_fsr_data =[]

for fruit in fruits:
    rgb[fruit] = {tp: [] for tp in ["r","g","b"]}
    for i in range(1, 31):  
        rgb_datas = {}
        for tp in types:
            if tp=="A1":
                continue
            file_name = f'{fruit}{i}.txt'
            # file_path = f"/mnt/d/imperial/AML_Lab/real_data2/{tp}/{file_name}"
            file_path = f"D:\imperial\AML_lab\\real_data2\{tp}\{file_name}"

            # read data
            with open(file_path, 'r') as file:
                data = [list(map(int, line.strip().split())) for line in file]
            #pdb.set_trace()
            
            if tp == "A0": # fsr process
                # filter all zero
                filtered_fsr = [x for x in data if x[0] != 0]
                filtered_fsr = np.array(filtered_fsr)
                all_fsr_data.append(filtered_fsr)

                # count avg value at the end
                num = int(0.1*len(filtered_fsr))
                tmp = np.mean(np.array(filtered_fsr[-num:]))
                fsr_end_avg.append(tmp)

                # count going_up time
                rise_amount = np.max(filtered_fsr) - np.min(filtered_fsr)
                ninety_percent_rise = 0.9 * rise_amount
                index = np.argmax(filtered_fsr >= np.min(filtered_fsr) + ninety_percent_rise)
                fsr_going_up_time.append(index)
            
            if tp in ["r", "g", "b"]: # rgb process
                rgb_mean = np.mean(data, axis=0)
                rgb_datas[tp] = rgb_mean
            
        # scale rgb
        total_sum = sum(rgb_datas.values())
        scaled_rgb_datas = {tp: rgb_datas[tp] / total_sum for tp in ["r", "g", "b"]}
        
        for tp in ["r", "g", "b"]:
            rgb[fruit][tp].append(scaled_rgb_datas[tp])

for index, item in enumerate(fruits):
    tmp=[]
    for i in range(30):
        tmp.append(np.array([rgb[f'{item}']['r'][i], rgb[f'{item}']['g'][i], rgb[f'{item}']['b'][i]]))
    if index == 0:
        rgb_data = np.array(tmp).reshape(1,30,3)
    else:
        rgb_data = np.concatenate((rgb_data, np.array(tmp).reshape(1,30,3)), axis=0)

rgb_data = rgb_data.reshape(5,30,3)

fsr_going_up_time = np.array(fsr_going_up_time).reshape(5,30,1)
fsr_end_avg = np.array(fsr_end_avg).reshape(5,30,1)

# count variance
avg = [] ; var = []
for i in range(0,5,1):
    avg.append(np.mean(fsr_end_avg[i].flatten()))
avg = np.array(avg) ; fsr_avg = fsr_end_avg.reshape(150)
for i in range(0,150,1):
    var.append(fsr_avg[i] - avg[int(i//30)])
var = np.array(var) ; var = var.reshape(5,30,1)


# final fsr_data
fsr_data = np.concatenate((fsr_going_up_time, fsr_end_avg),axis=2)

label=[]
for i in range(0,5,1):
    label.extend([i]*30)
label = np.array(label).reshape(5,30)

total_data = np.concatenate((fsr_data,rgb_data), axis=2) # 5*30*5
#total_data = total_data[:,:,-3:]

train_data = [] ; train_label = []
test_data = [] ; test_label = []


for i in range(5): # 7 classes
    X_train, X_test, y_train, y_test = train_test_split(
        total_data[i],
        label[i],
        test_size=5,  
        random_state=20+i
    )
    
    train_data.append(X_train); train_label.append(y_train)
    test_data.append(X_test) ; test_label.append(y_test)

train_data = np.concatenate(train_data, axis=0) ; train_label = np.concatenate(train_label, axis=0)
test_data = np.concatenate(test_data, axis=0) ; test_label = np.concatenate(test_label, axis=0)

'''
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(train_data[:,:3], train_label)
dump(clf, "knn_final.joblib")
'''

# train (fsr+rgb)
clf = RandomForestClassifier(n_estimators=100)  
clf.fit(train_data, train_label)
dump(clf, 'rf_final.joblib')


# test
y_pred = clf.predict(test_data)
acc = np.sum(y_pred == test_label) / len(test_label)
print("acc: ", acc)

'''
# rgb train 
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_data[:,3:], train_label)
'''

'''
# rgb test
fsr_incorrect_indices = np.where(y_pred_fsr != test_label)[0]
rgb_test_data_incorrect = test_data[fsr_incorrect_indices]
rgb_test_labels_incorrect = test_label[fsr_incorrect_indices]
y_pred_rgb = knn.predict(rgb_test_data_incorrect[:,3:])
rgb_acc = np.sum(y_pred_rgb == rgb_test_labels_incorrect) / len(rgb_test_labels_incorrect)
print("rgb_acc: ", rgb_acc)

rgb_incorrect_indices = np.where(y_pred_rgb != rgb_test_labels_incorrect)[0]
final_error_num = len(rgb_incorrect_indices)
print("final_acc: ", (210-final_error_num)/210)
'''