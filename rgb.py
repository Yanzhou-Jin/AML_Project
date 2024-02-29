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

with open('processed_data.pkl', 'rb') as handle:
    data = pickle.load(handle)

fruits = ['apple', 'lime', 'lemon', 'orange', 'pear', 'tomato']
for item in fruits:
    data[f'{item}'].pop("A0")
    data[f'{item}'].pop("A1")
rgb = data

# 很重要：我们把RGB全部当成了时间序列的形式
# 6个物体，每个物体16个数据
# 测试的时候，直接将rgb保留16组（也就是16个rgb，多的部分截去，30s时间内肯定不止16个rgb)，然后将这16个rgb取平均，来作为knn.predict的输入
# 但是必须要知道，我们模型训练并不是在时间维度上取平均，而是在10个样本上取平均，因此训练时，用于每个物体的实际上是一串16个rgb的序列

for index, item in enumerate(fruits):
    tmp=[]
    for i in range(16):
        tmp.append(np.array([rgb[f'{item}']['r'][i], rgb[f'{item}']['g'][i], rgb[f'{item}']['b'][i]]))
    if index == 0:
        rgb_data = np.array(tmp).reshape(1,16,3)
        rgb_label = np.array([index for _ in range(16)]).reshape(1,16)
    else:
        rgb_data = np.concatenate((rgb_data, np.array(tmp).reshape(1,16,3)), axis=0)
        rgb_label = np.concatenate((rgb_label, np.array([index for _ in range(16)]).reshape(1,16)), axis=0)

train_data = []
train_label = []
test_data = []
test_label = []

for i in range(6):
    X_train, X_test, y_train, y_test = train_test_split(
        rgb_data[i],
        rgb_label[i],
        test_size=4,  
        random_state=10+i
    )
    
    train_data.append(X_train)
    train_label.append(y_train)
    test_data.append(X_test)
    test_label.append(y_test)

train_data = np.concatenate(train_data, axis=0)
train_label = np.concatenate(train_label, axis=0)
test_data = np.concatenate(test_data, axis=0)
test_label = np.concatenate(test_label, axis=0)


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_data, train_label)
predicted_label = knn.predict(test_data)
test_accuracy = sum(predicted_label==test_label) / len(test_label)
print(test_accuracy)

'''
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义一个卷积层，输入通道为1，输出通道为3，卷积核大小为1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=1)
        # 定义一个全连接层，输入特征数量为3（卷积层输出通道数），输出特征数量为6
        self.fc1 = nn.Linear(in_features=9, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=6)

    def forward(self, x):
        # 应用卷积层
        x = F.relu(self.conv1(x))
        # 在进入全连接层之前，将数据平铺
        x = x.view(-1, 9)
        # 应用全连接层
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x

model = SimpleCNN()

train_data = torch.tensor(train_data).reshape(-1,1,3).to(torch.float)
test_data = torch.tensor(test_data).reshape(-1,1,3).to(torch.float)
train_label = torch.tensor(train_label).to(torch.int64) ; train_label = torch.nn.functional.one_hot(train_label, num_classes=6).float()
test_label = torch.tensor(test_label).to(torch.int64) ; test_label = torch.nn.functional.one_hot(test_label, num_classes=6).float()

dataset = TensorDataset(train_data, train_label)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        output = model(data)
        #pdb.set_trace()
        loss = criterion(output, target)

        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
'''