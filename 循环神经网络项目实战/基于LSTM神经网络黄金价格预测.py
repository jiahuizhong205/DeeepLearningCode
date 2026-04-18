import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,LSTM
import keras

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#加载数据集
dataset = pd.read_csv("LBMA-GOLD.csv",index_col=[0])
# print(dataset)

#设置训练集长度
training_len = 1256 - 200

#获取训练集
training_set = dataset.iloc[0:training_len,[0]]

#获取测试集数据
test_set = dataset.iloc[training_len:,[0]]

#对数据集进行归一化
sc = MinMaxScaler(feature_range=(0,1))
train_set_scaled = sc.fit_transform(training_set)
test_set = sc.transform(test_set)
#不能写成sc.fit_transform(test_set),fit_transform 做了两件事：
# fit：从数据里学习最大值、最小值# transform：用学到的最值做归一化
# 对训练集：fit ✅
# 对测试集：又重新 fit 了一遍 ❌
# 这会导致：训练集和测试集使用了不同的归一化标准 模型训练时看到的是一套尺度 预测时用另一套尺度 结果完全不可信，模型必崩

#设置训练集特征和训练集标签
x_train = []
y_train = []

#设置测试集特征和测试集标签
x_test = []
y_test = []

#利用for循环进行训练集特征和标签的制作，提取数据中连续5天作为特征
for i in range(5,len(train_set_scaled)):
    x_train.append(train_set_scaled[i - 5:i,0])
    y_train.append(train_set_scaled[i,0])

#将训练集用list转化为array格式
x_train,y_train = np.array(x_train),np.array(y_train)

#循环神经网络的样本格式应该是【样本数，时间步，特征个数】
x_train = np.reshape(x_train,(x_train.shape[0],5,1))
# print(x_train.shape)

#利用for循环进行测试集特征和标签的制作，提取数据中连续5天作为特征
for i in range(5,len(test_set)):
    x_test.append(test_set[i - 5:i,0])
    y_test.append(test_set[i,0])

#将测试集用list转化为array格式
x_test,y_test = np.array(x_test),np.array(y_test)

#循环神经网络的样本格式应该是【样本数，时间步，特征个数】
x_test = np.reshape(x_test,(x_test.shape[0],5,1))
# print(x_test.shape)

#利用keras搭建神经网络
model = keras.Sequential()
model.add(LSTM(80,return_sequences=True,activation = 'relu'))
model.add(LSTM(100,return_sequences=False,activation= 'relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#对搭建好的神经网络进行编译
model.compile(loss = 'mse',optimizer=keras.optimizers.Adam(0.01))

#利用神经网络对训练集进行训练
history = model.fit(x_train,y_train,batch_size=32,epochs=100,validation_data=(x_test,y_test))

#保存训练好的模型
model.save('model.h5')

#绘制loss值
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')
plt.title("LSTM神经网络loss值")
plt.legend()
plt.show()
