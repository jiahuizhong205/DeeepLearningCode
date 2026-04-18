import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import load_model

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

#设置测试集特征和测试集标签
x_test = []
y_test = []

#利用for循环进行测试集特征和标签的制作，提取数据中连续5天作为特征
for i in range(5,len(test_set)):
    x_test.append(test_set[i - 5:i,0])
    y_test.append(test_set[i,0])

#将测试集用list转化为array格式
x_test,y_test = np.array(x_test),np.array(y_test)

#循环神经网络的样本格式应该是【样本数，时间步，特征个数】
x_test = np.reshape(x_test,(x_test.shape[0],5,1))
# print(x_test.shape)

#导入模型
model = load_model('model.h5')

#利用模型进行测试
predicted = model.predict(x_test)
# print(predicted.shape)

#进行预测值的反归一化
prediction = sc.inverse_transform(predicted)
# print(prediction)

#对测试值的标签进行反归一化
real = sc.inverse_transform(test_set[5:])
# print(real)

#打印模型的评价指标
rmse = sqrt(mean_squared_error(prediction,real))
mape = np.mean(np.abs((real-prediction)/prediction))
print('rmse',rmse)
print('mape',mape)

#预测真实值和预测值的对比
plt.plot(real,label='真实值')
plt.plot(prediction,label = '预测值')
plt.title("基于LSTM神经网络的黄金价格预测")
plt.legend()
plt.show()