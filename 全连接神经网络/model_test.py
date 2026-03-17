import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from keras.models import load_model

from 逻辑回归实战.逻辑回归实战 import dataset

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#加载数据集
dataset = pd.read_csv("breast_cancer_data.csv")
#print(dataset)

#提取数据中的特征
X = dataset.iloc[:,:-1]
#print(X)

#提取数据集中的标签
Y = dataset['target']


#划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#将数据标签转换为one——hot向量格式
y_test_one = to_categorical(y_test,2)

#数据特征进行归一化
sc = MinMaxScaler(feature_range = (0,1))
x_test = sc.fit_transform(x_test)

#导入训练好的模型
model = load_model("model.h5")

#利用训练好的模型进行测试
predict = model.predict(x_test)
# print(predict)

y_pred = np.argmax(predict,axis=1)
# print(y_pred)

#将识别的结果转化成汉字
result = []
for i in range(len(y_pred)):
    if y_pred[i] == 0:
        result.append("良性")
    else:
        result.append("恶性")
# print(result)

report = classification_report(y_test,y_pred,labels= [0,1],target_names=["良性","恶性"])
print(report)