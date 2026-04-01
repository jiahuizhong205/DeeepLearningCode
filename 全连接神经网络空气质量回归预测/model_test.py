import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.models import load_model
from math import  sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error


#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#导入数据集
dataset = pd.read_csv("data.csv")

# ==============================================
# 【正确步骤1】先划分特征和标签
# ==============================================
X = dataset.iloc[:,:-1].values  # 特征
Y = dataset.iloc[:,-1].values   # 标签（回归任务）

# 把Y变成二维，适应scaler
Y = Y.reshape(-1, 1)

# ==============================================
# 【正确步骤2】先划分数据集！！！（最重要）
# ==============================================
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=21)

# ==============================================
# 【正确步骤3】分别归一化特征 X 和标签 Y
# ==============================================
# 归一化特征
sc_x = MinMaxScaler(feature_range=(0,1))
x_train_scaled = sc_x.fit_transform(x_train)  # 只在训练集fit
x_test_scaled = sc_x.transform(x_test)        # 测试集只用训练集参数

# 归一化标签（神经网络建议做）
sc_y = MinMaxScaler(feature_range=(0,1))
y_train_scaled = sc_y.fit_transform(y_train)
y_test_scaled = sc_y.transform(y_test)


#加载已经训练好的模型
model = load_model("model.h5")


#利用训练好的模型进行预测
y_pred_scaled = model.predict(x_test_scaled)

# 逆归一化得到真实预测值
y_pred = sc_y.inverse_transform(y_pred_scaled)
# ====================================================

# 打印对比结果
print("归一化预测值：\n", y_pred_scaled[:5])
print("真实预测值（逆归一化后）：\n", y_pred[:5])
print("真实标签：\n", y_test[:5])

# 1. 导入库
# 2. 读取数据（必须和训练同一份处理逻辑）
# 3. 拆分 X、Y
# 4. 划分训练集/测试集（random_state 必须和训练一样）
# 5. 重新定义 sc_x，并只在训练集fit：sc_x.fit_transform(x_train)
# 6. 归一化测试集特征：sc_x.transform(x_test)
# 7. 重新定义 sc_y，并只在训练集fit：sc_y.fit_transform(y_train)
# 8. 加载模型：load_model("model.h5")
# 9. 用【归一化后的测试特征】预测：model.predict(x_test_scaled)
# 10. 对预测结果逆归一化：sc_y.inverse_transform(y_pred_scaled)