import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense

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

# ==============================================
# 搭建神经网络（指定input_dim）
# ==============================================
model = keras.Sequential()
model.add(Dense(16, activation='relu', input_dim=x_train.shape[1]))  # 指定输入维度
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# ==============================================
# 编译：使用adam，比SGD好太多
# ==============================================
model.compile(loss='mse', optimizer='adam')

# ==============================================
# 训练
# ==============================================
history = model.fit(
    x_train_scaled,
    y_train_scaled,
    epochs=100,
    batch_size=24,
    verbose=2,
    validation_data=(x_test_scaled, y_test_scaled)
)

model.save("model.h5")

# ==============================================
# 绘制loss曲线
# ==============================================
plt.plot(history.history['loss'],label='训练loss')
plt.plot(history.history['val_loss'],label='验证loss')
plt.title("全连接神经网络loss值图")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

# ==============================================
# 【重要】预测后逆归一化，得到真实预测值
# ==============================================
y_pred_scaled = model.predict(x_test_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled)  # 还原真实值
y_test_real = sc_y.inverse_transform(y_test_scaled)
