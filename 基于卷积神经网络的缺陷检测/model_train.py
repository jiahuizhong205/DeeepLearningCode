import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D


#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#读取训练集数据
data_train = './data/train'
data_train = pathlib.Path(data_train)

#读取训练集数据
data_val = './data/val'
data_val = pathlib.Path(data_val)


#数据类别放置到列表数据中
CLASS_NAMES = np.array(['Cr','In','Pa','Ps','Rs','Sc'])

#设置图片大小和批次数
BATCH_SIZE = 64
IMG_HEIGHT = 32
IMG_WIDTH = 32

#对数据进行归一化处理
image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

#训练集生成器
train_data_gen = image_generator.flow_from_directory(directory=str(data_train),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                     classes=list(CLASS_NAMES))

#验证集生成器
val_data_gen = image_generator.flow_from_directory(directory=str(data_val),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                     classes=list(CLASS_NAMES))

#利用keras搭建卷积神经网络
model = keras.Sequential()
model.add(Conv2D(filters=6,kernel_size=5,input_shape=(32,32,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=16,kernel_size=5,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=120,kernel_size=5,activation='relu'))
model.add(Flatten())
model.add(Dense(84,activation='relu'))
model.add(Dense(6,activation='softmax'))

#编译卷积神经网络
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

#传入数据集进行训练
history = model.fit(train_data_gen,validation_data=val_data_gen,epochs=50)

#保存训练好的模型
model.save("model.h5")

#绘制loss图
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')
plt.title("CNN神经网络loss值")
plt.legend()
plt.show()

#绘制准确率
plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='val')
plt.title("CNN神经网络accuracy值")
plt.legend()
plt.show()