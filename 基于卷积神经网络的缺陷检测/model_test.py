import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2



#数据类别放置到列表数据中
CLASS_NAMES = np.array(['Cr','In','Pa','Ps','Rs','Sc'])

#设置图片大小
IMG_HEIGHT = 32
IMG_WIDTH = 32

#加载模型
model = load_model("model.h5")

#数据读取与预处理
src = cv2.imread("data/val/Cr/Cr_32.bmp")
src = cv2.resize(src,(32,32))
src = src.astype("int32")
src = src / 255

#扩充数据的维度
test_img = tf.expand_dims(src,0)
# print(test_img.shape)

preds = model.predict(test_img)
# print(preds)
score = preds[0]
# print(score)

print('模型预测的结果为{}，概率为{}'.format(CLASS_NAMES[np.argmax(score)],np.max(score)))