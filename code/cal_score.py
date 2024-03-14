
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import os
# 加载ResNet50模型
model = ResNet50(weights='imagenet')

# 加载两张图片



img_path22 = '/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/similar1/3.jpg'

folder_path = "/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/similar1"
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):  # 如果是文件而不是文件夹
        image_path1 = file_path

        img_path2 = image_path1
        print(img_path2)
        img1 = image.load_img(img_path22, target_size=(224, 224))
        x1 = image.img_to_array(img1)
        x1 = np.expand_dims(x1, axis=0)
        x1 = preprocess_input(x1)

        img2 = image.load_img(img_path2, target_size=(224, 224))
        x2 = image.img_to_array(img2)
        x2 = np.expand_dims(x2, axis=0)
        x2 = preprocess_input(x2)

        # 获取两张图片的特征向量
        feat1 = model.predict(x1)
        feat2 = model.predict(x2)

        # 计算特征向量的相似度
        similarity = np.dot(feat1, feat2.T) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        print(similarity)
        # 判断两张图片是否相似
        if similarity > 0.4:
            print("两张图片的衣服相似")
        else:
            print("两张图片的衣服不相似")