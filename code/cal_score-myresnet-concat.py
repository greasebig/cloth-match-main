
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import os
from PIL import Image

model = torchvision.models.resnet50(pretrained=True)


num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)


#model.load_state_dict(torch.load('/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/checkpoint/resnet50_weights.pth'))
#model.load_state_dict(torch.load('/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/checkpoint/resnet50_weights-all.pth'))
#model.load_state_dict(torch.load('/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/checkpoint/checkpoint_epoch_10.pth'))
model.load_state_dict(torch.load('/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/checkpoint/resnet50_weights-concat.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((192, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


folder_path = "/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/train_test_256/concat-test-sub"

#img_path1 = '/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/test-sub/fashionMENJackets_Vestsid0000065304_4full.jpg'
#folder_path = "/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/test-sub"
#img_path1 = '/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/trainsub/fashionMENJackets_Vestsid0000008405_1front.jpg'
#folder_path = "/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/trainsub"


for file_name in os.listdir(folder_path):

    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):  # 如果是文件而不是文件夹
        
        img_path2 = file_path
        print(img_path2)

        

        image2 = Image.open(img_path2)
        image2 = transform(image2)
        image2 = image2.unsqueeze(0)  # 增加 batch 维度

        
        # 推理
        with torch.no_grad():
            
            output2 = model(image2)

        # 获取预测结果
        

        probabilities2 = torch.softmax(output2, dim=1)
        predicted_class2 = torch.argmax(probabilities2).item()
        print(predicted_class2)
        if predicted_class2 == 0:
            print("两张图片的衣服相似")
        else:
            print("两张图片的衣服不相似")