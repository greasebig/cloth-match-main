cloth match in efficient way           
基于deepfashion      
resnet50 + multi class        
训练集过拟合正确        
测试集完全不能用        

法二  
concat     
90% 测试集正确率
先用preprocess_data多分类
再concat_data造数据
训练resnet-cloth-concat
测试cal_score-myresnet-concat