import os
import shutil

# 定义函数来遍历文件夹并将相同类别的文件移动到新建的文件夹中
def organize_files(folder_path, arrange_train):
    # 获取所有文件
    files = os.listdir(folder_path)
    # 用字典来存储类别和文件列表
    category_dict = {}
    
    # 遍历文件
    for file_name in files:
        # 提取类别名
        category = file_name.split('_')[-1]
        category = file_name.replace(category, '')[:-1]
        # 如果类别名不存在于字典中，创建一个新的键值对
        if category not in category_dict:
            category_dict[category] = [file_name]
        else:
            category_dict[category].append(file_name)
    
    # 遍历字典，为每个类别创建文件夹并移动文件
    for category, files in category_dict.items():
        # 创建新的文件夹
        new_folder_path = os.path.join(arrange_train, category)
        os.makedirs(new_folder_path, exist_ok=True)
        # 移动文件
        for file_name in files:
            old_file_path = os.path.join(folder_path, file_name)
            new_file_path = os.path.join(new_folder_path, file_name)
            shutil.move(old_file_path, new_file_path)
        print(f"Moved {len(files)} files to {new_folder_path}")

# 调用函数并指定文件夹路径
folder_path = "/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/train_test_256/train"
arrange_train = "/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/train_test_256/arrange_train"
os.makedirs(arrange_train, exist_ok=True)
organize_files(folder_path, arrange_train)