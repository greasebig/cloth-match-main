import os
import random
from PIL import Image
from tqdm import tqdm
# Function to horizontally concatenate two images
def concat_images(image1, image2):
    width1, height1 = image1.size
    width2, height2 = image2.size
    new_width = width1 + width2
    new_height = max(height1, height2)

    new_image = Image.new('RGB', (new_width, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))
    return new_image

# Function to concatenate images in a folder pairwise and save them in a new folder
def concatenate_images_in_folder(index, folder_path, output_folder, root_folder_paths, output_folder1, subfolders):
    
    images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    num_images = 0

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            image1 = Image.open(os.path.join(folder_path, images[i]))
            image2 = Image.open(os.path.join(folder_path, images[j]))
            new_image = concat_images(image1, image2)
            new_image.save(os.path.join(output_folder, f"{index}_{i}_{j}.jpg"))
            num_images += 1
    
    for i in range(num_images):
        random_subfolder_name = random.choice(subfolders)
        random_subfolder_path = os.path.join(root_folder_paths, random_subfolder_name)

        while random_subfolder_path == folder_path: # Ensure folders are different
            random_subfolder_name = random.choice(subfolders)
            random_subfolder_path = os.path.join(root_folder_paths, random_subfolder_name)

        random_images = [f for f in os.listdir(random_subfolder_path) if f.endswith('.jpg') or f.endswith('.png')]

        image1 = random.choice(images)
        image2 = random.choice(random_images)

        image1 = Image.open(os.path.join(folder_path, image1))
        image2 = Image.open(os.path.join(random_subfolder_path, image2))

        new_image = concat_images(image1, image2)
        new_image.save(os.path.join(output_folder1, f"{index}_{i}.jpg"))


# Function to concatenate images from different folders
def concatenate_images_from_different_folders(folder_paths, output_folder, num_images):
    
    for _ in range(num_images):
        folder1 = random.choice(folder_paths)
        folder2 = random.choice(folder_paths)
        while folder1 == folder2: # Ensure folders are different
            folder2 = random.choice(folder_paths)

        image1 = random.choice(os.listdir(folder1))
        image2 = random.choice(os.listdir(folder2))

        image1 = Image.open(os.path.join(folder1, image1))
        image2 = Image.open(os.path.join(folder2, image2))

        new_image = concat_images(image1, image2)
        new_image.save(os.path.join(output_folder, f"{random.randint(0, 999999)}.jpg"))

# Main function
if __name__ == "__main__":
    root_folder_paths = '/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/train_test_256/arrange_test' # Update with your folder paths
    output_folder0 = "/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/train_test_256/concatdata-test/folder0"
    output_folder1 = "/data/lujunda/baidu/duomo/data/leftdata/clipscore/cloth-matvh/train_test_256/concatdata-test/folder1"
    os.makedirs(output_folder0, exist_ok=True)
    os.makedirs(output_folder1, exist_ok=True)
    #for folder_path in folder_paths:
    #    concatenate_images_in_folder(folder_path, output_folder1)

    #concatenate_images_from_different_folders(folder_paths, output_folder0, num_images=len(os.listdir(output_folder1)))

    subfolders = [f for f in os.listdir(root_folder_paths) if os.path.isdir(os.path.join(root_folder_paths, f))]
    for index, (root, dirs, files) in tqdm(enumerate(os.walk(root_folder_paths))):
        #print("当前目录:", root)
        #print("包含的文件夹:", dirs)
        #print("包含的文件:", files)
        #print()
        concatenate_images_in_folder(index, root, output_folder0, root_folder_paths, output_folder1, subfolders)

    

    

    # 随机选择一个子文件夹
    

    #print("随机选择的子文件夹路径:", random_subfolder_path)