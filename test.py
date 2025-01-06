import os
 
def get_image_files(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
    """
    获取指定目录下所有图片文件的路径。
 
    :param directory: 要遍历的目录路径
    :param extensions: 图片文件的扩展名列表
    :return: 图片文件路径的列表
    """
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                image_files.append(os.path.join(root, file))
    return image_files
 
# 示例使用
directory_path = r'E:\CODE\2D_detect\test_img'  # 替换为你的图片文件夹路径
image_files = get_image_files(directory_path)
 
# 迭代图片文件路径
for image_path in image_files:
    print(image_path)
    # 在这里可以添加处理每个图片文件的代码