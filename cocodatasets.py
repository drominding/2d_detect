from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import matplotlib.patches as patches

# 指定数据集路径和标注文件
annotation_file = 'E:/data/COCO_HOME/annotations_trainval2017/person_keypoints_val2017.json'
dataDir = 'E:/data/COCO_HOME/val2017'
# 初始化COCO对象
coco = COCO(annotation_file)
 

 
# 获取所有图像的ID


catIds = coco.getCatIds(catNms=['person'])  # 只获取人的关键点

imgIds = coco.getImgIds(catIds=catIds)

# 获取第一张图像的ID
imgId = imgIds[10]

imgId = 376900
# 加载图像信息
img = coco.loadImgs(imgId)[0]
 
# 获取图像的文件路径
filePath = f'{dataDir}/{img["file_name"]}'
 
# 显示图像
img_data = mpimg.imread(filePath)
plt.imshow(img_data)
plt.axis('off')  # 关闭坐标轴

# 获取关键点标注
annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
print(f"anns:{anns}")
# 假设只有一个人的关键点标注（实际情况可能更复杂）
#keypoints = anns[0]['keypoints']
#print(f"keypoints:{keypoints}")
#num_keypoints = len(keypoints) // 3  # 每个关键点有三个值：x, y, visibility
 


# COCO关键点顺序和连接规则（这里是一个简单的示例，你可能需要根据实际情况调整）
# 注意：这个顺序和连接规则是基于COCO数据集的关键点定义

skeleton = [
    [5, 3], [3, 1], [1, 2], [2, 4], 
    [7, 6], [13, 12], [7, 13], [6, 12],
    [11, 9], [9, 7], [10, 8], [8, 6],
    [17, 15], [15, 13], [16, 14], [14, 12]
]
skeleton = [[x - 1, y - 1] for x, y in skeleton]
# 遍历每个人的关键点标注
for ann in anns:
    print(f"ann:{ann}")
    '''绘制关键点'''
    color_initial = (random.random(),random.random(),random.random()) #对于每一个人生成随机颜色
    keypoints = ann['keypoints']
    num_keypoints = len(keypoints) // 3  # 计算实际的关键点数量
    
    # 检查关键点数量是否足够（至少17个关键点）
    if num_keypoints < 17:
        print(f"Warning: Annotation with ID {ann['id']} has less than 17 keypoints.")
        continue  # 跳过这个标注
    
    # 绘制关键点
    for i in range(num_keypoints):
        x, y, v = keypoints[3*i], keypoints[3*i+1], keypoints[3*i+2]
        if v > 0:  # 只绘制可见的关键点
            plt.scatter(x, y, c='r', s=50)  # 使用红色圆点表示关键点
            plt.text(x, y, str(i+1), fontsize=12, color='blue', ha='center', va='center')
    
    # 绘制连接关键点的线段（只针对存在的关键点）
    valid_keypoints = [i for i in range(17) if keypoints[3*i+2] > 0]  # 获取所有可见关键点的索引
    for (start, end) in skeleton:
        if start in valid_keypoints and end in valid_keypoints:  # 只连接可见的关键点
            x1, y1 = keypoints[3*start], keypoints[3*start+1]
            x2, y2 = keypoints[3*end], keypoints[3*end+1]
            plt.plot([x1, x2], [y1, y2], '-', color = color_initial)  # 使用蓝色线段连接关键点

    '''绘制segmentation'''
    seg = ann['segmentation'][0]
    # print(seg)
    points = np.array(seg).reshape((-1, 2)) # 将坐标转换为NumPy数组，并重塑为(n, 2)的形状，其中n是顶点的数量
    #print(points)
    #print('points[:, 0]')
    #print(points[:, 0])
    # 使用Matplotlib绘制多边形
    plt.plot(points[:, 0], points[:, 1], '-', color = color_initial, alpha=0.35)  # 'r-'表示红色实线
    plt.fill(points[:, 0], points[:, 1], color = color_initial, alpha=0.35)  # 填充多边形，红色，透明度0.35
    '''绘制bbox'''
    bbox = ann['bbox']
    xset = [bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]]
    yset = [bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]]
    print(xset)
    print(yset)
    plt.plot(xset,yset,'y-')


plt.show()
