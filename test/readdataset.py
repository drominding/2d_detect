from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import matplotlib.patches as patches
def run(annotation_file,dataDir):
    coco = COCO(annotation_file)

    # 获取所有图像的ID
    catIds = coco.getCatIds(catNms=['person'])  # 只获取人的关键点
    imgIds = coco.getImgIds(catIds=catIds)
    
    # 遍历全部的imgIds，每次打开一张图片
    for imgId in imgIds[100:150]:
        print(imgId)
        # 加载图像信息
        img = coco.loadImgs(imgId)[0] # 直接load是一个[{}]结构数据，所以要取出里面的第一个索引内容
        filePath = f'{dataDir}/{img["file_name"]}' # 获取包含注释信息的图片地址
        # 加载图像
        img_data = mpimg.imread(filePath)
        plt.figure(figsize=(15,7))
        plt.imshow(img_data)
        plt.text(1,1,str(filePath))
        plt.axis('off')  # 关闭坐标轴
        #加载标注信息
        annIds = coco.getAnnIds(imgIds=imgId)
        anns = coco.loadAnns(annIds)

        #遍历每一张图片的所有标注信息，这里每一个标注信息是一个人，可能有错误
        personnum = 0
        for ann in anns:
            # 对于每一个人生成随机颜色
            color_initial = (random.random(),random.random(),random.random()) 
            personnum += 1
            # 记录是否有keypoint，和segmentation
            debugkey = 0
            debugseg = 0
            '''
            绘制关键点
            coco keypoint 17个关键点与关节点对应顺序
            https://blog.csdn.net/qq_42548340/article/details/127036434
            '''
            # 创建骨架
            skeleton = [
                        [5, 3], [3, 1], [1, 2], [2, 4], 
                        [7, 6], [13, 12], [7, 13], [6, 12],
                        [11, 9], [9, 7], [10, 8], [8, 6],
                        [17, 15], [15, 13], [16, 14], [14, 12]
                        ]
            skeleton = [[x - 1, y - 1] for x, y in skeleton] # 索引从0开始，所以骨架数字要-1
            

            try:
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
            except KeyError:
                print(f"person {personnum} no keypoints!")
                debugkey = 1
            

            '''绘制segmentation'''
            try:
                seg = ann['segmentation'][0]
                points = np.array(seg).reshape((-1, 2)) # 将坐标转换为NumPy数组，并重塑为(n, 2)的形状，其中n是顶点的数量
                # 计算多边形的中心点（这里使用顶点坐标的平均值作为近似）
                centroid = points.mean(axis=0)
                # 在中心点添加文字
                plt.text(centroid[0], centroid[1], f'person{personnum}', color = 'yellow', ha='center', va='center')
                '''
                # 计算边界框(bbox),重绘边界框
                minx, miny = points.min(axis=0)
                maxx, maxy = points.max(axis=0)
                xset = [minx,maxx,maxx,minx,minx]
                yset = [miny,miny,maxy,maxy,miny]
                plt.plot(xset,yset, color = color_initial)
                '''
                # 使用Matplotlib绘制多边形
                plt.plot(points[:, 0], points[:, 1], '-', color = color_initial, alpha=0.35)  # 'r-'表示红色实线
                plt.fill(points[:, 0], points[:, 1], color = color_initial, alpha=0.35)  # 填充多边形，红色，透明度0.35
            except KeyError:
                print(f"person {personnum} no segmentation!")
                debugseg = 1

            '''绘制bbox
            # coco2017 数据集中的bbox是错误的, 所以这个部分需要重新绘制
            try:
                bbox = ann['bbox']
                xset = [bbox[0],bbox[2],bbox[2],bbox[0],bbox[0]]
                yset = [bbox[1],bbox[1],bbox[3],bbox[3],bbox[1]]
                plt.plot(xset,yset, color = color_initial)
            except KeyError:
                print(f"person {personnum} no bbox!")
            '''
            '''重绘制bbox'''
            if debugkey == 0:
                # 通过keypoints绘制bbox
                keypoints_reshaped = np.array(keypoints).reshape((-1, 3))
                keypoints_filtered = keypoints_reshaped[~(keypoints_reshaped == [0, 0, 0]).all(axis=1)]
                bboxfromkeypoints = keypoints_filtered[:, :2]
                if debugseg == 0:
                    # 通过keypoints+segmentation绘制bbox
                    combined_data = np.concatenate((bboxfromkeypoints, points), axis=0)
                else:
                    combined_data = bboxfromkeypoints
            else:
                if debugseg == 0:
                    combined_data = points
                else:
                    combined_data = np.array([])
            if not combined_data.size:
                print(f"person {personnum} no bbox")
            else:
                minx, miny = combined_data.min(axis=0)
                maxx, maxy = combined_data.max(axis=0)
                xset = [minx,maxx,maxx,minx,minx]
                yset = [miny,miny,maxy,maxy,miny]
                plt.plot(xset,yset, color = color_initial)            


        plt.text(1,20,f'This img has {personnum} person',color = 'yellow')
        plt.show()







if __name__ == '__main__':
    annotation_file = 'E:/data/COCO_HOME/annotations_trainval2017/person_keypoints_val2017.json'
    dataDir = 'E:/data/COCO_HOME/val2017'
    run(annotation_file,dataDir)