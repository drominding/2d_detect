import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
 
import time #计算fps值



#两个初始化
mpPose = mp.solutions.pose
pose = mpPose.Pose()
 #初始化画图工具
mpDraw = mp.solutions.drawing_utils
 
#调用摄像头，在同级目录下新建Videos文件夹，然后在里面放一些MP4文件，方便读取
# cap = cv2.VideoCapture(0) #调用摄像头
cap = cv2.VideoCapture("data/student_video_30fps.mp4")
#计算pfs值需要用到的变量，先初始化以一下
pTime = 0

while cap.isOpened():
#读取图像
    success, img = cap.read()
    if success == False:
        break
    #转换为RGB格式，因为Pose类智能处理RGB格式，读取的图像格式是BGR格式
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #处理一下图像
    plt.imshow(imgRGB)
    results = pose.process(imgRGB)
    
    # print(results.pose_landmarks)
    #检测到人体的话：
    if results.pose_landmarks:







        #如果我们想对33个关键点中的某一个进行特殊操作，需要先遍历33个关键点



        for id, keypoint in enumerate(results.pose_landmarks.landmark):
        #打印出来的关键点坐标都是百分比的形式，我们需要获取一下视频的宽和高
            h, w, c = img.shape
            print(id, keypoint)
            #将x乘视频的宽，y乘视频的高转换成坐标形式
            cx, cy = int(keypoint.x * w), int(keypoint.y * h)
            #使用cv2的circle函数将关键点特殊处理
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            text_y = cy - 10 if cy - 10 > 0 else cy + 10  # 确保文本不会超出图像边界
            cv2.putText(img, str(id), (cx - 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)  # 使用抗锯齿字体
         #计算fps值
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.imwrite("saved_image.png", img)
        cv2.waitKey(1)


        pose_landmarks = results.pose_landmarks.landmark
        # print(pose_landmarks)
        print(pose_landmarks[0].visibility)
        # 只选择部分关键节点
        part_keypoint_index = [
            16,14,12,
            15,13,11,
            28,26,24,
            27,25,23
        ]
        #定义骨架
        skeleton = [
            [16,14],[14,12],
            [15,13],[13,11],
            [12,11],[11,23],[23,24],[24,12],
            [28,26],[26,24],
            [27,25],[25,23]

        ]

        #用黄色连接身体
        h, w, c = img.shape
        # 获取可见关键点索引
        visible_keypoints = [i for i in range(32) if pose_landmarks[i].visibility > 0.5]
        # 绘出可见点
        for i in visible_keypoints:
            plt.scatter(pose_landmarks[i].x*w,pose_landmarks[i].y*h, c='r', s=50)
        
        # 按骨架连接可见点
        for (start,end) in skeleton:
            print(f'start:{start} and end:{end}')
            if start in visible_keypoints and end in visible_keypoints:
                x1,y1,z1 = pose_landmarks[start].x*w,pose_landmarks[start].y*h,pose_landmarks[start].z*w
                x2,y2,z2 = pose_landmarks[end].x*w,pose_landmarks[end].y*h,pose_landmarks[end].z*w
                print(f'点{start}:{pose_landmarks[start]}')
                print(f'点{end}:{pose_landmarks[end]}')
                plt.plot([x1,x2], [y1,y2],[z1,z2], marker='o', linestyle='-', color='b')
        plt.show()
        '''
        print(f'visible_keypoints:{visible_keypoints}')
        # 根据骨架绘制关键点连线
        for (start,end) in skeleton:
            print(f'start:{start} and end:{end}')
            if start in visible_keypoints and end in visible_keypoints:
                x1,y1,z1 = pose_landmarks[start].x*w,pose_landmarks[start].y*h,pose_landmarks[start].z*w
                x2,y2,z2 = pose_landmarks[end].x*w,pose_landmarks[end].y*h,pose_landmarks[end].z*w
                print(f'点{start}:{pose_landmarks[start]}')
                print(f'点{end}:{pose_landmarks[end]}')
                plt.plot([x1,y1], [x2,y2], marker='o', linestyle='-', color='b')
        plt.show()        
        '''

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in part_keypoint_index:
            if pose_landmarks[i].visibility > 0.5:
                x = pose_landmarks[i].x*w
                y = pose_landmarks[i].y*h
                z = pose_landmarks[i].z*w
                ax.scatter(x, y, z, c='r', marker='o')
                print(f'第{i}个点的x,y,z:{x}{y}{z}')
        plt.show()
        '''

    '''
    exit_program = False
    while True:
        key = cv2.waitKey(1) & 0xFF  # & 0xFF是为了确保按键值是8位无符号整数
        if key == 32: #空格键的 ASCII码
            break
        elif key == 27:
            exit_program = True
            break
    if exit_program == True:
        break
    '''