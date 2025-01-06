from colorama import init, Fore, Back, Style
import cv2
import numpy as np
import mediapipe as mp
import time #计算fps值
import utils
import os
# 初始化 colorama
init(autoreset=True)


#加载学生视频
# 初始化MediaPipe的姿态模型
mpPose = mp.solutions.pose
pose = mpPose.Pose()
#初始化画图工具
mpDraw = mp.solutions.drawing_utils
 
#调用摄像头，在同级目录下新建Videos文件夹，然后在里面放一些MP4文件，方便读取
cap = cv2.VideoCapture(0)


'''获取参考示例文件'''
directory_path = r'E:\CODE\2D_detect\test_img'  # 替换为你的图片文件夹路径
image_files = utils.get_image_files(directory_path)

teacher_reasults = []
imgnum = 0
mpPose = mp.solutions.pose

for image_path in image_files:
    pose1 = mpPose.Pose()

    # 加载教师图片
    image1 = cv2.imread(image_path)
    # 将图片转换为RGB格式，因为MediaPipe需要RGB格式的图片
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # 对第一张图片进行关键点检测
    results1 = pose1.process(image1_rgb)

    mpDraw1 = mp.solutions.drawing_utils
    imgnum+=1
    # 绘制并保存
    mpDraw1.draw_landmarks(image1, results1.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results1.pose_landmarks.landmark):
        h, w, c = image1.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image1, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        text_y = cy - 10 if cy - 10 > 0 else cy + 10
        cv2.putText(image1, str(id), (cx - 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"E:\\CODE\\2D_detect\\checked_img\\{imgnum}.png", image1)
    # 将所有结果进行保存
    teacher_reasults.append([results1,image1])

#计算pfs值需要用到的变量，先初始化以一下
pTime = 0
imgcount = 0
while cap.isOpened():
    for results1,image1 in teacher_reasults:
        imgcount +=1
        print(f'第{imgcount}次作业')
        min_accuracy = 0
        while min_accuracy < 0.9:
            # 需要保持动作5s
            hold_accuracy_list = []
            for second in range(5):
                # 每20帧截取一次
                accuracy_list = []
                frame = 0
                while frame < 20:
                    accuracy = 0
                    #读取图像
                    success, img = cap.read()
                    if success == False:
                        break
                    frame+=1
                    #转换为RGB格式，因为Pose类智能处理RGB格式，读取的图像格式是BGR格式
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #处理一下图像
                    results = pose.process(imgRGB)
                    # print(results.pose_landmarks)
                    #检测到人体的话：
                    if results.pose_landmarks:
                        '''
                        计算学生的节点角度
                        '''

                        #计算相似度
                        countprocess = utils.count2point(results,img,results1,image1,student_ratio=1,teacher_ratio=1)
                        accuracy = countprocess.count(12,16)
                        # 将准确路存储在list中
                        accuracy_list.append(accuracy)

                    #使用mpDraw来刻画人体关键点并连接起来
                        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                        #如果我们想对33个关键点中的某一个进行特殊操作，需要先遍历33个关键点
                        for id, lm in enumerate(results.pose_landmarks.landmark):
                        #打印出来的关键点坐标都是百分比的形式，我们需要获取一下视频的宽和高
                            h, w, c = img.shape
                            # print(id, lm)
                            #将x乘视频的宽，y乘视频的高转换成坐标形式
                            cx, cy = int(lm.x * w), int(lm.y * h)
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
                    # cv2.imwrite("saved_image.png", img)
                    cv2.waitKey(1)

                # 5帧结束 获得最小准确率
                total_sum = sum(accuracy_list)
                # 计算列表的长度
                count = len(accuracy_list) + 0.000001
                # min_accuracy = min(accuracy_list)
                # 计算平均数
                average_accuracy = total_sum / count
                print(f'每20帧的准确率{accuracy_list}')
                print(f'每20帧的准确率{average_accuracy}')
                print(f'第{imgcount}次作业')
                hold_accuracy_list.append(average_accuracy)
            min_accuracy = min(hold_accuracy_list)
            print(Fore.RED + f'5秒钟内最差的成绩{min_accuracy}')
        input('enter')
                
