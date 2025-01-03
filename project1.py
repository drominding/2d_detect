'''
先读取老师姿态，再根据老师姿态评估学生姿态
'''
from colorama import init, Fore, Back, Style
import cv2
import numpy as np
import mediapipe as mp
import time #计算fps值
# 初始化 colorama
init(autoreset=True)

'''
计算动作相似程度
    对于结果进行加工，仅保留手臂和双腿的关键点
    左边的手: 12-14, 12,16
    右边的手: 11-13, 11-15
    左边的脚: 24-16, 24-28
    右边的脚: 23-25, 25-27
    左边的躯干: 12-24
    右边的躯干: 11,23
'''

class count2point:
    def  __init__(self, student_results,student_img,teacher_results,teacher_img,student_ratio=1, teacher_ratio=1):
        self.student_results = student_results
        self.student_img = student_img
        self.student_ratio = student_ratio

        self.teacher_results = teacher_results
        self.teacher_img = teacher_img
        self.teacher_ratio = teacher_ratio
        
    
    def vector(self,fix_point,free_point):
        student_angle_vector = obtain_angle_vector(fix_point,free_point,self.student_results,self.student_img)
        teacher_angle_vector = obtain_angle_vector(fix_point,free_point,self.teacher_results,self.teacher_img)
        return cosine_similarity(student_angle_vector,teacher_angle_vector)
    

    
    def distance(self,fix_point,free_point,student_ratio = None,teacher_ratio = None):

        sh, sw, sc = self.student_img.shape
        th, tw, tc = self.teacher_img.shape
        print(f'thecher.shpe = {self.teacher_img.shape}')
        scale_index = th/sh
        print(f'scale2 = {scale_index} {sh} {th}')
        student_point_distance = count_distance(fix_point,free_point,self.student_results,self.student_img) * scale_index 
        teahcer_point_distance = count_distance(fix_point,free_point,self.teacher_results,self.teacher_img) 

        if student_ratio == None:
            student_point_distance = student_point_distance * self.student_ratio
        else:
            student_point_distance = student_point_distance * student_ratio

        if teacher_ratio == None:
            teahcer_point_distance = teahcer_point_distance * self.teacher_ratio
        else:
            teahcer_point_distance = teahcer_point_distance * teacher_ratio
        return student_point_distance, teahcer_point_distance

def count_distance(fix_point,free_point,results,img):
    h, w, c = img.shape
    '''计算两点之间的距离'''
    results = results.pose_landmarks.landmark
    fix_point_x = results[fix_point].x*w
    fix_point_y = results[fix_point].y*h
    free_point_x = results[free_point].x*w
    free_point_y = results[free_point].y*h
    return np.linalg.norm(np.array([fix_point_x,fix_point_y]) - np.array([free_point_x,free_point_y]))
        


def cosine_similarity(vec1, vec2):
    # 将向量转换为NumPy数组（如果它们还不是的话）
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 归一化向量
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    # 计算点积作为余弦相似度
    cos_sim = np.dot(vec1_norm, vec2_norm)

    # 将结果归一化
    norm_cos_sim = (cos_sim + 1)/2
    
    return norm_cos_sim




def obtain_angle_vector(fix_point,free_point,results,img):
    '''
    获取节点的角度向量
    '''
    results = results.pose_landmarks.landmark
    x = results[free_point].x - results[fix_point].x
    y = results[free_point].y - results[fix_point].y
    z = results[free_point].z - results[fix_point].z
    h,w, c = img.shape
    # vec = [x*w,y*h,z*w/5]
    # 只计算xy，不计算深度z
    vec = [x*w,y*h]
    return vec

def similarity_score_simple(aa, bb):
    """平面上有两组点a1,a2,b1,b2。先计算点a1a2的距离aa,再计算点b1b2的距离bb,
    如果aa与bb长度越相近.输出结果越接近1.差异越大输出结果越接近0"""
    # 使用指数函数来平滑距离差异，并映射到 [0, 1] 区间
    # 这里的 beta 是一个控制平滑程度的参数，可以根据需要调整
    beta = 1.0
    similarity = np.exp(-np.abs(aa - bb) / beta)
    # 由于指数函数总是返回正数，我们不需要再使用夹逼法
    return similarity

# 初始化MediaPipe的姿态模型
mpPose1 = mp.solutions.pose
pose1 = mpPose1.Pose()
mpDraw1 = mp.solutions.drawing_utils

# 加载教师图片
image1 = cv2.imread(r'E:\CODE\2D_detect\data\stu_0.png')
# 将图片转换为RGB格式，因为MediaPipe需要RGB格式的图片
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# 对第一张图片进行关键点检测
results1 = pose1.process(image1_rgb)

#获取手臂的角度向量
vector1 = obtain_angle_vector(12,14,results1,image1)

# 绘图并保存
mpDraw1.draw_landmarks(image1, results1.pose_landmarks, mpPose1.POSE_CONNECTIONS)
for id, lm in enumerate(results1.pose_landmarks.landmark):
    h, w, c = image1.shape
    cx, cy = int(lm.x * w), int(lm.y * h)
    cv2.circle(image1, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    text_y = cy - 10 if cy - 10 > 0 else cy + 10
    cv2.putText(image1, str(id), (cx - 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

cv2.imwrite("image1.png", image1)



#加载学生视频
# 初始化MediaPipe的姿态模型
mpPose = mp.solutions.pose
pose = mpPose.Pose()
#初始化画图工具
mpDraw = mp.solutions.drawing_utils
 
#调用摄像头，在同级目录下新建Videos文件夹，然后在里面放一些MP4文件，方便读取
cap = cv2.VideoCapture(r"E:\CODE\2D_detect\data\teacher_video_30fps.mp4")
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
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    #检测到人体的话：
    if results.pose_landmarks:
        '''
        计算学生的节点角度
        '''
        vector2 = obtain_angle_vector(12,14,results,img)
        #计算相似度
        countprocess = count2point(results,img,results1,image1,student_ratio=184/462)
        print(countprocess.vector(12,14))
        print(f'距离相似程度{countprocess.distance(12,14)}')
        print(f'左大臂相似度{cosine_similarity(vector1,vector2)}')
        print(f'左小臂相似度{countprocess.vector(14,16)}')
        print(f'左肩到手掌相似度{countprocess.vector(12,16)}')
        print(Fore.RED + f'左右肩膀相似程度{countprocess.vector(12,11)}')
        # 计算距离
        h, w, c = img.shape
        h1, w1, c1 = image1.shape
        print(f'student图片尺寸为：{w} x {h}')
        print(f'teacher图片尺寸为：{w1} x {h1}')
        # 按高度统一尺寸，只改变student的尺寸计算缩放因子
        scale_index = h1/h 
        print(f'scale1 = {scale_index} {h} {h1}')
        distans1 = np.linalg.norm(np.array([results.pose_landmarks.landmark[14].x*w, results.pose_landmarks.landmark[14].y*h]) - np.array([results.pose_landmarks.landmark[12].x*w, results.pose_landmarks.landmark[12].y*h]))
        # print(f'点14,到12的距离(像素）是{distans1}')
        print(Fore.RED + f'距离{countprocess.distance(11,23)}')
        print(count_distance(11,23,results,img)*scale_index)
        print(count_distance(11,23,results1,image1))


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
    input("push enter")