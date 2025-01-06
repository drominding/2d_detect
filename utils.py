from colorama import init, Fore, Back, Style
import cv2
import numpy as np
import mediapipe as mp
import time #计算fps值
import os

def colorprint(num):
    if num > 0.9:
        color = Fore.RED
    elif num < 0.5:
        color = Fore.GREEN
    else:
        color = Fore.RESET
    print(f'{color}{str(num)}')

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
        # print(f'thecher.shpe = {self.teacher_img.shape}')
        # scale_index = th/sh
        # print(f'scale2 = {scale_index} {sh} {th}')
        # 统一图像尺寸
        student_point_distance = count_distance(fix_point,free_point,self.student_results,self.student_img)
        teahcer_point_distance = count_distance(fix_point,free_point,self.teacher_results,self.teacher_img)

        if student_ratio == None:
            student_point_distance = student_point_distance / self.student_ratio
        else:
            student_point_distance = student_point_distance / student_ratio

        if teacher_ratio == None:
            teahcer_point_distance = teahcer_point_distance / self.teacher_ratio
        else:
            teahcer_point_distance = teahcer_point_distance / teacher_ratio

        result = percentage_error(student_point_distance,teahcer_point_distance)
        # print(count_weight(teahcer_point_distance))
        # print(student_point_distance,teahcer_point_distance)
        student_weight = count_weight(student_point_distance)
        teaher_weight = count_weight(teahcer_point_distance)

        return result
    
    def count(self,fix_point,free_point,student_ratio = None,teacher_ratio = None):
        teahcer_point_distance = count_distance(fix_point,free_point,self.teacher_results,self.teacher_img)
        weight = count_weight(teahcer_point_distance)
        return self.vector(fix_point,free_point)*(1-weight) + self.distance(fix_point,free_point,student_ratio,teacher_ratio)*weight

def count_distance(fix_point,free_point,results,img,ratio = 1):
    h, w, c = img.shape
    '''计算两点之间的距离'''
    results = results.pose_landmarks.landmark
    fix_point_x = results[fix_point].x*w
    fix_point_y = results[fix_point].y*h
    free_point_x = results[free_point].x*w
    free_point_y = results[free_point].y*h
    result = np.linalg.norm(np.array([fix_point_x,fix_point_y]) - np.array([free_point_x,free_point_y]))
    # 归一化尺寸
    result = result * 1000/h 
    return result
        


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



def percentage_error(a, b):
    def sigmoid(x, center, scale):
        return 1 / (1 + np.exp(-(x - center) / scale))
    def smooth_output(a, b):
        # Sigmoid function parameters
        center_high = 100
        center_low = 20
        scale = (center_high - center_low) / 4  # Scale factor to control smoothness
    
        # Calculate sigmoid values for a and b
        sigmoid_a = sigmoid(a, center_high, scale)
        sigmoid_b = sigmoid(b, center_high, scale)
    
        # Invert sigmoid values to get the desired behavior:
        # Close to 0 when a or b is greater than 100, close to 1 when both are less than 20
        inverted_sigmoid_a = 1 - sigmoid_a
        inverted_sigmoid_b = 1 - sigmoid_b
    
        # Combine sigmoid values (using multiplication to enforce both conditions)
        # and scale to the desired output range (0 to 1)
        combined = inverted_sigmoid_a * inverted_sigmoid_b
    
        # Since we want a smooth transition from 1 (both < 20) to 0 (both > 100),
        # we don't need any additional scaling if the sigmoid is properly tuned.
        # However, we can apply a final scaling if necessary.
    
        # Note: The output will never be exactly 0 or 1 due to the nature of the sigmoid function,
        # but it will be very close to these values when a and b are far from the center points.
        return combined


    if a == b:
        return 1
    else:
        ans = smooth_output(a, b) + (1- abs(a-b)/max(a,b))*(1-smooth_output(a, b))
        if ans > 1:
            return 1
        else :
            return ans

def count_weight(distance):
    # 根据教师图片关节点之间的距离计算权重,距离越大，权重越低
    distance = distance
    weight = 15 * 1/distance
    if weight > 1:
        weight = 1
    return weight

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