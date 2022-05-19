import cv2
import math
import numpy as np
import json
import glob
from skimage import measure
import os
from delete_area import index_max_second
import matplotlib.pyplot as plt
class Point():
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def vector(self,other):
        #calculated the nother point#
        xx=self.x-other.x
        yy=self.y-other.y
        return Point(xx,yy)
    def cross(self,other):
        return (self.x*other.y-self.y*other.x)
    def dot(self,other):
        xx=self.x*other.x
        yy=self.y*other.y
        return (xx+yy)
    def mochang(self):#calculated the distance
        return (self.x**2+self.y**2)**0.5

class Point_line_2d():
    def __init__(self,x,y,a,b,A,B):
        self.x=x
        self.y=y
        self.a=a
        self.b=b
        self.A=A
        self.B=B
    def calculate(self):
        point=Point(self.x,self.y)
        point_in_line=Point(self.a,self.b)
        point_in_zero=Point(0,0)
        point_line_direction=Point(self.A,self.B)

        vector1=point.vector(point_in_line)
        vector2=point_in_zero.vector(point_line_direction)

        return math.fabs(vector1.cross(vector2))/(vector2.mochang()+0.000001)
def measure_rvot_ao_final(img_path):
    image = cv2.imread(img_path)
    # print('img tpye', image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bindary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bindary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 2:
        print("______________________________________fushi")
        kerner = np.ones((3,3), np.uint8)
        erosion = cv2.erode(image, kerner)
        index_ao, index_rvot = index_max_second(erosion)
        CD_distance, DE_distance, EF_distance = measure_rvot_ao(erosion, index_ao, index_rvot)
    else:
        kerner = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(image, kerner)
        index_ao, index_rvot = index_max_second(erosion)
        CD_distance, DE_distance, EF_distance = measure_rvot_ao(erosion, index_ao, index_rvot)
        # CD_distance, DE_distance, EF_distance = measure_rvot_ao(image, 0, 1)
    return CD_distance, DE_distance, EF_distance
def measure_rvot_ao_final1(img_path):
    image = cv2.imread(img_path)
    # print('img tpye', image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bindary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bindary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 2:

        CD_distance, DE_distance, EF_distance = measure_rvot_ao(image, 0, 1)
    else:

        CD_distance, DE_distance, EF_distance = measure_rvot_ao(image, 0, 1)
    return CD_distance, DE_distance, EF_distance

def measure_rvot_ao(img,index_ao,index_rvot):


    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('img_gray shape', img_gray.shape)
    _, bindary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bindary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('contours length', len(contours))
    y_ao_max = contours[index_ao][0][0]  # AO的y值最大值点
    print('contours', contours[index_ao][0][0][0])
    for i in range(len(contours[index_ao])):
        # print('i', i)
        # print('contours i', contours[0][i])
        if contours[index_ao][i][0][1] <= y_ao_max[1]:
            y_ao_max = contours[index_ao][i][0]
    print('Y AO max:', y_ao_max)
    Y_AO_MAX = tuple(y_ao_max)
    print('point ', np.array(Y_AO_MAX))

    length = 100000
    point_minlength_rvot = []  # 距离AO的Y值最大值点最近的点
    for j in range(len(contours[index_rvot])):

        point_rvot = np.array(contours[index_rvot][j][0])
        point = point_rvot - Y_AO_MAX
        length_AR = math.hypot(point[0], point[1])
        # print('length :', length)
        if length_AR <= length:
            length = length_AR
            point_minlength_rvot = contours[index_rvot][j][0]
    print('min length point', point_minlength_rvot)

    Y_rvot_min_Right = contours[index_rvot][0][0]  # 右侧y最小值点
    Y_rvot_min_Left = contours[index_rvot][0][0]  # 左侧最小值点
    X_rvot_max = contours[index_rvot][0][0]  # RVOTx的最大值点
    X_rvot_min = contours[index_rvot][0][0]  # RVOTx的最小值点
    Y_rvot_max = contours[index_rvot][0][0]
    point_A = contours[index_rvot][0][0]  # A点，两种情况，可能是左侧y值最小值点，也可能不是
    length_minao_yleft = 100000
    length_max_x = 0
    # length_pointY_list = []

    for x in range(len(contours[index_rvot])):
        if contours[index_rvot][x][0][0] >= X_rvot_max[0]:
            X_rvot_max = contours[index_rvot][x][0]
        if contours[index_rvot][x][0][0] <= X_rvot_min[0]:
            X_rvot_min = contours[index_rvot][x][0]
        if contours[index_rvot][x][0][1] <= Y_rvot_max[1]:
            Y_rvot_max = contours[index_rvot][x][0]


        if contours[index_rvot][x][0][1] >= Y_rvot_min_Left[1] and contours[index_rvot][x][0][0] <= \
                point_minlength_rvot[0]:
            Y_rvot_min_Left = contours[index_rvot][x][0]

    # y_min_right

    print('X_rvot_max', X_rvot_max)

    # 找A点
    for k in range(len(contours[index_ao])):
        point_ao = np.array(contours[index_ao][k][0])
        Y_rvot_min_Left = np.array(Y_rvot_min_Left)
        point = Y_rvot_min_Left - point_ao
        length_min_yleft_ao = math.hypot(point[0], point[1])
        if length_min_yleft_ao <= length_minao_yleft:
            length_minao_yleft = length_min_yleft_ao
        X_rvot_min = np.array(X_rvot_min)
        pointxy = X_rvot_min - Y_rvot_min_Left
        length_Xmax_ymin = math.hypot(pointxy[0], pointxy[1])
    print('length xy', length_Xmax_ymin)
    print('length miny_ao', length_minao_yleft)
    length_pointY_Y_max = 0
    length_point_rvot_right_ming = 0
    K_Y_left_min = (Y_rvot_min_Left[1] - point_minlength_rvot[1]) / (
                Y_rvot_min_Left[0] - point_minlength_rvot[0] + 1e-10)
    for d in range(len(contours[index_rvot])):


        if length_Xmax_ymin / (length_minao_yleft + 1e-10) >= 4 or length_Xmax_ymin == 0:
            point_A = Y_rvot_min_Left

        else:
            x1 = int(point_minlength_rvot[0])
            y1 = int(point_minlength_rvot[1])
            x2 = int(Y_rvot_min_Left[0] - x1)
            y2 = int(Y_rvot_min_Left[1] - y1)

            # X_max_rvot

            x3 = X_rvot_max[0]
            y3 = X_rvot_max[1]

            if contours[index_rvot][d][0][0] >= Y_rvot_min_Left[0] and contours[index_rvot][d][0][0] <= \
                    point_minlength_rvot[0] \
                    and contours[index_rvot][d][0][1] >= point_minlength_rvot[1]:

                x = int(contours[index_rvot][d][0][0])
                y = int(contours[index_rvot][d][0][1])
                K_A = (y - point_minlength_rvot[1]) / (x - point_minlength_rvot[0] + 1e-10)

                length_p_line = Point_line_2d(x, y, x1, y1, x2, y2)
                distance = length_p_line.calculate()


                if distance >= int(length_pointY_Y_max) and abs(K_A) >= abs(K_Y_left_min):
                    length_pointY_Y_max = distance
                    point_A = contours[index_rvot][d][0]
    for idx_minleft in range(len(contours[index_rvot])):
        x1 = point_minlength_rvot[0]
        y1 = point_minlength_rvot[1]

        # X_max_rvot

        x3 = X_rvot_max[0] - x1
        y3 = X_rvot_max[1] - y1
        if contours[index_rvot][idx_minleft][0][0] >= point_minlength_rvot[0] and contours[index_rvot][idx_minleft][0][
            1] >= X_rvot_max[1]:

            x_right = contours[index_rvot][idx_minleft][0][0]
            y_right = contours[index_rvot][idx_minleft][0][1]
            print('y_right', y_right)
            length_p_line_rvot = Point_line_2d(x_right, y_right, x1, y1, x3, y3)
            distance_rvot = length_p_line_rvot.calculate()


            if distance_rvot >= length_point_rvot_right_ming:
                length_point_rvot_right_ming = distance_rvot
                Y_rvot_min_Right = contours[index_rvot][idx_minleft][0]

    # AO最近的点
    point_AO_A = []
    point_AO_B = []
    length_A_min = 1000000
    length_B_min = 1000000
    for idx in range(len(contours[index_ao])):
        point_A = np.array(point_A)
        Y_rvot_min_Right = np.array(Y_rvot_min_Right)
        point_AO_idx = np.array(contours[index_ao][idx][0])
        point_vectorA = point_A - point_AO_idx
        length_A = math.hypot(point_vectorA[0], point_vectorA[1])
        # print("length A", length_A)
        if length_A <= length_A_min:
            length_A_min = length_A
            point_AO_A = contours[index_ao][idx][0]
        point_vectorB = Y_rvot_min_Right - point_AO_idx
        length_B = math.hypot(point_vectorB[0], point_vectorB[1])
        if length_B <= length_B_min:
            length_B_min = length_B
            point_AO_B = contours[index_ao][idx][0]
    length_B_min_rvot = 1000000
    point_B_RVOT_final = []
    for index_B in range(len(contours[index_rvot])):

        point_rvot_b = np.array(contours[index_rvot][index_B][0])
        point = point_rvot_b - point_AO_B
        length_B_rvot = math.hypot(point[0], point[1])
        # print('length :', length)
        if length_B_rvot <= length_B_min_rvot:
            length_B_min_rvot = length_B_rvot
            point_B_RVOT_final = contours[index_rvot][index_B][0]

    print('AA', point_A)
    print("AO a", point_AO_A)
    print("AO b", point_AO_B)
    if point_AO_A[1] == point_AO_B[1]:
        point_AO_B[1] = point_AO_B[1] + 2
    print("AO b", point_AO_B)
    point_mid_AO = [int((point_AO_A[0] + point_AO_B[0]) / 2), int((point_AO_A[1] + point_AO_B[1]) / 2)]

    k_AB = (point_AO_A[1] - point_AO_B[1]) / (point_AO_A[0] - point_AO_B[0] + 1e-10)  # AB斜率, +1e-10 防止➗0
    k_value_min = 10000000
    k_value_max = 10000000
    k_value_min_rvot = 10000000
    k_value_max_rvot = 10000000
    point_AO_min_final = []
    point_AO_max_final = []
    point_rvot_min_final = []
    point_rvot_max_final = []
    K_vertical = (point_AO_B[0] - point_AO_A[0]) / (point_AO_A[1] - point_AO_B[1] + 1e-10)  # 与AB垂直的斜率

    print('斜率乘积：', k_AB * K_vertical)

    # 求直线
    b = point_mid_AO[1] - K_vertical * point_mid_AO[0]

    print('直线b', b)
    #
    for id in range(len(contours[index_ao])):
        if abs(contours[index_ao][id][0][1] - K_vertical * contours[index_ao][id][0][0] - b) <= k_value_min \
                and contours[index_ao][id][0][1] <= point_mid_AO[1]:
            k_value_min = abs(contours[index_ao][id][0][1] - K_vertical * contours[index_ao][id][0][0] - b)
            point_AO_min_final = contours[index_ao][id][0]
        if abs(contours[index_ao][id][0][1] - K_vertical * contours[index_ao][id][0][0] - b) <= k_value_max \
                and contours[index_ao][id][0][1] >= point_mid_AO[1]:
            k_value_max = abs(contours[index_ao][id][0][1] - K_vertical * contours[index_ao][id][0][0] - b)
            point_AO_max_final = contours[index_ao][id][0]
    RVOT_mid = [int((Y_rvot_max[0] + point_minlength_rvot[0]) / 2),
                int(3 * (point_minlength_rvot[1] + Y_rvot_max[1]) / 5)]
    for idrvot in range(len(contours[index_rvot])):


        if abs(contours[index_rvot][idrvot][0][1] - K_vertical * contours[index_rvot][idrvot][0][
            0] - b) <= k_value_max_rvot \
                and contours[index_rvot][idrvot][0][1] >= RVOT_mid[1]:
            k_value_max_rvot = abs(
                contours[index_rvot][idrvot][0][1] - K_vertical * contours[index_rvot][idrvot][0][0] - b)
            point_rvot_max_final = contours[index_rvot][idrvot][0]

    # k xiangjin
    # K_vertical
    K_rvot_value = 100000
    for k_rvot in range(len(contours[index_rvot])):
        k_rvot_final = (point_rvot_max_final[1] - contours[index_rvot][k_rvot][0][1]) / (
                    point_rvot_max_final[0] - contours[index_rvot][k_rvot][0][0] + 1e-10)
        print('k_rvot_final', k_rvot_final)
        if abs(K_vertical / (k_rvot_final + 1e-10) - 1) <= K_rvot_value :
            K_rvot_value = abs(K_vertical / (k_rvot_final + 1e-10) - 1)
            point_rvot_min_final = contours[index_rvot][k_rvot][0]

    point_CD = np.array(point_AO_max_final) - np.array(point_AO_min_final)
    point_DE = np.array(point_rvot_max_final) - np.array(point_rvot_min_final)

    point_EF = np.array(X_rvot_max) - np.array(point_B_RVOT_final)
    print('point_point_AO_max_final', point_AO_max_final)
    print('point_cd', point_CD)
    CD_distance = math.hypot(point_CD[0], point_CD[1])
    DE_distance = math.hypot(point_DE[0], point_DE[1])
    EF_distance = math.hypot(point_EF[0], point_EF[1])
    print("CD distance:", CD_distance)
    print("DE distance:", DE_distance)
    print("EF distance:", EF_distance)

    if CD_distance / (EF_distance + 1e-10) >= 2:
        Y_rvot_min_Right = point_B_RVOT_final
        point_EF = np.array(X_rvot_max) - np.array(Y_rvot_min_Right)
        EF_distance = math.hypot(point_EF[0], point_EF[1])
    print("EF distance:", EF_distance)

    return CD_distance, DE_distance, EF_distance






