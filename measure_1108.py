import cv2
import math
import numpy as np
import json
import glob
from skimage import measure
import os
from measure_1117 import measure_rvot_ao, measure_rvot_ao_final, measure_rvot_ao_final1
from delete_area import index_max_second
from error import pred_erode, pred_erode_31

def line_segment_length(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        json_p1 = json_data['shapes'][0]['points'][0]
        json_p2 = json_data['shapes'][0]['points'][1]
        json_p1 = np.array(json_p1)
        json_p2 = np.array(json_p2)
        json_p = json_p2 - json_p1
        line_length = math.hypot(json_p[0], json_p[1])
    return line_length
class Point():
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def vector(self,other):
        xx=self.x-other.x
        yy=self.y-other.y
        return Point(xx,yy)
    def cross(self,other):
        return (self.x*other.y-self.y*other.x)
    def dot(self,other):
        xx=self.x*other.x
        yy=self.y*other.y
        return (xx+yy)
    def mochang(self):
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

        return math.fabs(vector1.cross(vector2))/vector2.mochang()
def measure_ao_rvot(img_path):

    img = cv2.imread(img_path)
    print(img.shape)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print('img_gray shape', img_gray.shape)
    _, bindary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bindary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = 100000
    y_ao_max = contours[0][0][0]  # AO的y值最大值点
    point_minlength_rvot = []  # 距离AO的Y值最大值点最近的点
    Y_rvot_min_Right = contours[1][0][0]  # 右侧y最小值点
    Y_rvot_min_Left = contours[1][0][0]  # 左侧最小值点
    X_rvot_max = contours[1][0][0]  # RVOTx的最大值点
    X_rvot_min = contours[1][0][0]  # RVOTx的最小值点
    Y_rvot_max = contours[1][0][0]
    point_A = contours[1][0][0]  # A点，两种情况，可能是左侧y值最小值点，也可能不是
    length_minao_yleft = 100000
    length_max_x = 0
    length_pointY_Y_max = 0
    point_AO_A = []
    point_AO_B = []
    length_A_min = 1000000
    length_B_min = 1000000
    k_value_min = 10000000
    k_value_max = 10000000
    k_value_min_rvot = 10000000
    k_value_max_rvot = 10000000
    point_AO_min_final = []
    point_AO_max_final = []
    point_rvot_min_final = []
    point_rvot_max_final = []
    #print('contours', contours[0][0][0][0])
    for i in range(len(contours[0])):
        # print('i', i)
        # print('contours i', contours[0][i])
        if contours[0][i][0][1] <= y_ao_max[1]:
            y_ao_max = contours[0][i][0]
    # print('Y AO max:', y_ao_max)
    Y_AO_MAX = tuple(y_ao_max)
    # print('point ', np.array(Y_AO_MAX))

    for j in range(len(contours[1])):
        # print('contours j', contours[1][j][0])
        # point_AO_y_max = np.array(Y_AO_MAX)
        point_rvot = np.array(contours[1][j][0])
        point = point_rvot - Y_AO_MAX
        length_AR = math.hypot(point[0], point[1])
        # print('length :', length)
        if length_AR <= length:
            length = length_AR
            point_minlength_rvot = contours[1][j][0]
    #print('min length point', point_minlength_rvot)
    for x in range(len(contours[1])):
        if contours[1][x][0][0] >= X_rvot_max[0]:
            X_rvot_max = contours[1][x][0]
        if contours[1][x][0][0] <= X_rvot_min[0]:
            X_rvot_min = contours[1][x][0]
        if contours[1][x][0][1] >= Y_rvot_max[0]:
            Y_rvot_max = contours[1][x][0]
        if contours[1][x][0][1] >= Y_rvot_min_Right[1] and contours[1][x][0][0] >= point_minlength_rvot[0]:
            Y_rvot_min_Right = contours[1][x][0]

        if contours[1][x][0][1] >= Y_rvot_min_Left[1] and contours[1][x][0][0] <= point_minlength_rvot[0]:
            Y_rvot_min_Left = contours[1][x][0]
        # 找A点
    for k in range(len(contours[0])):
        point_ao = np.array(contours[0][k][0])
        Y_rvot_min_Left = np.array(Y_rvot_min_Left)
        point = Y_rvot_min_Left - point_ao
        length_min_yleft_ao = math.hypot(point[0], point[1])
        if length_min_yleft_ao <= length_minao_yleft:
            length_minao_yleft = length_min_yleft_ao
        X_rvot_min = np.array(X_rvot_min)
        pointxy = X_rvot_min - Y_rvot_min_Left
        length_Xmax_ymin = math.hypot(pointxy[0], pointxy[1])
    K_Y_left_min = (Y_rvot_min_Left[1] - point_minlength_rvot[1]) / (Y_rvot_min_Left[0] - point_minlength_rvot[0] + 1e-10)

    for d in range(len(contours[1])):
        if length_Xmax_ymin / (length_minao_yleft+1e-10) >= 4:
            point_A = Y_rvot_min_Left

        else:
            x1 = point_minlength_rvot[0]
            y1 = point_minlength_rvot[1]
            x2 = Y_rvot_min_Left[0] - x1
            y2 = Y_rvot_min_Left[1] - y1

            if contours[1][d][0][0] >= Y_rvot_min_Left[0] and contours[1][d][0][0] <= point_minlength_rvot[0] \
                    and contours[1][d][0][1] >= point_minlength_rvot[1]:
                x = contours[1][d][0][0]
                y = contours[1][d][0][1]
                K_A = (y - point_minlength_rvot[1]) / (x - point_minlength_rvot[0] + 1e-10)

                length_p_line = Point_line_2d(x, y, x1, y1, x2, y2)
                distance = length_p_line.calculate()

                if distance >= length_pointY_Y_max and abs(K_A) >= abs(K_Y_left_min):
                    length_pointY_Y_max = distance
                    point_A = contours[1][d][0]
    for idx in range(len(contours[0])):
        point_A = np.array(point_A)
        Y_rvot_min_Right = np.array(Y_rvot_min_Right)
        point_AO_idx = np.array(contours[0][idx][0])
        point_vectorA = point_A - point_AO_idx
        length_A = math.hypot(point_vectorA[0], point_vectorA[1])
        # print("length A", length_A)
        if length_A <= length_A_min:
            length_A_min = length_A
            point_AO_A = contours[0][idx][0]
        point_vectorB = Y_rvot_min_Right - point_AO_idx
        length_B = math.hypot(point_vectorB[0], point_vectorB[1])
        if length_B <= length_B_min:
            length_B_min = length_B
            point_AO_B = contours[0][idx][0]
    point_mid_AO = [int((point_AO_A[0] + point_AO_B[0]) / 2), int((point_AO_A[1] + point_AO_B[1]) / 2)]

    #k_AB = (point_AO_A[1] - point_AO_B[1]) / (point_AO_A[0] - point_AO_B[0] + 1e-10)  # AB斜率, +1e-10 防止➗0
    K_vertical = (point_AO_B[0] - point_AO_A[0]) / (point_AO_A[1] - point_AO_B[1] + 1e-10)  # 与AB垂直的斜率
    b = point_mid_AO[1] - K_vertical * point_mid_AO[0]
    for id in range(len(contours[0])):
        if abs(contours[0][id][0][1] - K_vertical * contours[0][id][0][0] - b) <= k_value_min \
                and contours[0][id][0][1] <= point_mid_AO[1]:
            k_value_min = abs(contours[0][id][0][1] - K_vertical * contours[0][id][0][0] - b)
            point_AO_min_final = contours[0][id][0]
        if abs(contours[0][id][0][1] - K_vertical * contours[0][id][0][0] - b) <= k_value_max \
                and contours[0][id][0][1] >= point_mid_AO[1]:
            k_value_max = abs(contours[0][id][0][1] - K_vertical * contours[0][id][0][0] - b)
            point_AO_max_final = contours[0][id][0]
    RVOT_mid = [(Y_rvot_max[0] + point_minlength_rvot[0]) / 2, (Y_rvot_max[1] + point_minlength_rvot[1]) / 2]
    for idrvot in range(len(contours[1])):
        if abs(contours[1][idrvot][0][1] - K_vertical * contours[1][idrvot][0][0] - b) <= k_value_min_rvot \
                and contours[1][idrvot][0][1] <= RVOT_mid[1]:
            k_value_min_rvot = abs(contours[1][idrvot][0][1] - K_vertical * contours[1][idrvot][0][0] - b)
            point_rvot_min_final = contours[1][idrvot][0]
        if abs(contours[1][idrvot][0][1] - K_vertical * contours[1][idrvot][0][0] - b) <= k_value_max_rvot \
                and contours[1][idrvot][0][1] >= RVOT_mid[1]:
            k_value_max_rvot = abs(contours[1][idrvot][0][1] - K_vertical * contours[1][idrvot][0][0] - b)
            point_rvot_max_final = contours[1][idrvot][0]
    point_CD = np.array(point_AO_max_final) - np.array(point_AO_min_final)
    point_DE = np.array(point_rvot_max_final) - np.array(point_rvot_min_final)
    point_EF = np.array(X_rvot_max) - np.array(Y_rvot_min_Right)
    CD_distance = math.hypot(point_CD[0], point_CD[1])
    DE_distance = math.hypot(point_DE[0], point_DE[1])
    EF_distance = math.hypot(point_EF[0], point_EF[1])
    return CD_distance, DE_distance, EF_distance

datadir_0 = "/home/guolibao/cardiac-jbhi/cardiac-4ch/cardiac-rvot"
datadir = "/home/guolibao/train_addition/1030_test_rvot_pyt"
# path = "/home/guolibao/cardiac-jbhi/cardiac-4ch/cardiac-rvot/result_all/result_all_com/33_2/"
path ="/home/guolibao/train_addition/Ablation_ddnet/Ablation_result/ablation6/result_all/"
###################################################################################text label
datadir_1 = "//home/guolibao/train_addition/Ablation_ddnet/Ablation_result/ablation6/"
txt_fname = "/home/guolibao/cardiac-jbhi/cardiac-4ch/cardiac-rvot/dic/val_shan.txt"
with open(txt_fname, 'r')as f:
    images = f.read().split()
img_rvot_labels = [os.path.join(datadir, 'val_label', i) for i in images]
img_text_labels = [os.path.join(datadir_1, 'bin4', i) for i in images]
# json_fname = '/home/guolibao/cardiac-jbhi/cardiac-4ch/cardiac-rvot/dic/val_juli_256_shan_quan.txt'
# with open(json_fname, 'r')as f:
#     json_name = f.read().split()
line_segment = [os.path.join(datadir_0, 'val_juli_256', i) for i in images]
##########################################################################################xiugai
result_label_txt=path+'label_33_2'+'.txt'
label_txt=open(result_label_txt,'w')
result_label_txt1=path+'label_text_ablation6'+'.txt'################################################gaizheli
label_txt1=open(result_label_txt1,'w')
for name_lv,name_text,name_line in zip(img_rvot_labels,img_text_labels,line_segment):
    print(name_lv)
    print(name_text)
    print(name_line.replace('.png', '.json'))
    name_line = name_line.replace('.png', '.json')

    CD_D,DE_D,EF_D = measure_rvot_ao_final1(name_lv)

    CD_D1, DE_D1, EF_D1 = measure_rvot_ao_final(name_text)

    with open(name_line, 'r') as f:
        json_data = json.load(f)
        json_p1 = json_data['shapes'][0]['points'][0]
        json_p2 = json_data['shapes'][0]['points'][1]
        json_p1=np.array(json_p1)
        json_p2=np.array(json_p2)
        json_p=json_p2-json_p1
        line_length=math.hypot(json_p[0],json_p[1])
    CD_distance = (5 * CD_D) / line_length
    DE_distance = (5 * DE_D) / line_length
    EF_distance = (5 * EF_D) / line_length
    CD_distance1 =  ((5 * CD_D1) / line_length)
    DE_distance1 = ((5 * DE_D1) / line_length)
    EF_distance1 = ((5 * EF_D1) / line_length)

    label_num = name_lv+','+str(CD_distance)+','+str(DE_distance)+','+str(EF_distance)+'\n'
    label_num1 = name_text + ',' + str(CD_distance1) + ',' + str(DE_distance1) + ',' + str(EF_distance1) + '\n'

    label_txt.write(label_num)
    label_txt1.write(label_num1)
    print('CD distance', CD_distance)
    print('DE distance', DE_distance)
    print('EF distance', EF_distance)
    print('CD distance1', CD_distance1)
    print('DE distance1', DE_distance1)
    print('EF distance1', EF_distance1)
label_txt.close()
label_txt1.close()
#


