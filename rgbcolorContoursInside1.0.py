# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:30:06 2019

@author: 郝蛤蛤
"""
import numpy as np
import cv2 as cv
from PIL import Image
import pandas as pd
import csv
import sys
import operator
import os
from functools import reduce
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

samplenum = int(sys.argv[1])
concentration = sys.argv[2]
image_file = sys.argv[3]


def imgcrop(crop_num, file):  # 将图像分割成几等分
    # crop_num = 4
    # file = "imgcrop.png"
    img = Image.open(file)
    img_width = img.size[0]
    img_heigh = img.size[1]
    croped_img_width = img_width / crop_num
    for i in range(0, crop_num):
        image = img.crop((croped_img_width * i, 0, croped_img_width * (i + 1), img_heigh))
        image.save("imgcrop_no" + str(i) + ".png")


def imgconcatenate(concatenate_num):  # 将图像拼接起来
    concatenate_list = []
    for i in range(0, concatenate_num):
        exec("img" + str(i) + "= cv.imread('imgcrop" + str(i) + ".png')")
        exec("concatenate" + str(i) + "= cv.cvtColor(img" + str(i) + ", cv.COLOR_BGR2GRAY)")
        exec("concatenate_list.append(" + str("concatenate") + str(i) + ")")
    concatenate = np.concatenate(concatenate_list, axis=1)
    return concatenate


def computeRGB(inputimagefile, filename):#计算RGB
    img = io.imread(inputimagefile + ".tif")  # 在此处定义图片的名称
    img = np.array(img)  # 将图片转换成好处理的格式。
    # print(img.shape)
    # 读取图片的size
    datax = pd.read_csv(filename + ".csv", usecols=[0])  # 读取csv中第一列的数据，并转化为二维数组
    datay = pd.read_csv(filename + ".csv", usecols=[1])

    listx = datax.values.tolist()
    listy = datay.values.tolist()
    X = reduce(operator.add, listx)  # 将二维数组将至一维数组
    Y = reduce(operator.add, listy)

    re_lst = []  # 用于暂时储存结果的空间

    # 字符集
    charsetx = np.array(X)  # 要非常注意，对应读取X1,Y1  X2,Y2   X3,Y3   X4,Y4
    charsety = np.array(Y)  # 要非常注意，此处的y轴，是酶标板的长轴，有12个数据，建立y轴的值的索引库 在此修改y轴的长度
    # 读取数据的顺序是，从左上角第一列开始读，然后读第二列，第三列、第四列
    for index in range(len(charsetx)):  # 定义index的长度为charsetx的个数
        i = index
        j = index
        buf = np.array([0, 0, 0])
        numberx = 0
        numbery = 0
        axis_x = charsetx[i] - numberx
        axis_y = charsety[j] - numbery
        for inner_x in range(numberx * 2 + 1):
            for inner_y in range(numbery * 2 + 1):
                buf = buf + img[axis_x + inner_x][axis_y + inner_y]
        # 输出的数据是每个选定点为中心的一个正方形的平均rgb
        buf = buf / ((numberx * 2 + 1) * (numbery * 2 + 1))
        # print(buf) 用来检查数据的
        re_lst.append(str(buf[0]) + "," + str(buf[1]) + "," + str(buf[2]))

    # csvFile = open("POI.csv", "w", encoding='utf8', newline='')  # 创建csv文件
    csvFile = open('RGB' + str(filename) + '.csv', 'w', newline='')
    doc = csv.writer(csvFile)  # 创建写的对象
    doc.writerow(["R", "G", "B"])
    # 输出临时保存的结果到最终文本，   '\n'是换行符，即系每输入完一行就回车换行
    for i in range(len(re_lst)):
        csvFile.write(re_lst[i] + '\n')


def brightness_compute(filename):#计算亮度
    data = pd.read_csv(filename + '.csv')
    meanR = data['R'].mean()
    meanG = data['G'].mean()
    meanB = data['B'].mean()
    brightness = 0.3 * meanR + 0.6 * meanG + 0.1 * meanB
    return brightness


def cluster(x, k):#聚类算法
    cluster_result = KMeans(n_clusters=k, random_state=0).fit(x)
    cluster_label = cluster_result.labels_ #标签
    cluster_centers = cluster_result.cluster_centers_ #质心

    return cluster_label


def createcsvfile():#生成轮廓的XY坐标文件，并计算生成RGB文件
    location = []
    brightness_list = []  # 亮度表
    label = cluster(cluster_point, samplenum)

    for j in range(0, samplenum):
        with open('coordinates' + str(j) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Y', 'X'])
            csvfile.close()

    for point_0 in cluster_point:#录入所有的点
        point_0_label = cluster_point.index(point_0)
        with open('coordinates' + str(label[point_0_label]) + '.csv', 'a+', newline='') as csvfilecopy:
            writercopy = csv.writer(csvfilecopy)
            writercopy.writerow(point_0)
            csvfilecopy.close()

    for count in range(0, samplenum):#给csv文件重新命名，从左到右排列
        df = pd.read_csv('coordinates' + str(count) + '.csv')
        name = int(df['X'].mean())
        location.append(name)
        df.to_csv(str(name) + '.csv', index=False)
        os.remove("coordinates" + str(count) + ".csv")

        computeRGB("imgcrop", str(name))
        # brightness_list.append(brightness_compute("RGBcoordinates"))
    location.sort()
    for RGBname in location:
        brightness_list.append(brightness_compute('RGB' + str(RGBname)))
    return brightness_list


if __name__ == '__main__':
    # imgcrop("test1.jpg",836, 658, 1382, 805)
    concentration_read = open(concentration, "r")#读取初始化坐标文件
    concentration_list = concentration_read.read().split(',')
    concentration_read.close()

    src = cv.imread(image_file)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) #转换为灰度图
    cluster_point = [] #轮廓内的所有坐标点，用于做聚类算法分类
    # dst = np.zeros((height,width,1), np.uint8)#颜色反转
    # for i in range(0, height):
    #     for j in range(0, width):
    #         grayPixel = gray[i, j]
    #         dst[i, j] = 255-grayPixel

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
    dst = cv.filter2D(gray, -1, kernel=kernel) #图像锐化
    blurred_noneEqual = cv.GaussianBlur(dst, (9, 9), 0) #高斯模糊去噪声
    blurred_noneEqual = cv.GaussianBlur(blurred_noneEqual, (9, 9), 0)
    blurred_noneEqual = cv.GaussianBlur(blurred_noneEqual, (9, 9), 0)
    cv.imwrite("blurred_noneEqual.png", blurred_noneEqual)
    # blurred_noneEqual = cv.blur(gray, (1, 3))
    imgcrop(samplenum, "blurred_noneEqual.png")  # 分割图像为N等分
    for read_img in range(0, samplenum):
        exec("src" + str(read_img) + "= cv.imread('imgcrop_no" + str(read_img) + ".png')")
    for normalize_num in range(0, samplenum):
        # 批量化图像增强
        exec("blurred" + str(normalize_num) + "= cv.normalize(src" + str(normalize_num) +
             ", dst=None, alpha=350, beta=10, norm_type=cv.NORM_MINMAX)")
        # cv.imwrite("imgcrop" + str(normalize_num) + ".png", blurred1)
        exec("cv.imwrite('imgcrop" + str(normalize_num) + ".png', blurred" + str(normalize_num) + ")")
        exec("os.remove('imgcrop_no" + str(normalize_num) + ".png')")
    # blurred = cv.normalize(blurred_noneEqual, dst=None, alpha=350, beta=10, norm_type=cv.NORM_MINMAX)#直方图正规化图像增强
    # ret, thresh = cv.threshold(blurred, 157, 255, cv.THRESH_BINARY_INV) #图像二值化，这里的第二个参数可设置深度
    blurred = imgconcatenate(samplenum)  # 图像拼接
    cv.imwrite("blurred.png", blurred)
    os.remove("blurred_noneEqual.png")
    color = []
    for colornum in range(samplenum + 1):
        colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        color_0 = ""
        for i in range(6):
            color_0 += colorArr[random.randint(0, 14)]  # 随机生成颜色
        color.append('#' + color_0)

    # color = ['red', 'pink', 'orange', 'gray', 'blue']
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121)

    ax1.set_xlabel('ug')
    ax1.set_ylabel('Brightness')
    ax1.set_title('Outline_Inside')

    # for para in [200, 190, 180, 170, 160, 150, 140, 130, 120, 110]:
    for para in range(130, 210, 5):
        ret, thresh = cv.threshold(blurred, para, 255, cv.THRESH_BINARY) #图像二值化，这里的第二个参数可设置深度
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #画出轮廓
        height, width = thresh.shape #获取轮廓的高度与宽度
        for withe_x in range(height):
            for withe_y in range(width):
                if thresh[withe_x, withe_y] == 255:
                    cluster_point.append([withe_x, withe_y])#收录轮廓内坐标

        for num in range(-1, len(contours)):
            draw_img = cv.drawContours(src.copy(), contours, num, (0, 0, 255), 1)
            cv.imwrite("Contours" + str(num) + ".jpg", draw_img)  # 生成画出目标框的图片
        # for i in range(0, len(contours)):
        #     x, y, w, h = cv.boundingRect(contours[i])
        #     draw_img2 = cv.rectangle(src, (x,y), (x+w,y+h), (153,153,0), 5) #画出矩形轮廓

        cv.imwrite("imgcrop.tif", src)#生成裁剪出的tif格式照片
        try:
            brightness = createcsvfile()
        except IndexError:
            print("目标检测数量没有达到要求，正在调整参数为：" + str(para))
            continue
        else:
            if len(brightness) == samplenum:
                ax1.plot(concentration_list, brightness, color='r', label='Gelquant', marker='o', markersize=6,
                         markeredgecolor='black', markerfacecolor='brown')
                break

    print("正在生成聚类结果图")
    ax2 = plt.subplot(122)
    ax2.set_xlabel('XLabel')
    ax2.set_ylabel('YLabel')
    ax2.set_title('Cluster')
    label = cluster(cluster_point, samplenum)
    for point in cluster_point:
        point_label = cluster_point.index(point)
        ax2.scatter(point[1], point[0], marker='o', s=8, c=color[label[point_label]])#画出聚类算法计算后分类的点

    # cv.imwrite("Contours.jpg", draw_img)#生成画出目标框的图片
    # cv.imshow("input image", draw_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # for filenum in range(0, 51):
    #     try:
    #         os.remove("Contours" + str(filenum) + ".jpg")
    #         os.remove("coordinates" + str(filenum) + ".csv")
    #         os.remove("RGBcoordinates" + str(filenum) + ".csv")
    #     except FileNotFoundError:
    #         print("正在删除多余的文件....")

    print(brightness)

    brightness_concentration = dict(zip(concentration_list, brightness))
    with open("ex1data3.txt", "w") as txtfile:
        for item in brightness_concentration.items():
            print(item)
            item = str(item).replace('(', '')
            item = str(item).replace(')', '')
            item = str(item).replace("'", '')
            txtfile.write(item + '\n')
    plt.savefig("全局平均.jpg")
    plt.show()
