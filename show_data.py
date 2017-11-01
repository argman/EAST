# coding:utf-8
import os
import glob
import numpy as np
from PIL import Image
import codecs
# import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# name = input("What is your name? ")
dataPath = "/home/neo/Dataset/ICDAR2015/TextLocalization/TextLocalization_test_images"

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train')

image_list = glob.glob(os.path.join(DATA_PATH, '*', 'image*.jpg'))


def show_all():
    for image in image_list:
        label_file_name = image[:-4] + '.txt'
        im = Image.open(image)
        im = np.array(im).astype(np.uint8)
        plt.imshow(im)
        labels = codecs.open(label_file_name, 'r', 'utf8').readlines()

        for label in labels:
            label = label.strip().split(',')
            LL = label[:-2]
            location = [int(i) for i in LL]
            context = label[-1].strip('"')
            # print(location)
            plt.plot([location[0], location[2], location[4], location[6], location[0]], [
                     location[1], location[3], location[5], location[7], location[1]], 'r-')
            # ,bbox={'facecolor':'white', 'alpha':0.5}
            plt.text(location[0], location[1], context, fontsize=8, color='#33FF36')
        plt.show()
        name = input("Are you continue? ")
        if name == 'n' or name == 'no':
            break


def show_num(num):
    img_name = 'image_' + str(num) + '.jpg'
    for image in image_list:
        if img_name in image:
            label_file_name = image[:-4] + '.txt'
            im = Image.open(image)
            im = np.array(im).astype(np.uint8)
            plt.imshow(im)
            labels = open(label_file_name, 'r', encoding='utf8').readlines()

            for label in labels:
                label = label.strip().split(',')
                LL = label[:-2]
                location = [int(i) for i in LL]
                context = label[-1].strip('"')
                # print(location)
                plt.plot([location[0], location[2], location[4], location[6], location[0]], [
                         location[1], location[3], location[5], location[7], location[1]], 'r-')
                plt.plot(location[0], location[1], 'b*')
                # ,bbox={'facecolor':'white', 'alpha':0.5}
                plt.text(location[0], location[1], context, fontsize=8, color='#33FF36')
            plt.show()


if __name__ == '__main__':
    n = int(input("num"))
    show_num(n)
# print(image_list[0:2],'\n',label_file_name)
