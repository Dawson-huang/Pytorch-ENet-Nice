import os 
import sys
import cv2
import numpy as np

def makepairs():
    train = open("train_cityscapes.txt", 'w')
    val = open("val_cityscapes.txt", 'w')

    dataname = ["train/", "val/"]
    for i in range(2):
      imagedirs = os.listdir("leftImg8bit/" + dataname[i])
      dir_num = len(imagedirs)
      for j in range(dir_num):
        images = os.listdir("leftImg8bit/" + dataname[i] + "/" + imagedirs[j])
        image_num = len(images)
        for k in range(image_num):
          image = images[k]
          lists = image.split('leftImg8bit.png')
          if i == 0:
            train.write("/Cityscapes/leftImg8bit/train/" + imagedirs[j] + "/" + image + " /Cityscapes/gtFine/train/" + imagedirs[j] + "/" + lists[0] + "gtFine_labelTrainIds.png\n")
          else:
            val.write("/Cityscapes/leftImg8bit/val/" + imagedirs[j] + "/" + image + " /Cityscapes/gtFine/val/" + imagedirs[j] + "/" + lists[0] + "gtFine_labelTrainIds.png\n")

if __name__ == '__main__':
    makepairs()

