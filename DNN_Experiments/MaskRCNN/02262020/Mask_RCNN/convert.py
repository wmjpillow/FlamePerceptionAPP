#!/usr/bin/env python
# convert jpg tp png
from glob import glob
import cv2
pngs = glob('./*.jpg')

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'png', img)


# delete jpg files
import glob
import os

dir = "/Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/Models/MaskRCNN/02_26_2020/Mask_RCNN/dataset/train"

for jpgpath in glob.iglob(os.path.join(dir, '*.jpg')):
    os.remove(jpgpath)