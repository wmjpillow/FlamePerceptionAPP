import cv2
import numpy as np
import sys
# import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
import csv
# import pandas as pd
import math
# get_ipython().magic(u'matplotlib inline')
# REFERENCE: https://github.com/benjamincastillo2020/FireDetectionCode
# https://stackoverflow.com/questions/22704936/reading-every-nth-frame-from-videocapture-in-opencv
cap1 = cv2.VideoCapture('../Videos/CandleFire.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
count = 0

ID = 0
while 1:
        ret, frame = cap1.read()  # reading the image
        if count%10==0:
        # flame shape area
            gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gs_neg = 255 - gs.astype(int)
            gs_neg[gs_neg > 50] = 1000
            gs_neg = gs_neg.astype('uint8')
            show_imagename = ['negative']
            show_image = [gs_neg]
            im   = Image.fromarray(gs_neg).convert('RGBA').convert('RGB')
            imnp = np.array(im)
        # https://stackoverflow.com/questions/11433604/opencv-setting-all-pixels-of-specific-bgr-value-to-another-bgr-value
            imnp[np.where((imnp == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
            imnp[np.where((imnp == [232, 232, 232]).all(axis=2))] = [0, 0, 0]
            # imnp[np.where((imnp == [0, 0, 0]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [1, 1, 1]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [2, 2, 2]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [4, 4, 4]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [5, 5, 5]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [6, 6, 6]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [7, 7, 7]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [8, 8, 8]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [9, 9, 9]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [10, 10, 10]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [11, 11, 11]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [12, 12, 12]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [13, 13, 13]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [14, 14, 14]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [15, 15, 15]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [16, 16, 16]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [17, 17, 17]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [18, 18, 18]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [19, 19, 19]).all(axis=2))] = [0, 0, 255]
            imnp[np.where((imnp == [20, 20, 20]).all(axis=2))] = [0, 0, 255]
        #     imnp[np.where((imnp == [15, 15, 15]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [16, 16, 16]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [17, 17, 17]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [18, 18, 18]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [19, 19, 19]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [20, 20, 20]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [21, 21, 21]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [22, 22, 22]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [23, 23, 23]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [24, 24, 24]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [25, 25, 25]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [26, 26, 26]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [27, 27, 27]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [28, 28, 28]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [29, 29, 29]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [30, 30, 30]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [31, 31, 31]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [32, 32, 32]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [33, 33, 33]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [34, 34, 34]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [35, 35, 35]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [36, 36, 36]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [37, 37, 37]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [38, 38, 38]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [39, 39, 39]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [40, 40, 40]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [41, 41, 41]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [42, 42, 42]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [43, 43, 43]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [44, 44, 44]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [45, 45, 45]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [46, 46, 46]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [47, 47, 47]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [48, 48, 48]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [49, 49, 49]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [50, 50, 50]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [51, 51, 51]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [52, 52, 52]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [53, 53, 53]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [54, 54, 54]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [55, 55, 55]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [56, 56, 56]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [57, 57, 57]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [58, 58, 58]).all(axis=2))] = [255, 255, 255]
        #     imnp[np.where((imnp == [59, 59, 59]).all(axis=2))] = [255, 255, 255]

            print(imnp)
            h, w = imnp.shape[:2]
            colours, counts = np.unique(imnp.reshape(-1,3), axis=0, return_counts=1)
            SumCount=0
            SumProportion=0
            # print(colours)
            for index, colour in enumerate(colours):
               count = counts[index]
               proportion = (100 * count) / (h * w)
               # print(f"   Colour: {colour}, count: {count}, proportion: {proportion:.2f}%")
               if index<=15:
                 SumCount=SumCount+count
                 SumProportion=SumProportion+proportion
            # print(colours)
            cv2.imshow('ng',imnp)
            # print(imnp)

        # Bounding Box
            sub_image = fgbg.apply(frame)  # background subtraction
            ret, thresh = cv2.threshold(sub_image, 127, 255, 0)  # thresholding
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finding the contours
            areas = [cv2.contourArea(c) for c in contours]
            if len(areas) > 1:
               max_index = np.argmax(areas)
               cnt = contours[max_index]
               if areas[max_index] > 70:
                  x, y, w, h = cv2.boundingRect(cnt)
                  cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                  string_ = "fire" + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
                  ID = ID+1
                  # print(ID, ',', str(h), ',', SumCount, ',', SumProportion)
                  cv2.putText(frame, 'fire', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                  # cv2.imshow('fire detection', frame)
            cv2.waitKey(300)
            k = cv2.waitKey(300) & 0xFF == ord('q')
            if k == 27:
                break
        count+=1

cap1.release()
cv2.destroyAllWindows()
