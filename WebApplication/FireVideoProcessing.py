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

def FireVideoProcess():
  cap1 = cv2.VideoCapture('./static/Videos/4cm_test.mp4')
  fgbg = cv2.createBackgroundSubtractorMOG2()
  count = 0

  with open("./static/data/4cm_test.csv", "w", newline='') as csvFile:
     writer = csv.writer(csvFile)
     writer.writerow(['I', 'H', 'S', 'P'])
     sys.stdout = csvFile
     # sys.stdout = open("Data.csv", "w")
     # names = ['I', 'H', 'S', 'P']
     # print('I', ',', 'H', ',', 'S', ',', 'P')
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
            h, w = imnp.shape[:2]
            colours, counts = np.unique(imnp.reshape(-1,3), axis=0, return_counts=1)
            SumCount=0
            SumProportion=0
            for index, colour in enumerate(colours):
               count = counts[index]
               proportion = (100 * count) / (h * w)
               if index<=20:
                 SumCount=SumCount+count
                 SumProportion=SumProportion+proportion

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
                  print(ID, ',', str(h), ',', SumCount, ',', SumProportion)
                  cv2.putText(frame, 'fire', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                  # cv2.imshow('fire detection', frame)
            cv2.waitKey(300)
            k = cv2.waitKey(300) & 0xFF == ord('q')
            if k == 27:
                break
        count+=1

  cap1.release()
  cv2.destroyAllWindows()
  sys.stdout.close()

if __name__ == '__main__':
    FireVideoProcess()