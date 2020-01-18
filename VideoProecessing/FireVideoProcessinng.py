import cv2
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image

# get_ipython().magic(u'matplotlib inline')
# REFERENCE: https://github.com/benjamincastillo2020/FireDetectionCode

cap1 = cv2.VideoCapture('/Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/WebApplication/static/Videos/4cm_test.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

def processLog(filename):
    # print(f"Processing log: {filename}")
    p=Path(f"{filename}")
    # print(f"{p.stem}")
    # Open this image and make a Numpy version for easy processing
    im   = Image.fromarray(filename).convert('RGBA').convert('RGB')
    imnp = np.array(im)
    h, w = imnp.shape[:2]
    # Get list of unique colours...
    # Arrange all pixels into a tall column of 3 RGB values and find unique rows (colours)
    colours, counts = np.unique(imnp.reshape(-1,3), axis=0, return_counts=1)
    # Get area and portion of black color
    SumCount=0
    SumProportion=0
    # Iterate through unique colours
    for index, colour in enumerate(colours):
        count = counts[index]
        proportion = (100 * count) / (h * w)
        if index<=20:
          SumCount=SumCount+count
          SumProportion=SumProportion+proportion
    print(f"  {SumCount}, {SumProportion}  ")


# sys.stdout = open("height.txt", "w")
index = 0
while (1):
    ret, frame = cap1.read()  # reading the image

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
            #            serport.write(string_)
            index = index+1
            print(index, ',', str(h))
            cv2.putText(frame, 'fire', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow('fire detection', frame)


    # flame shape area
    gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gs_neg = 255 - gs.astype(int)
    gs_neg[gs_neg > 50] = 1000
    gs_neg = gs_neg.astype('uint8')
    # Show Image List
    show_imagename = ['negative']
    show_image = [gs_neg]
    # n_showimg = len(show_image)
    # print(n_showimg)
    processLog(gs_neg)
    # cv2.imshow(show_imagename[0], show_image[0])

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap1.release()
cv2.destroyAllWindows()
# sys.stdout.close()


