import cv2
import numpy as np
import sys
# import skimage
# import serial
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from werkzeug import secure_filename

# get_ipython().magic(u'matplotlib inline')
# REFERENCE: https://github.com/benjamincastillo2020/FireDetectionCode
# Passing a cv2 frame from view to Flask:
# https://stackoverflow.com/questions/46373163/passing-a-cv2-frame-from-view-to-template
# Video Streaming with Flask:
# https://blog.miguelgrinberg.com/post/video-streaming-with-flask


def BoundingBox():

  cap1 = cv2.VideoCapture('./static/Videos/4cm_test.mp4')
  fgbg = cv2.createBackgroundSubtractorMOG2()
  # serport = serial.Serial("COM1", 115200)

  sys.stdout = open("./static/data/height.txt", "w")
  index = 0
  while (1):
    ret, frame = cap1.read()  # reading the image
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
            # cv2.putText(frame, 'fire', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # cv2.imshow('fire detection', frame)
            cv2.imwrite('t.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')
        # else:
        #     print("none")
            # break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
  cap1.release()
  cv2.destroyAllWindows()
  sys.stdout.close()

if __name__ == '__main__':
    BoundingBox()
