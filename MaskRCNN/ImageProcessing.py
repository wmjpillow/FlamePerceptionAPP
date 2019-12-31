import cv2
from PIL import Image
import os,sys
import glob
import mahotas

path="/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/data/JPEGImages_copy"
path_of_save= "/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/data/testImages"

imgpath = sorted(glob.glob("/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/data/JPEGImages/*.jpg"))

for image in glob.glob( os.path.join(path, '*.jpg')):
     print (image)
     imagepath= str(image)
     print (imagepath)
     image2 = mahotas.imread(image)
     gs = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

     gs_neg = 255 - gs.astype(int)
     gs_neg[gs_neg>10] = 1200
     gs_neg = gs_neg.astype('uint8')


     show_image = [gs_neg]
     show_imagename = ['negative']
     n_showimg = len(show_image)
     print (n_showimg)

     # show_image.save(os.path.join(path_of_save, image))
     mahotas.imsave(imagepath, gs_neg)



# for k in range (0,n_showimg):
#      show_image[k].save(os.path.join(path_of_save, image))
#      print (k)
#      cv2.imshow(show_imagename[k], show_image[k])

#Show Image List
# show_imagename = ['negative']
# show_image = [gs_neg]

# n_showimg = len(show_image)

#Image Showing Sequencing
# for k in range (0,n_showimg):
#     cv2.imshow(show_imagename[k],show_image[k])

cv2.waitKey(0)
cv2.destroyAllWindows()
