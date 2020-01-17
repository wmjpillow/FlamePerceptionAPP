import cv2

img = cv2.imread('/Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/WebApplication/static/img/Area/frame2610.jpg')
gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


gs_neg = 255 - gs.astype(int)
gs_neg[gs_neg>50] = 1000
gs_neg = gs_neg.astype('uint8')


#Show Image List
show_imagename = ['negative']
show_image = [gs_neg]

n_showimg = len(show_image)

print(n_showimg)
cv2.imshow(show_imagename[0],show_image[0])
# cv2.imwrite(show_imagename[0],show_image[0])

#Image Showing Sequencing
# for k in range (0,n_showimg):
#     cv2.imshow(show_imagename[k],show_image[k])

cv2.waitKey(0)
cv2.destroyAllWindows()