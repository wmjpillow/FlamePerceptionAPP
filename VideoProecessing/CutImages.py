from PIL import Image
import os.path, sys

path = "/Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/WebApplication/static/img/Area"
dirs = os.listdir(path)

def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)         #corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((10, 10, 972, 850)) #corrected
            imCrop.save(f + '.jpg', "BMP", quality=100)

crop()