import glob
import os
import numpy
import cv2

for fil in glob.glob('./images/*.jpeg'):
    img= cv2.imread(fil)
    img = cv2.resize(img, (608,608))
    cv2.imwrite(fil, img)
    
