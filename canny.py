from sympy import im


import cv2
import numpy as np

img = cv2.imread("screenshot.png",0)
cv2.imwrite("canny.png",cv2.Canny(img, 200, 300))
