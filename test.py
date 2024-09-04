import cv2
import numpy as np

img = cv2.imread(r"binPic.png")

# 图片有点大 先对其 缩放 方便展示
img = cv2.resize(img, None, fx=1, fy=1)
h, w, ch = img.shape

# 旋转矩阵
M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1)  # 旋转默认为逆时针

new = cv2.warpAffine(img, M, (int(w*1), int(h*1)))

cv2.imshow('img', img)
cv2.imshow('new', new)
cv2.waitKey(0)