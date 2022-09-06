from bardetec_gyy import detect_code
import cv2 as cv
import glob
from math import fabs,sin,radians,cos


angle_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 90]

def opencv_rotate(img, angle):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0
    # 2.1获取M矩阵
    """
    M矩阵
    [
    cosA -sinA (1-cosA)*centerX+sinA*centerY
    sinA cosA  -sinA*centerX+(1-cosA)*centerY
    ]
    """
    M = cv.getRotationMatrix2D(center, angle, scale)
    # 2.2 新的宽高，radians(angle) 把角度转为弧度 sin(弧度)
    new_H = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
    new_W = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
    # 2.3 平移
    M[0, 2] += (new_W - w) / 2
    M[1, 2] += (new_H - h) / 2
    rotate = cv.warpAffine(img, M, (new_W, new_H), borderValue=(0, 0, 0))
    return rotate

images = glob.glob("picture/img*")
print(images)
for img in images:
    pic = cv.imread(img)
    print(detect_code(pic))

    # for angle in angle_list:
    #     pic2 = opencv_rotate(pic,angle)
    #     # cv.imshow("pic",pic2)
    #     # cv.waitKey(100000)
    #     bar = detect_code(pic2)
    #     if bar != "BarError":
    #         print(img, bar,'角度:',angle)
    #         break

# cv.imshow('hh',img)
# cv.waitKey(10000)