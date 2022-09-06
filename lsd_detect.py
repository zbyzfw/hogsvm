import cv2
import numpy as np

gray0 = cv2.imread('picture/818.png',0)
# print(gray0.shape)
# dst = cv2.cvtColor(gray0,cv2.COLOR_GRAY2BGR)
# 根据高度判断l值,l值越大直线越细致
dst = np.uint8(gray0)
print(dst.dtype)
_,gray1 = cv2.threshold(gray0, 200,255,cv2.THRESH_TRUNC)
cv2.imshow("aa", gray1)
cv2.waitKey(1000000)
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, 0.1)  # 检测直线
# final_im = cv2.normalize(src=dst, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
lines = lsd.detect(dst)
# print(lines[0][0])
lines2 = []
angle = []
if lines[0] is not None:
    # 遍历每条检测出的线
    for line in lines[0]:
        # 获取合适长度的线(line1-line3绝对值 > 4*模板高度/5)
        print('线宽:',np.abs(line[0][0] - line[0][2]))

        if np.abs(line[0][0] - line[0][2]) > 60:
            # y = np.expand_dims(line, axis=0)
            # print(y)
            ag = np.arctan(float(line[0][3] - line[0][1]) / float(line[0][2] - line[0][0])) * 180 / np.pi
            if 30> ag >16:
            # np.append(lines2,y,axis=0)
                lines2.append(line)
                angle.append(ag)
                # print(lines2)
        # if np.abs(line[0][0] - line[0][2]) > (4 * w2 / 5) \
        #         and np.abs(line[0][1] - line[0][3]) < (h2 / 10):
            # if int((line[0][2] - line[0][1]))>200:
            # line2 = line
            # print(line2)
            # 判断获取到的线是否准确
            # if line2[0][3] != line2[0][1] and line2[0][2] != line2[0][0]:
            #     break
        try:
            print('角度:',np.arctan(float(line[0][3] - line[0][1]) /
                              float(line[0][2] - line[0][0])) * 180 / np.pi)
            # 角度摆正
            # 计算角度 对边/临边 算出角度后乘180/pi
            # angle = np.arctan(float(line2[0][3] - line2[0][1]) /
            #                   float(line2[0][2] - line2[0][0])) * 180 / np.pi

            # 将图像以一定角度旋转,获得转换矩阵
            # M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # 仿射变换将图像矫正对齐,宽高和原来一致
            # rotated = cv2.warpAffine(img, M, (h1, w1))
            # 灰度变换

            # if len(rotated.shape) == 3:
            #     gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            # else:
            #     gray2 = rotated.copy()
        except Exception as e:
            print(e)
print('合并',angle)
lines2 = np.array(lines2)
# y = np.expand_dims(lines2, axis=0)

with_line_img = lsd.drawSegments(dst, lines2)
# cv2.imshow("img1",with_line_img)
# cv2.imwrite("line_pic.jpg",with_line_img)
# cv2.waitKey(1000000)

