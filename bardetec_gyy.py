import cv2
import sys
import os
import numpy as np
import pyzbar.pyzbar
import warnings
import configparser
from xml.dom.minidom import Document
import math
import traceback
import logging
import datetime
from pyzbar.pyzbar import ZBarSymbol
# 霍夫变化及svm向量机包
import joblib
import skimage.feature
from math import fabs,sin,radians,cos


angle_list = [-2,-1,-0.5,0.3,0.5,0.8,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 90]

warnings.filterwarnings("ignore")
if not os.path.exists('barLog'):
    os.makedirs('barLog')
logfilename = f"barLog/Log{datetime.date.today().strftime('%Y%m%d')}.log"
logging.basicConfig(level=logging.INFO, filename=logfilename, format="%(asctime)s: %(module)s: %(message)s")
cf = configparser.ConfigParser()


# img_path = sys.argv[1]
# ini_path = sys.argv[2]
# ili_path = sys.argv[3]
img_path = 'photo/1/20220531003419165.jpg'
ini_path = 'photo/tray6_1.ini'
ili_path= '550'
# 加载SVM模型权重
clf = joblib.load("train_model.m")


def detect_code(src):
    # bars = pyzbar.pyzbar.decode(src,symbols=[ZBarSymbol.QRCODE])
    bars = pyzbar.pyzbar.decode(src)
    if len(bars) == 0:
        for angle in angle_list:
            pic2 = opencv_rotate(src,angle)
            # cv.imshow("pic",pic2)
            # cv.waitKey(100000)
            bars = pyzbar.pyzbar.decode(pic2)
            print('角度:', angle)
            if len(bars) != 0:
                break
    else:
        print('角度:0')
    if len(bars) == 0:
        src_gau = cv2.GaussianBlur(src, (5, 5), 0)
        bars = pyzbar.pyzbar.decode(src_gau)

    if len(bars) == 0:
        imgAutoscaled = np.zeros(src.shape, np.uint8)
        cv2.intensity_transform.autoscaling(src, imgAutoscaled)
        bars = pyzbar.pyzbar.decode(imgAutoscaled)

    # if len(bars) == 0 or len(bars) > 1:
    #     # 创建CLAHE对象
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     # 限制对比度的自适应阈值均衡化
    #     roi = clahe.apply(src)
    #     bars = pyzbar.pyzbar.decode(roi)
    # if len(bars) == 0:
    #     src_gau = cv2.GaussianBlur(src, (5, 5), 0)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     # 限制对比度的自适应阈值均衡化bars
    #     roi = clahe.apply(src_gau)
    #     bars = pyzbar.pyzbar.decode(roi)
    # print(bars)
    barcodeData = 'BarError'
    if bars:
        for barcode in bars:
            barcodeData = str(barcode.data.decode("utf-8")).strip()
    return barcodeData


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
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 2.2 新的宽高，radians(angle) 把角度转为弧度 sin(弧度)
    new_H = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
    new_W = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
    # 2.3 平移
    M[0, 2] += (new_W - w) / 2
    M[1, 2] += (new_H - h) / 2
    rotate = cv2.warpAffine(img, M, (new_W, new_H), borderValue=(0, 0, 0))
    return rotate


if __name__ == '__main__':
    try:
        result_list = []
        # print("11")
        doc = Document()
        if not os.path.exists(img_path):
            # print('112')
            nodeError = doc.createElement('ERROR')
            doc.appendChild(nodeError)
            nodeError.appendChild(doc.createTextNode(f"No such file or directory:{img_path}"))
        elif not os.path.exists(ini_path):
            # print('123')
            nodeError = doc.createElement('ERROR')
            doc.appendChild(nodeError)
            nodeError.appendChild(doc.createTextNode(f"No such file or directory:{ini_path}"))
        else:
            cf.read(ini_path)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            bardetect = cv2.barcode_BarcodeDetector()
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            # cv2.imshow("img1",cv2.resize(img,(600,600)))
            # cv2.waitKey(10000)
            std=np.std(gray)
            # print(std)
            if std==0.0:
                nodeError = doc.createElement('PHOTO ERROR')
                doc.appendChild(nodeError)
                nodeError.appendChild(doc.createTextNode(f"No such file or directory:{ini_path}"))
            else:
                root = doc.createElement('DATA')
                doc.appendChild(root)
                nodeResult = doc.createElement('RESULT')
                root.appendChild(nodeResult)
                for i in range(1, len(cf.sections()) + 1):
                    top_x = int(cf.get(f"Location{i}", "topX"))
                    top_y = int(cf.get(f"Location{i}", "topY"))
                    width = int(cf.get(f"Location{i}", "width"))
                    height = int(cf.get(f"Location{i}", "height"))
                    try:
                        roi = gray[top_y:top_y+height, top_x:top_x+width]
                    except:
                        nodeLocation = doc.createElement(f'Location_{i}')
                        nodeResult.appendChild(nodeLocation)
                        nodeLocation.appendChild(doc.createTextNode("Location error"))
                        continue
                    roi2=roi.copy()
                    gray0 = cv2.GaussianBlur(roi2, (9, 9), 0)
                    # 剪裁图片并进行霍夫变换
                    # svm_roi = cv2.resize(roi, (256, 256))
                    # hf = skimage.feature.hog(svm_roi, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
                    # 预测结果 0为存在电表1为不存在
                    # svm_result = clf.predict([hf])
                    # lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, 0.7)  # 检测直线
                    # lines = lsd.detect(gray0)
                    circles1 = cv2.HoughCircles(gray0, cv2.HOUGH_GRADIENT, 1, 100, param1=200, param2=30, minRadius=0,
                                                maxRadius=0)
                    try:
                        length = (circles1.shape)[1]
                    except:
                        length=0
                    # print(length)
                    # if  length<20:
                    # if svm_result[0]==1:
                    if 0:
                        print('预测为空表位')
                        data = 'null'
                        nodeLocation = doc.createElement(f'Location_{i}')
                        nodeResult.appendChild(nodeLocation)
                        nodeLocation.appendChild(doc.createTextNode(data))
                        # print('222')
                    else:
                        data = detect_code(roi)
                        if data == 'BarError':
                            corners = bardetect.detectAndDecode(roi)[3]
                            if corners is not None:
                                corners = np.int0(corners)
                                rect_corners = np.array(corners[0])
                                rect = cv2.minAreaRect(rect_corners)
                                if rect[1][0] > rect[1][1]:
                                    w = int(rect[1][0])
                                    h = int(rect[1][1])
                                else:
                                    w = int(rect[1][1])
                                    h = int(rect[1][0])
                                pts1 = np.float32([corners[0][1], corners[0][2], corners[0][0], corners[0][3]])
                                pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                                M = cv2.getPerspectiveTransform(pts1, pts2)
                                new_roi = cv2.warpPerspective(roi, M, (w, h))
                                data = detect_code(new_roi)
                                result_list.append(data)
                        nodeLocation = doc.createElement(f'Location_{i}')
                        nodeResult.appendChild(nodeLocation)
                        nodeLocation.appendChild(doc.createTextNode(data))
            fp = open('result.xml', 'w')
            doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='UTF-8')
            fp.close()
            logging.info(f'接收参数：[图片路径：{img_path}，配置文件路径：{ini_path} ]\n识别结果：{result_list}\n')
    except:
        s = traceback.format_exc()
        logging.error(s)
        doc = Document()
        nodeError = doc.createElement('ERROR')
        doc.appendChild(nodeError)
        nodeError.appendChild(doc.createTextNode(f"An unknown error has occurred, see details in {logfilename}"))
        fp = open('result.xml', 'w')
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='UTF-8')
        fp.close()