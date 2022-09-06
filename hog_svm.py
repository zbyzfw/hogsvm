import os
import cv2
import skimage.color
import skimage.feature
import skimage.io
import skimage.transform
import sklearn.svm
import glob

# 图片调整
def read_and_preprocess(im_path):
    print(im_path)
    im = cv2.imread(im_path,0)
    # if len(im.shape) == 3:
    #     im = skimage.color.rgb2gray(im)
    # im = skimage.transform.resize(im, (128, 128))
    im = cv2.resize(im,(256,256))
    return im


def get_data_tr():
    X = []
    Y = []
    for entry in os.scandir(r'data/train/cat'):
        # print('========',entry)
        im = read_and_preprocess(entry.path)
        hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        X.append(hf)
        Y.append(0)
    for entry in os.scandir(r'data/train/dog'):
        im = read_and_preprocess(entry.path)
        hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        X.append(hf)
        Y.append(1)
    return X, Y


def get_data_te():
    X = []
    Y = []
    for entry in os.scandir(r'data/test/cat'):
        im = read_and_preprocess(entry.path)
        hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        X.append(hf)
        Y.append(0)
    for entry in os.scandir(r'data/test/dog'):
        im = read_and_preprocess(entry.path)
        hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        X.append(hf)
        Y.append(1)
    return X, Y


# 训练
Xtr, Ytr = get_data_tr()
clf = sklearn.svm.SVC(probability=True)
clf.fit(Xtr, Ytr)

# 测试
Xte, Yte = get_data_te()
r = clf.predict(Xte)
s = 0
for i in range(len(r)):
    if r[i] == Yte[i]:
        s += 1
print('acc:', s / len(r))

import joblib
joblib.dump(clf, "train_model.m")
