import cv2
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
import skimage.feature

import sklearn
import joblib

clf = joblib.load("train_model.m")

def read_and_preprocess(im_path):
    print(im_path)
    im = skimage.io.imread(im_path)
    if len(im.shape)==3:
        im = skimage.color.rgb2gray(im)
    im = skimage.transform.resize(im, (256, 256))
    return im


# im = skimage.io.imread('data/test/dog/.jpg')
# im = skimage.color.rgb2gray(im)
im = read_and_preprocess('data/test/dog/014.jpg')
hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))


r = clf.predict([hf])

print(r)
# skimage.io.imshow(opt_entry)
# skimage.io.show()
