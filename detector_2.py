import cv2
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
import skimage.feature

import sklearn
from sklearn.externals import joblib

clf = joblib.load("train_model.m")


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            if y + window_size[1] > image.shape[0] or x + window_size[0] > image.shape[1]:
                continue
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


im = skimage.io.imread('/tmp/foo.png')
im = skimage.color.rgb2gray(im)

opt_score = 0
opt_entry = None
opt_minx = 0
opt_miny = 0
for minx, miny, entry in sliding_window(im, (256, 256), (32, 32)):
    entry = skimage.transform.resize(entry, (256, 256))
    hf = skimage.feature.hog(entry, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    r = clf.predict_proba([hf])
    if r[0][0] > opt_score:
        opt_score = r[0][0]
        opt_minx = minx
        opt_miny = miny
        opt_entry = entry

skimage.io.imshow(opt_entry)
skimage.io.show()
