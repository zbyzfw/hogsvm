import cv2
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import skimage.io
import skimage.transform


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            if y + window_size[1] > image.shape[0] or x + window_size[0] > image.shape[1]:
                continue
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


im = skimage.io.imread('ball/detect.jpg')
im = skimage.color.rgb2gray(im)

fig, axes = plt.subplots()
div = axes.imshow(im)
gaiter = sliding_window(im, (256, 256), (16, 16))


def update(*args):
    minx, miny, entry = next(gaiter)
    new_img = np.copy(im)
    new_img = skimage.color.gray2rgb(new_img)

    r0 = miny
    c0 = minx
    r1 = r0 + entry.shape[0]
    c1 = c0 + entry.shape[1]

    cv2.rectangle(new_img, (c0, r0), (c1, r1), (1, 1, 0), 5)
    div.set_data(new_img)


ani = matplotlib.animation.FuncAnimation(fig, update, interval=50, repeat=False)
plt.show()
