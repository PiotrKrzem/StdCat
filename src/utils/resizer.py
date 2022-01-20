import os
import xml.etree.ElementTree as et
import numpy as np
import cv2


# specify paths before use
path_img = r''
path_off = r''
path_dest = r''
#
images = []
offsets = []
changesize = True
width = 224
height = 224
blurparams = (10, 10)

for root, dirs, files in os.walk(path_img):
    for file in files:
        images.append(os.path.join(root, file))


for root, dirs, files in os.walk(path_off):
    for file in files:
        offsets.append(os.path.join(root, file))
print(len(images))
for i in range(len(images)):

    tree = et.parse(offsets[i])
    root = tree.getroot()

    for time in root.iter('xmax'):
        xmax = time.text

    for time in root.iter('xmin'):
        xmin = time.text

    for time in root.iter('ymax'):
        ymax = time.text

    for time in root.iter('ymin'):
        ymin = time.text

    img = cv2.imread(images[i])

    for j in range(-10, 10, 5):
        if int(xmin) + j > 0 and int(ymin) + j > 0 and int(ymax) + j < img.shape[0] and int(xmax) + j < img.shape[1]:
            roi = img[int(ymin) + j : int(ymax) + j, int(xmin) + j : int(xmax) + j]
            if changesize == True:
                roi = cv2.resize(roi, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(path_dest + 'dog' + str(i) + 'ver' + str(j) + '.jpg', roi)
            # flip x axis
            flip = cv2.flip(roi, 1)
            cv2.imwrite(path_dest + 'dog' + str(i) + 'ver' + str(j) + 'ver_flipped_x.jpg', flip)
            # flip y axis
            flip = cv2.flip(roi, 0)
            cv2.imwrite(path_dest + 'dog' + str(i) + 'ver' + str(j) + 'ver_flipped_y.jpg', flip)
            # flib both x and y axis
            flip = cv2.flip(roi, -1)
            cv2.imwrite(path_dest + 'dog' + str(i) + 'ver' + str(j) + 'ver_flipped_xy.jpg', flip)
            # blur image
            image = cv2.blur(roi, blurparams)
            cv2.imwrite(path_dest + 'dog' + str(i) + 'ver' + str(j) + 'ver_blurry.jpg', flip)
