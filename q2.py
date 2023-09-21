import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import time
from scipy.signal import correlate2d as Conv2D
import ffmpeg 
from skimage.segmentation import felzenszwalb
from skimage.segmentation import active_contour
from scipy.spatial import Delaunay as get_triangles
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


def read_image(name):
    return cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)

sea_bed_image = read_image('res06.jpg')
octopus_image = read_image('res05.jpg')[200:1400,350:1300,:]
octopus_image = np.flip(octopus_image, axis = 1)
octopus_image = np.where(octopus_image.mean(axis = 2, keepdims=True) == 0, 255, octopus_image)

def blend_gray_scale(source, destination, laplacian, mask):
    h, w = source.shape
    index = np.arange(source.size).reshape(source.shape)
    sparse_row = []
    sparse_col = []
    sparse_data = []
    b = []
    edge = np.logical_and((Conv2D(mask, np.ones((3, 3)), mode='same') > 0) ,np.logical_not(mask))

    x, y = np.where(edge)
    for i in range(x.size):
            sparse_row.append(len(b))
            sparse_col.append(index[x[i], y[i]])
            sparse_data.append(1)
            b.append(destination[x[i], y[i]])

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if mask[i, j] : 
                sparse_row.append(len(b))
                sparse_col.append(index[i , j])
                sparse_data.append(-4)

                sparse_row.append(len(b))
                sparse_col.append(index[i - 1, j])
                sparse_data.append(1)

                sparse_row.append(len(b))
                sparse_col.append(index[i + 1, j])
                sparse_data.append(1)

                sparse_row.append(len(b))
                sparse_col.append(index[i, j - 1])
                sparse_data.append(1)

                sparse_row.append(len(b))
                sparse_col.append(index[i, j + 1])
                sparse_data.append(1)

                b.append(laplacian[i, j])
            

    b = np.array(b)
    M = coo_matrix((sparse_data, (sparse_row, sparse_col)), shape = (b.shape[0], index.size))
    print('<', end = '')
    res = lsqr(M, b)[0]
    print('>')
    res = res.reshape(source.shape)
    res = np.where(mask, res, destination)
    return res

def blend(source, destination, background_color = 128):
    laplacian = cv2.Laplacian(source, ddepth=3)
    mask = source.mean(axis =2) != background_color
    res = np.stack([blend_gray_scale(source[:,:,0], destination[:,:,0], laplacian[:,:,0], mask),
                    blend_gray_scale(source[:,:,1], destination[:,:,1], laplacian[:,:,1], mask),
                    blend_gray_scale(source[:,:,2], destination[:,:,2], laplacian[:,:,2], mask)], axis = 2)
    return res, laplacian


tmp_image = octopus_image
tmp_image = np.where(octopus_image.mean(axis = 2, keepdims=True) > 200, np.ones(3, dtype='uint8') * 128, octopus_image)
tmp_image = cv2.resize(tmp_image,dsize=(0, 0), fx = 1/4, fy  =1/4)
tmp_image_2 = sea_bed_image.copy()
tmp_image_2 = tmp_image_2[230:230+tmp_image.shape[0], 740:740+tmp_image.shape[1],:]
tmp, laplac = blend(tmp_image, tmp_image_2, background_color=128)


bkgrd = cv2.resize(sea_bed_image, dsize=(0, 0), fx = 1, fy= 1)
bkgrd[230:230+tmp_image.shape[0], 740:740+tmp_image.shape[1],:] = np.maximum(np.minimum(tmp, 255), 0).astype('uint8')

cv2.imwrite('res07.jpg', np.flip(bkgrd, axis =2))
