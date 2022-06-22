import cv2
import numpy as np


def my_resize_nearest_interpolation(src, scale):
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst), np.uint8)
    for row in range(h_dst):
        for col in range(w_dst):
            r = min(int(row / scale + 0.5), h-1)
            c = min(int(col / scale + 0.5), w-1)
            dst[row, col] = src[r, c]

    return dst


if __name__ == '__main__':
    img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

    down_cv2 = cv2.resize(img, dsize=(0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    down_up_cv2 = cv2.resize(down_cv2, dsize=(0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_LINEAR)

    # down_my = my_resize_nearest_interpolation(img, scale=0.25)
    down_my = my_resize_nearest_interpolation(img, scale=3.0)
    down_up_my = my_resize_nearest_interpolation(down_my, scale=4.0)

    cv2.imshow('original image', img)
    cv2.imshow('down_cv2_n image', down_cv2)
    cv2.imshow('down_up_cv2_n', down_up_cv2)
    cv2.imshow('down_my', down_my)
    cv2.imshow('down_up_my', down_up_my)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
