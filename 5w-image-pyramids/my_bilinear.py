import cv2
import numpy as np


def my_bilinear(src, scale):

    (h, w) = src.shape

    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst), np.uint8)

    ############################################
    # TODO                                     #
    # my_bilinear 완성                          #
    ############################################
    src = cv2.copyMakeBorder(src, 0, 1, 0, 1, cv2.BORDER_REPLICATE)  # 오른쪽, 아래에 padding 값을 주어서 out of index error를 방지한다.

    for row in range(h_dst):
        for col in range(w_dst):
            # center_r = row / scale
            # center_c = col / scale
            #
            # t = center_r - int(center_r)
            # s = center_c - int(center_c)
            #
            # r1 = int(center_r)
            # r2 = min(int(center_r + 1), h - 1)
            # c1 = int(center_c)
            # c2 = min(int(center_c + 1), w - 1)
            #
            # value = 0
            # value += (1 - s) * (1 - t) * src[r1, c1]
            # value += s * (1 - t) * src[r1, c2]
            # value += (1 - s) * t * src[r2, c1]
            # value += s * t * src[r2, c2]
            #
            # value = np.round(value)
            # dst[row, col] = value

            src_row = min(int(row / scale), h - 1)
            src_col = min(int(col / scale), w - 1)
            dx = (col / scale) - src_col
            dy = (row / scale) - src_row
            val = (1 - dx) * (1 - dy) * src[src_row, src_col] + dx * (1 - dy) * src[src_row, src_col+1] + (1 - dx) * dy * src[src_row + 1, src_col] + dx * dy * src[src_row + 1, src_col + 1]
            dst[row, col] = val

    return dst


if __name__ == '__main__':
    img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

    down_cv2 = cv2.resize(img, dsize=(0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    down_up_cv2 = cv2.resize(down_cv2, dsize=(0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_LINEAR)

    down_my = my_bilinear(img, scale=0.25)
    down_up_my = my_bilinear(down_my, scale=4.0)

    cv2.imshow('original image', img)
    cv2.imshow('down_cv2_n image', down_cv2)
    cv2.imshow('down_up_cv2_n', down_up_cv2)
    cv2.imshow('down_my', down_my)
    cv2.imshow('down_up_my', down_up_my)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # x = np.arange(9).reshape(3, 3)
    # my_bilinear(x, scale=2)

