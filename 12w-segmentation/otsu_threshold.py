import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_threshold(img, th=120):
    ######################################################
    # TODO                                               #
    # 실습시간에 배포된 코드 사용                             #
    ######################################################
    dst = np.zeros(img.shape, img.dtype)
    dst[img >= th] = 255
    dst[img < th] = 0

    return dst


def my_otsu_threshold(img):
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    p = hist / np.sum(hist) + 1e-7

    ######################################################
    # TODO                                               #
    # Otsu 방법을 통해 threshold 구한 후 이진화 수행          #
    # cv2의 threshold 와 같은 값이 나와야 함                 #
    ######################################################

    q1 = np.zeros(256).astype('float64')
    q2 = np.zeros(256).astype('float64')
    m1 = np.zeros(256).astype('float64')
    m2 = np.zeros(256).astype('float64')

    q1[0] = p[0]
    m1[0] = 0
    q2 = 1 - q1[0]
    m2[0] = (np.sum(p * np.arange(0, 256)) - (q1[0] * m1[0])) / q2[0]

    for k in range(0, 255):
        q1[k+1] = q1[k] + p[k+1]
        q2[k+1] = 1 - q1[k+1]
        m1[k+1] = (q1[k]*m1[k] + (k+1)*p[k+1]) / q1[k+1]
        m2[k+1] = (q2[k]*m2[k] - (k+1)*p[k+1]) / q2[k+1]

    th = np.argmax(q1 * q2 * (m1 - m2) ** 2)
    dst = apply_threshold(img, th)

    return th, dst


if __name__ == '__main__':
    img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)

    my_otsu_threshold(img)

    th_cv2, dst_cv2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    th_my, dst_my = my_otsu_threshold(img)

    print('Threshold from cv2: {}'.format(th_cv2))
    print('Threshold from my: {}'.format(th_my))

    cv2.imshow('original image', img)
    cv2.imshow('cv2 threshold', dst_cv2)
    cv2.imshow('my threshold', dst_my)

    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
