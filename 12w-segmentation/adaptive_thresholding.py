import cv2
import numpy as np

def adaptive_threshold(img, group_num=4):
    img_split = np.hsplit(img, group_num)
    thresholds = list()
    dst = list()

    #################################################
    # TODO                                          #
    # 구분된 이미지마다 cv2의 otsu 방법으로 이진화.       #
    # 이후, 하나의 이미지로 다시 합친다.                 #
    #################################################

    for i in range(group_num):
        th, d = cv2.threshold(img_split[i], 0, 255, cv2.THRESH_OTSU)
        thresholds.append(th)
        dst.append(d)

    dst = np.concatenate(dst, axis=1)

    return dst, thresholds


if __name__ == '__main__':
    img = cv2.imread('circles_adaptive_threshold.png', cv2.IMREAD_GRAYSCALE)

    dst, val = adaptive_threshold(img, group_num=4)
    print('Threshold: ', val)

    cv2.imshow('original', img)
    cv2.imshow('adaptive threshold', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


