import math
import cv2
import numpy as np

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_filtering import my_filtering


def get_DoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################

    # y, x = np.mgrid[-(fsize//2):(fsize//2)+1, -(fsize//2):(fsize//2)+1]
    y, x = np.ogrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2):(fsize // 2) + 1]  # x, y 좌표 matrix 생성

    DoG_x = -(x / (sigma**2)) * np.exp(-(x*x)) / (2*sigma*sigma)
    DoG_y = -(y / (sigma**2)) * np.exp(-(x*x)) / (2*sigma*sigma)

    # 필터의 총 합을 0으로 만들기
    DoG_x = DoG_x - (DoG_x.sum()/fsize**2)
    DoG_y = DoG_y - (DoG_y.sum()/fsize**2)

    return DoG_x, DoG_y


def main():
    src = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_DoG_filter(fsize=3, sigma=1)

    dst_x = my_filtering(src, DoG_x, 'zero')
    dst_y = my_filtering(src, DoG_y, 'zero')

    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################
    dst = np.sqrt(dst_x**2 + dst_y**2)

    cv2.imshow('dst_x', dst_x/255)
    cv2.imshow('dst_y', dst_y/255)
    cv2.imshow('dst', dst/255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

