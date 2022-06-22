import math

import numpy as np


def get_space_gaus(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    gaus2D =   1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))

    return gaus2D


def get_pixel_gaus(msize, x=2, y=2, sigma=0.5):
    src = np.array([[3, 4, 6, 4, 3],
                    [7, 8, 2, 4, 3],
                    [4, 4, 5, 4, 3],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]])
    print(src)
    print(src[1][1])
    rad = msize // 2

    print(src[y-rad:y+rad+1][x-rad:x+rad+1])

    dst = src[y-rad:y+rad+1][x-rad:x+rad+1] - src[y][x]

    print(dst)


if __name__ == '__main__':
    msize = 5
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]

    # print(get_space_gaus(msize))

    get_pixel_gaus(msize)