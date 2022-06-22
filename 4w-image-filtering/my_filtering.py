import cv2
import numpy as np


def my_padding(src, pad_shape, pad_type='zero'):
    """
    :param src: padding 이 추가될 원본 이미지
    :param pad_shape: mask 의 높이와 넓이
    :param pad_type: default = zero padding or repetition padding
    :return: padding 이 추가된 이미지
    """
    (h, w) = src.shape
    (p_h, p_w) = pad_shape

    pad_img = np.zeros((h+2*p_h, w+2*p_w), dtype=np.uint8)  # mask 가 src image 에서 초과하는 부분만큼 패딩추가하여 생성
    pad_img[p_h:p_h+h, p_w:p_w+w] = src  # src image 는 원래자리에 그대로 복사

    if pad_type == 'repetition':
        print('repetition padding')
        # up
        pad_img[:p_h, p_w:p_w+w] = src[0, :]
        # down
        pad_img[p_h+h:, p_w:p_w+w] = src[h-1, :]

        # left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        # right
        pad_img[:, p_w+w:] = pad_img[:, p_w+w-1:p_w+w]

        return pad_img
    else:
        print('zero padding')
        return pad_img  # padding 값이 0 인 상태 그대로 반환


def my_filtering(src, mask, pad_type='zero'):
    """
    :param src: filtering 될 original image
    :param mask: filtering mask
    :param pad_type: padding 처리 방법 zero padding or repetition padding
    :return: filtered image
    """
    (h, w) = src.shape  # origin image matrix
    src_pad = my_padding(src, (mask.shape[0]//2, mask.shape[1]//2), pad_type)  # padding 처리된 image matrix
    dst = np.zeros((h, w))

    #########################################################
    # TODO                                                  #
    # dst 완성                                               #
    # dst : filtering 결과 image                             #
    #########################################################
    (m_h, m_w) = mask.shape  # mask matrix

    for i in range(w):
        for j in range(h):
            # mask matrix * (padding 처리된 image 의 mask 사이즈의 영역 matrix)
            val = np.sum(np.multiply(mask, src_pad[i:i+m_h, j:j+m_w]))
            val = np.clip(val, 0, 255)  # overflow 방지 255 이상 값은 255 로 유지, 0 이하 값은 0 로 유지
            dst[i, j] = val

    dst = (dst+0.5).astype(np.uint8)

    return dst

    
def get_average_mask(fshape):
    """
    :param fshape: mask matrix
    :return: 총합이 '1'인 2x2 average mask matrix
    """
    print('get average filter')

    mask = np.ones(fshape)  # 입력한 크기의 모든 원소가 1 인 matrix 생성
    mask = mask / mask.sum()  # matrix의 총합을 1 로 만든다
    
    return mask


def get_sharpening_mask(fshape):
    """
    :param fshape: mask matrix
    :return: 총합이 '1'인 2x2 sharpening mask matrix
    """
    print('get sharpening filter')
    ##################################################
    # TODO                                           #
    # mask 완성                                       #
    ##################################################

    (h, w) = fshape
    mask = np.zeros((h, w))
    mask[h//2, w//2] = 2
    average_mask = get_average_mask(fshape)
    mask = mask - average_mask

    return mask


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # 3x3 filter
    dst_average_3x3 = my_filtering(src, get_average_mask(fshape=(3, 3)))
    dst_sharpening_3x3 = my_filtering(src, get_sharpening_mask(fshape=(3, 3)))

    # 11x13 filter
    dst_average_11x13 = my_filtering(src, get_average_mask(fshape=(11, 13)))
    dst_sharpening_11x13 = my_filtering(src, get_sharpening_mask(fshape=(11, 13)))

    cv2.imshow('original', src)

    cv2.imshow('average filter 3x3', dst_average_3x3)
    cv2.imshow('sharpening filter 3x3', dst_sharpening_3x3)
    cv2.imshow('average filter 11x13', dst_average_11x13)
    cv2.imshow('sharpening filter 11x13', dst_sharpening_11x13)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
