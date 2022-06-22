import cv2
import numpy as np


def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w), dtype=int)
    pad_img[p_h:p_h + h, p_w:p_w + w] = src

    if pad_type == 'repetition':
        print('repetition padding')

        # up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]
        # left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        # right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1:p_w + w]

    else:
        print('zero padding')

    return pad_img


def dilation(B, S):
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                           #
    ###############################################
    b_h, b_w = B.shape
    s_h, s_w = S.shape
    p_h, p_w = (S.shape[0] // 2, S.shape[1] // 2)
    pad_img = my_padding(B, (p_h, p_w))

    # 패딩처리 된 상태에서 픽셀값 추가 -> index out of range 방지
    dst = pad_img.copy()

    for row in range(0, b_h):
        for col in range(0, b_w):
            if B[row, col]:
                dst[row:row+s_h, col:col+s_w] = cv2.bitwise_or(pad_img[row:row+s_h, col:col+s_w], S)

    # remove padding
    dst = dst[p_h:p_h+b_h, p_w:p_w+b_w]

    return dst


def erosion(B, S):
    ###############################################
    # TODO                                        #
    # erosion 함수 완성                            #
    ###############################################
    b_h, b_w = B.shape
    s_h, s_w = S.shape
    p_h, p_w = (S.shape[0] // 2, S.shape[1] // 2)
    pad_img = my_padding(B, (p_h, p_w))
    dst = B.copy()

    for row in range(0, b_h):
        for col in range(0, b_w):
            if B[row, col]:
                if np.all(cv2.bitwise_and(pad_img[row:row + s_h, col:col + s_w], S) == 1):
                    dst[row, col] = 1
                else:
                    dst[row, col] = 0

    return dst


def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                            #
    ###############################################
    dst = erosion(B, S)
    dst = dilation(dst, S)
    return dst


def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                            #
    ###############################################
    dst = dilation(B, S)
    dst = erosion(dst, S)
    return dst


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])

    cv2.imwrite('test_binary_image.png', (B*255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('test_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('test_erosion.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('test_opening.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('test_closing.png', img_closing)

    test_img = cv2.imread("morphology_img.png",cv2.IMREAD_GRAYSCALE)
    test_img = test_img / 255.
    test_img = np.round(test_img)
    test_img = test_img.astype(np.uint8)

    img_dilation = dilation(test_img, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(test_img, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(test_img, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(test_img, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)





