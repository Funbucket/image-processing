import numpy as np
import cv2
import matplotlib.pyplot as plt

MAX_GRAY_LEVEL = 255


def my_calcHist(img):
    """
    :param img:
    :return: src image에 대한 histogram array
    2차원 배열 이미지 행렬값을 일차원으로 변환하여 각 gray level에 맞는 pixel 개수를 구하여 반환
    """

    # hist = np.zeros(256)
    # (h, w) = img.shape
    # for row in range(h):
    #     for col in range(w):
    #         hist[img[h, w]] += 1

    return np.bincount(img.flatten(), minlength=MAX_GRAY_LEVEL + 1)


def my_PDF2CDF(pdf):
    """
    :param pdf:
    :return: pdf 값을 누적시킨 1차원 배열
    """

    # pdf_len = pdf.shape[0]
    # cdf = np.zeros((pdf_len,))
    # cdf[0] = pdf[0]
    # for i in range(1, pdf_len):
    #     cdf[i] = pdf[i] + cdf[i - 1]

    return np.cumsum(pdf)


def my_normalize_hist(hist, pixel_num):
    """
    :param hist: histogram 1 차원 배열
    :param pixel_num: image의 전체 pixel 수
    :return: normalized 된 histogram
    """
    return hist / pixel_num


def my_denormallize(normalized, gray_level):
    """
    :param normalized:
    :param gray_level:
    :return: 누적 pdf 에 max gray level을 곱한 1차원 배열
    """
    return normalized * gray_level


def my_calcHist_equalization(denormalized, hist):
    """
    :param denormalized:
    :param hist:
    :return:
    """

    # hist_equal = np.zeros(denormalized.shape).astype('int')
    # for i in range(len(hist_equal)):
    #     hist_equal[denormalized[i]] += hist[i]

    equalized_histogram = np.zeros(MAX_GRAY_LEVEL + 1)

    for i, n in enumerate(denormalized):
        equalized_histogram[n] += hist[i]

    return equalized_histogram


def my_equal_img(src, gray_level):
    """
    :param src:
    :param gray_level:
    :return:
    """
    (h, w) = src.shape

    dst = np.zeros_like(src)
    for i in range(h):
        for j in range(w):
            dst[i, j] = gray_level[src[i, j]]

    return dst


#input_image의  equalization된 histogram & image 를 return
def my_hist_equal(src):
    (h, w) = src.shape
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram)
    denormalized_output = my_denormallize(normalized_output, MAX_GRAY_LEVEL)
    output_gray_level = denormalized_output.astype(int)
    print(output_gray_level)
    hist_equal = my_calcHist_equalization(output_gray_level, histogram)

    ### dst : equalization 결과 image
    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal


if __name__ == '__main__':
    # Test on simple matrix
    test_img = np.array([[0, 1, 1, 1, 2], [2, 3, 3, 3, 3],
                    [3, 3, 3, 4, 4], [4, 4, 4, 4, 4],
                    [4, 5, 5, 5, 7]], dtype=np.uint8)

    hist = my_calcHist(test_img)

    dst, hist_equal = my_hist_equal(test_img)

    test_img_to_show = cv2.resize(test_img, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('equalization before image', test_img_to_show)
    test_dst_to_show = cv2.resize(dst, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('equalization after image', test_dst_to_show)

    plt.figure(1)
    plt.title('my histogram')
    plt.bar(np.arange(len(hist)), hist, width=0.5, color='g')

    plt.figure(2)
    plt.title('my histogram equalization')
    plt.bar(np.arange(len(hist_equal)), hist_equal, width=0.5, color='g')

    # Test on real image
    test_img = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(test_img)
    dst, hist_equal = my_hist_equal(test_img)

    cv2.imshow('equalizetion before image', test_img)
    cv2.imshow('equalizetion after image', dst)

    plt.figure(1)
    plt.title('my histogram')
    plt.bar(np.arange(len(hist)), hist, width=0.5, color='g')

    plt.figure(2)
    plt.title('my histogram equalization')
    plt.bar(np.arange(len(hist_equal)), hist_equal, width=0.5, color='g')

    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()

