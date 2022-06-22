import cv2
import numpy as np


def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
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


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (mask.shape[0] // 2, mask.shape[1] // 2), pad_type)
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row + mask.shape[0], col:col + mask.shape[1]] * mask)
            dst[row, col] = val

    return dst


def get_Gaussian_mask(fsize, sigma=1):
    y, x = np.mgrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2):(fsize // 2) + 1]

    # 2차 gaussian mask 생성
    gaus2D = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D


def get_sobel_mask():
    derivative = np.array([[-1, 0, 1]])
    blur = np.array([[1], [2], [1]])

    x = np.dot(blur, derivative)
    y = np.dot(derivative.T, blur.T)

    return x, y


def apply_gaussian_filter(src, fsize=3, sigma=1):
    #####################################################
    # TODO                                              #
    # src에 gaussian filter 적용                         #
    #####################################################
    mask = get_Gaussian_mask(fsize, sigma)
    dst = my_filtering(src, mask)  # 가우시안 필터링 적용

    return dst


def apply_sobel_filter(src):
    #####################################################
    # TODO                                              #
    # src에 sobel filter 적용                            #
    #####################################################
    x, y = get_sobel_mask()
    Ix = my_filtering(src, x)  # x축 sobel 필터링 적용
    Iy = my_filtering(src, y)  # y축 sobel 필터링 적용

    return Ix, Iy


def calc_magnitude(Ix, Iy):
    #####################################################
    # TODO                                              #
    # Ix, Iy 로부터 magnitude 계산                        #
    #####################################################
    magnitude = np.sqrt(Ix**2 + Iy**2)  # magnitude 계산

    return magnitude


def calc_angle(Ix, Iy, eps=1e-6):
    #####################################################
    # TODO                                              #
    # Ix, Iy 로부터 angle 계산                            #
    # numpy의 arctan 사용 O, arctan2 사용 X               #
    # 호도법이나 육십분법이나 상관 X                         #
    # eps     : Divide by zero 방지용                    #
    #####################################################

    # angle = np.arctan(Iy / (Ix + eps))  # Ix 가 0 일 수도 있기 때문에 eps 더해준다.
    # angle = np.rad2deg(angle)

    theta = np.arctan(Iy, Ix)
    angle = theta * (180 / np.pi)  # radian to degree

    return angle


def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # Non-maximum-supression 수행                                                       #
    # 스켈레톤 코드는 angle이 육십분법으로 나타나져 있을 것으로 가정                             #
    ####################################################################################
    (h, w) = magnitude.shape
    # angle의 범위 : -90 ~ 90
    largest_magnitude = np.zeros((h, w))
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            degree = angle[row, col]

            # if 0 <= degree < 45:
            #     rate = np.tan(np.deg2rad(degree))
            #     left_magnitude = (rate) * magnitude[row - 1, col - 1] + (1 - rate) * magnitude[row, col - 1]
            #     right_magnitude = (rate) * magnitude[row + 1, col + 1] + (1 - rate) * magnitude[row, col + 1]
            #     if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
            #         largest_magnitude[row, col] = magnitude[row, col]

            # 각도가 d일 때
            # d 각도의 픽셀과 동시에 180 + d 각도 방향의 픽셀과도 비교 해야함.
            # ex) 10도와 190도 -> 대략 우측과 좌측 픽셀
            # interpolation 방법은 linear로 구현

            if 0 <= degree < 45:
                if 0 <= degree < 22.5:  # deg 0 으로 nearest interpolation
                    m1 = magnitude[row, col + 1]
                    m2 = magnitude[row, col - 1]
                else:  # deg 45 으로 선형보간
                    m1 = magnitude[row + 1, col + 1]
                    m2 = magnitude[row - 1, col - 1]
            elif 45 <= degree <= 90:
                if 45 <= degree < 67.5:  # deg 45 으로 nearest interpolation
                    m1 = magnitude[row + 1, col + 1]
                    m2 = magnitude[row - 1, col - 1]
                else:  # deg 90 으로 선형보간
                    m1 = magnitude[row + 1, col]
                    m2 = magnitude[row - 1, col]
            elif -45 <= degree < 0:  # deg -45 으로 nearest interpolation
                if -45 <= degree < -22.5:
                    m1 = magnitude[row - 1, col + 1]
                    m2 = magnitude[row + 1, col - 1]
                else:  # deg 0 으로 선형보간
                    m1 = magnitude[row, col + 1]
                    m2 = magnitude[row, col - 1]
            elif -90 <= degree < -45:
                if -90 <= degree < -67.5:  # deg -90 으로 nearest interpolation
                    m1 = magnitude[row - 1, col]
                    m2 = magnitude[row + 1, col]
                else:  # deg -45 으로 선형보간
                    m1 = magnitude[row - 1, col + 1]
                    m2 = magnitude[row + 1, col - 1]

            else:
                print(row, col, 'error!  degree :', degree)

            if magnitude[row, col] >= m1 and magnitude[row, col] >= m2:  # 최대값 이 외 나머지 값 제거
                largest_magnitude[row, col] = magnitude[row, col]

    return largest_magnitude


def double_thresholding(src):
    dst = src.copy()

    # dst 범위 조정 0 ~ 255
    dst = (dst - np.min(dst)) / (np.max(dst) - np.min(dst))
    dst *= 255
    dst = dst.astype('uint8')

    # threshold는 정해진 값을 사용
    high_threshold_value = 40
    low_threshold_value = 5

    print(high_threshold_value, low_threshold_value)

    #####################################################
    # TODO                                              #
    # Double thresholding 수행                           #
    #####################################################

    STRONG_EDGE = 2
    WEEK_EDGE = 1
    NON_EDGE = 0
    (h, w) = dst.shape

    def tracking_hysteresis(trace, i, j):  # 재귀적으로 호출 되어 edge tracking 한다.
        h, w = trace.shape
        if i < 0 or i > h or j < 0 or j > w:  # base case : 가장자리에 도달
            return NON_EDGE
        else:  # recursive case : WEEK_EDGE, STRONG_EDGE, NON_EDGE 판별
            #  WEEK EDGE case
            if trace[i + 1, j - 1] == WEEK_EDGE:
                trace[i, j] = tracking_hysteresis(trace, i + 1, j - 1)
            elif trace[i + 1, j] == WEEK_EDGE:
                trace[i, j] = tracking_hysteresis(trace, i + 1, j)
            elif trace[i, j + 1] == WEEK_EDGE:
                trace[i, j] = tracking_hysteresis(trace, i, j + 1)
            elif trace[i + 1, j + 1] == WEEK_EDGE:
                trace[i, j] = tracking_hysteresis(trace, i + 1, j + 1)

            #  STRONG EDGE case
            elif trace[i + 1, j - 1] == STRONG_EDGE:
                trace[i, j] = STRONG_EDGE
            elif trace[i + 1, j] == STRONG_EDGE:
                trace[i, j] = STRONG_EDGE
            elif trace[i, j + 1] == STRONG_EDGE:
                trace[i, j] = STRONG_EDGE
            elif trace[i + 1, j + 1] == STRONG_EDGE:
                trace[i, j] = STRONG_EDGE

            else:  # NON_EDGE case
                trace[i, j] = NON_EDGE

            return trace[i, j]

    # threshold 를 기준으로 각 픽셀에 WEEK_EDGE, STRONG_EDGE, NON_EDGE 할당
    for row in range(h):
        for col in range(w):
            if dst[row, col] < low_threshold_value:
                dst[row, col] = NON_EDGE
            elif dst[row, col] > high_threshold_value:
                dst[row, col] = STRONG_EDGE
            else:
                dst[row, col] = WEEK_EDGE

    # WEEK_EDGE tracking
    for i in range(h):
        for j in range(w):
            if dst[i, j] == WEEK_EDGE:
                tracking_hysteresis(dst, i, j)

    # strong edge 제외 나머지 값 0 으로 초기화
    dst = np.where(dst == STRONG_EDGE, 255, 0)

    dst = dst.astype('float32') / 255.0
    return dst


def canny_edge_detection(src):
    # Apply low pass filter
    I = apply_gaussian_filter(src, fsize=3, sigma=1)

    # Apply high pass filter
    Ix, Iy = apply_sobel_filter(I)

    # Get magnitude and angle
    magnitude = calc_magnitude(Ix, Iy)
    angle = calc_angle(Ix, Iy)

    # Apply non-maximum-supression
    after_nms = non_maximum_supression(magnitude, angle)

    # Apply double thresholding
    dst = double_thresholding(after_nms)

    return dst, after_nms, magnitude


if __name__ == '__main__':
    img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0

    canny, after_nms, magnitude = canny_edge_detection(img)

    # 시각화 하기 위해 0~1로 normalize (min-max scaling)
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
    after_nms = (after_nms - np.min(after_nms)) / (np.max(after_nms) - np.min(after_nms))

    cv2.imshow('original', img)
    cv2.imshow('magnitude', magnitude)
    cv2.imshow('after_nms', after_nms)
    cv2.imshow('canny_edge', canny)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

