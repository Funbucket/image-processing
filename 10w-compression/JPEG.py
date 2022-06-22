import math
import numpy as np
import cv2
import time


def Quantization_Luminance(scale_factor):
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance * scale_factor


def img2block(src, n=8):
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    # blocks : 각 block을 추가한 list       #
    # return value : np.array(blocks)     #
    ######################################
    (h, w) = src.shape

    # if h % n != 0:
    #     h_pad = n - h%n
    # else:
    #     h_pad = 0
    #
    # if w % n != 0:
    #     w_pad = n - w%n
    # else:
    #     w_pad = 0
    #
    # dst = np.zeros((h+h_pad, w+w_pad))
    # dst[:h, :w] = src
    #
    # blocks = []
    # for row in range((h+h_pad)//n):
    #     for col in range((w+w_pad)//n):
    #         block = dst[row*n:(row+1)*n, col*n:(col+1)*n]
    #         blocks.append(block)

    blocks = list()

    for row in range(0, h, n):
        for col in range(0, w, n):
            blocks.append(np.double(src[row:row + n, col:col + n]))

    return np.array(blocks)


def DCT(block, n=8):
    ######################################
    # TODO                               #
    # DCT 완성                            #
    ######################################
    def C(w, n):
        if w == 0:
            return (1 / n) ** 0.5
        else:
            return (2 / n) ** 0.5

    dst = np.zeros(block.shape)

    v, u = dst.shape
    y, x = np.mgrid[0:u, 0:v]

    for v_ in range(v):
        for u_ in range(u):
            mask = np.cos(((2 * x + 1) * u_ * np.pi) / (2 * n)) * np.cos(((2 * y + 1) * v_ * np.pi) / (2 * n))
            val = block * mask
            val = C(u_, n) * C(v_, n) * np.sum(val)
            dst[v_, u_] = val

    return np.round(dst)


def my_zigzag_encoding(block, block_size=8):
    ######################################
    # TODO                               #
    # my_zigzag_encoding 완성             #
    ######################################
    zigzag = list()
    h, w = block.shape
    ZF = [0, True]  # index, Zero flag

    tmp = [[] for i in range(h + w - 1)]

    for row in range(h):
        for col in range(w):
            diagonal = row + col
            if (diagonal % 2) == 0:
                tmp[diagonal].insert(0, block[row][col])
            else:
                tmp[diagonal].append(block[row][col])
    i = 0
    for a in tmp:
        for j in a:
            if not ZF[1] and j == 0:
                ZF[0] = i
                ZF[1] = True
            elif ZF[1] and j == 0:
                pass
            else:
                ZF[1] = False
            zigzag.append(j)
            i += 1

    if ZF[1]:
        zigzag = zigzag[:ZF[0]]
        zigzag.append("EOB")

    return zigzag


def my_zigzag_decoding(block, block_size=8):
    ######################################
    # TODO                               #
    # my_zigzag_decoding 완성             #
    ######################################

    EOBF = [0, False]  # index, EOB flag
    length = block_size ** 2
    re_block = np.zeros((block_size, block_size))
    row = 0
    col = 0
    index = 0

    for i, b in enumerate(block):
        if b == "EOB":
            EOBF[0] = i
            EOBF[1] = True
            break

    if EOBF[0]:
        zz = block[:EOBF[0]] + ([0] * (length - EOBF[0]))
    else:
        zz = block

    while (row < block_size) and (col < block_size):
        if ((col + row) % 2) == 0:
            if row == 0:
                re_block[row, col] = zz[i]
                if col == block_size:
                    row = row + 1
                else:
                    col = col + 1
                index = index + 1
            elif (col == block_size - 1) and (row < block_size):
                re_block[row, col] = zz[index]
                row = row + 1
                index = index + 1
            elif (row > 0) and (col < block_size - 1):
                re_block[row, col] = zz[index]
                row = row - 1
                col = col + 1
                index = index + 1
        else:
            if (row == block_size - 1) and (col <= block_size - 1):
                re_block[row, col] = zz[index]
                col = col + 1
                index = index + 1
            elif col == 0:
                re_block[row, col] = zz[index]
                if row == block_size - 1:
                    col = col + 1
                else:
                    row = row + 1
                index = index + 1
            elif (row < block_size - 1) and (col > 0):
                re_block[row, col] = zz[index]
                row = row + 1
                col = col - 1
                index = index + 1
        if (row == block_size - 1) and (col == block_size - 1):
            re_block[row, col] = zz[index]
            break

    return re_block


def DCT_inv(block, n=8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################
    def C(w, n):
        if w == 0:
            return (1 / n) ** 0.5
        else:
            return (2 / n) ** 0.5

    dst = np.zeros(block.shape)

    x, y = dst.shape

    for x_ in range(x):
        for y_ in range(y):
            sum = 0
            for u_ in range(x):
                for v_ in range(y):
                    dct = block[u_][v_] * math.cos((2 * x_ + 1) * u_ * math.pi / (2 * n)) * math.cos(
                        (2 * y_ + 1) * v_ * math.pi / (2 * n)) * C(u_, n) * C(v_, n)
                    sum += dct
            dst[x_][y_] = sum

    return np.round(dst)


def block2img(blocks, src_shape, n=8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################
    dst = np.zeros(src_shape)
    h, w = dst.shape

    i = 0
    for row in range(0, h, n):
        for col in range(0, w, n):
            dst[row:row+n, col:col+n] = blocks[i]
            i += 1

    return dst


def Encoding(src, n=8, scale_factor=1):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)
    print("block = \n", src[150:158, 89:97])

    # subtract 128
    blocks -= 128
    b = np.double(src[150:158, 89:97]) - 128
    print("b = \n", b)

    # DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)

    # print DCT results
    bd = DCT(b, n=8)
    print("bd = \n", bd)

    # Quantization + thresholding
    Q = Quantization_Luminance(scale_factor)
    QnT = np.round(blocks_dct / Q)
    # print Quantization results
    bq = np.round(bd / Q)
    print("bq = \n", bq)

    # zigzag scanning
    zz = []
    for i in range(len(QnT)):
        zigzag = my_zigzag_encoding(QnT[i], block_size=n)
        zz.append(zigzag)
    return zz, src.shape, bq


def Decoding(zigzag, src_shape, bq, n=8, scale_factor=1):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        re_block = my_zigzag_decoding(zigzag[i], block_size=n)
        blocks.append(re_block)
    blocks = np.array(blocks)

    # Denormalizing
    Q = Quantization_Luminance(scale_factor=scale_factor)
    blocks = blocks * Q
    # print results Block * Q
    bq2 = bq * Q
    print("bq2 = \n", bq2)

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)

    # print IDCT results
    bd2 = DCT_inv(bq2, n=8)
    print("bd2 = \n", bd2)

    # add 128
    blocks_idct += 128

    # print block value
    b2 = bd2 + 128
    print("b2 = \n", b2)

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst, b2


def main():
    scale_factor = 1
    start = time.time()
    # src = cv2.imread('../imgs/Lenna.png', cv2.IMREAD_GRAYSCALE)
    src = cv2.imread('caribou.tif', cv2.IMREAD_GRAYSCALE)

    comp, src_shape, bq = Encoding(src, n=8, scale_factor=scale_factor)
    np.save('comp.npy', comp)
    np.save('src_shape.npy', src_shape)
    # print(comp)
    comp = np.load('comp.npy', allow_pickle=True)
    src_shape = np.load('src_shape.npy')
    recover_img, b2 = Decoding(comp, src_shape, bq, n=8, scale_factor=scale_factor)
    print("scale_factor : ", scale_factor, "differences between original and reconstructed = \n",
          src[150:158, 89:97] - b2)

    total_time = time.time() - start
    #
    print('time : ', total_time)
    if total_time > 12:
        print('감점 예정입니다.')
    print(recover_img.shape)
    cv2.imshow('original image', src)
    cv2.imshow('recover img', recover_img/255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
