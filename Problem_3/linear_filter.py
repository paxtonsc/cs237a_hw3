#!/usr/bin/env python3

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from numpy.lib.arraypad import pad


def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    ########## Code starts here ##########
    m, n, c = I.shape
    k, l, _ = F.shape

    I_bar = np.zeros((m+k-1, n+l-1, c))

    pad_width = int((k-1)//2)
    I_bar[pad_width:-pad_width, pad_width:-pad_width,:] = I

    G = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            conv = I_bar[i:i+k,j:j+l,:]*F
            G[i,j] = np.sum(conv)

    ########## Code ends here ##########

    return G


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########


    k, l, c = F.shape
    m, n, _ = I.shape

    I_bar = np.zeros((m+k-1, n+l-1, c))

    pad_width = int((k-1)//2)
    I_bar[pad_width:-pad_width, pad_width:-pad_width,:] = I

    G = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            t_ij = I_bar[i:i+k,j:j+l,:]
            conv = F*t_ij
            G[i,j] = np.sum(conv) / (np.linalg.norm(t_ij.reshape(-1))*np.linalg.norm(F.reshape(-1)))


    ########## Code ends here ##########


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():

    #paxFilter = (1/16)*np.array([[1, 2, 1], 
    #                            [2, 4, 2],
    #                            [1, 2, 1]])
    #I = np.array([[1, 2, 3],
    #            [4, 5, 6],
    #            [7, 8, 9]])
    #print(corr(paxFilter.reshape(*paxFilter.shape, -1), I.reshape(*I.shape, -1)))

    test_card = cv2.imread('test_card.png').astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 3, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = corr(filt, test_card)
        stop = time.time()
        print('Correlation function runtime:', stop - start, 's')
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
