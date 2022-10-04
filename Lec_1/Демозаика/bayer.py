import numpy as np


def get_bayer_masks(n_rows, n_cols):
    n_rows_2 = int((n_rows + (n_rows % 2)) / 2 + 1)
    n_cols_2 = int((n_cols + (n_cols % 2)) / 2 + 1)

    red = np.array([0, 1, 0, 0], 'bool').reshape(2, 2)
    green = np.array([1, 0, 0, 1], 'bool').reshape(2, 2)
    blue = np.array([0, 0, 1, 0], 'bool').reshape(2, 2)

    red = np.tile(red, [n_rows_2, n_cols_2])[:n_rows, :n_cols]
    green = np.tile(green, [n_rows_2, n_cols_2])[:n_rows, :n_cols]
    blue = np.tile(blue, [n_rows_2, n_cols_2])[:n_rows, :n_cols]

    a = np.dstack((red, green, blue))
    return a

def get_colored_img(raw_img):
    a = get_bayer_masks(len(raw_img), len(raw_img[0]))
    a = np.dstack((a[...,0]*raw_img, a[...,1]*raw_img, a[...,2]*raw_img))
    return a


def bilinear_interpolation(colored_img):
    h, w = colored_img[..., 0].shape
    res_matr = colored_img * 0

    for i in range(3):
        res = colored_img[..., i].copy()
        bayer = get_bayer_masks(h, w)[..., i]

        for j in range(1, h - 1):
            for k in range(1, w - 1):
                if bayer[j, k] == 0:
                    colored_img[..., i][j, k] = np.sum(res[j - 1:j + 2, k - 1:k + 2]) / np.count_nonzero(
                        bayer[j - 1:j + 2, k - 1:k + 2])

    res_matr[1:h - 1, 1:(w - 1)] = colored_img[1:h - 1, 1:(w - 1)]

    return res_matr


def compute_psnr(img_pred, img_gt):
    mask, h, w = img_pred.shape
    mse = np.sum((img_pred.astype(int) - img_gt.astype(int))**2) / (mask * h * w)
    if(mse == 0):
        raise ValueError
    else:
        return 10 * np.log10(np.max(img_gt)**2 / mse)