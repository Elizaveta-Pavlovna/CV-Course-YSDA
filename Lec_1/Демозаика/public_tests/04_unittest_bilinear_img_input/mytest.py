from bayer import bilinear_interpolation, get_colored_img
from common import assert_ndarray_equal
from glob import glob
from numpy import abs
from os.path import abspath, basename, dirname, join
from skimage import img_as_ubyte
from skimage.io import imread, imsave

test_dir = dirname(abspath(__file__))
for img_filename in sorted(glob(join(test_dir, '01.png'))):
    raw_img = img_as_ubyte(imread(join(test_dir, img_filename)))
    img = img_as_ubyte(bilinear_interpolation(get_colored_img(raw_img)))
    out_filename = join(test_dir, 'gt_' + basename(img_filename))
    gt_img = img_as_ubyte(imread(out_filename))
    r = slice(1, -1), slice(1, -1)


print(img[1], '\n\n\n', gt_img[1])
