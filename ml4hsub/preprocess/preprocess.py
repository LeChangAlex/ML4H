import pydicom
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import exposure, restoration, transform
from skimage.io import imread, imsave
import glob
from matplotlib import pyplot as plt
import SimpleITK as sitk
import pickle


import pydicom
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import exposure, restoration, transform
from skimage.io import imread, imsave
import glob
from matplotlib import pyplot as plt
import SimpleITK as sitk
import pickle


# BAD = []


def remove_clips(im):

    im[im == 0] = np.nan
    im[im == 1] = np.nan

    im_pd = pd.DataFrame(data=im)
    im = im_pd.interpolate(method='linear')

    return np.array(im)

def correct_background(im):
    # select background columns

    sides = np.concatenate((im[:, 0:30], im[:, -30:]), axis=1)
    blur = cv2.GaussianBlur(sides, (7, 7), 0)

    row_background = np.mean(blur, axis=1)

    im -= row_background[..., np.newaxis]


    return im



def preprocess(im):

    # if np.sum(im[:,:50]) > 20:
        # im[:, :50] = remove_clips(im[:, :50]) # left clips
        # im[:, :50] = remove_clips(im[:, :50]) # left clips
        # im[:, -50:] = remove_clips(im[:, -50:]) # right clips
    im = remove_clips(im)


    im[np.where(np.isnan(im))] = 0

    if np.sum(im[:, 0:30]) < 100:
        im = correct_background(im) # normalize band brightness

    kernel_width = im.shape[1] // 8

    # im = exposure.equalize_adapthist(im, nbins=256, kernel_size=(kernel_width, kernel_width))   # < GOOD

    # im = restoration.denoise_bilateral(im, multichannel=False)
    # im = restoration.denoise_wavelet(im)  # < GOOD

    return im


#
#
# def correct_bias_field(volume):
#     corrector = sitk.N4BiasFieldCorrectionImageFilter()
#     for i in range(8):
#         volume_image = sitk.GetImageFromArray(volume[i])
#         mask_image = sitk.OtsuThreshold( volume_image, 0, 1, 200 )
#         corrected_image = corrector.Execute(volume_image,mask_image)
#         volume[i] = sitk.GetArrayFromImage(corrected_image)
#     return volume


def normalize(im):
    cutoff = np.percentile(im, 99.995)
    im[im > cutoff] = cutoff
    im = (im - min(im.flatten())) / (max(im.flatten()) - min(im.flatten()))
    return im


# def crop_coords(image):
#     blur = cv2.GaussianBlur(image, (7,7), 0)
#     blur = cv2.GaussianBlur(blur, (63,63), 0)
#
#     points = np.argwhere(blur>0.15)
#     if len(points) == 0:
#         return 0, image.shape[0], 0, image.shape[1]
#     min_x = min(points, key=lambda im:im[0])[0]
#     max_x = max(points, key=lambda im:im[0])[0]
#     min_y = min(points, key=lambda im:im[1])[1]
#     max_y = max(points, key=lambda im:im[1])[1]
#     ratio = (max_y-min_y)/(max_x-min_x)
#     pad_y = 70
#     if ratio < 0.21:
#         pad_y += int(720*abs(ratio-0.21))
#     else:
#         pad_y -= 30
#     min_y = max(0, min_y-pad_y)
#     max_y = min(image.shape[1], max_y+pad_y)
#     pad_x = 70
#     if ratio > 0.24:
#         pad_x += int(720*abs(ratio-0.24))
#     else:
#            pad_x -= 40
#     min_x = max(0, min_x-pad_x)
#     max_x = min(image.shape[0], max_x+pad_x)
#
#     return min_x, max_x, min_y, max_y

def main():
    mri_directory = '/data/lechang/new_wbmri_wb'

    out_directory = '/data/lechang/new_wbmri_wb_preproc2'

    for n in tqdm(range(218, 550)):

        fn = "volume_{}.npy".format(n)
        try:
            arr = np.load(os.path.join(mri_directory, fn))
        except:
            continue

        for i in range(len(arr)):

            im = arr[i]

            # cutoff = np.percentile(im, 99.995)
            # if cutoff == 0:
            #     continue
            # im[im > cutoff] = cutoff

            # im = normalize(im)
            im = preprocess(im)

            arr[i] = im
        arr = normalize(arr)
        np.save(os.path.join(out_directory, fn), arr)

if __name__ == "__main__":
    main()