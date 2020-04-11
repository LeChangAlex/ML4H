import numpy as np
import argparse
import imutils
import glob
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from skimage import exposure
from skimage.exposure import match_histograms

template = np.load("/data/lechang/new_wbmri_wb_preproc/volume_103.npy")[31, 393:987, 140:607].astype(np.float32)
cv2.imwrite("chest_template.png", template * 255)
template_volume = np.load("/data/lechang/new_wbmri_wb_preproc/volume_103.npy").astype(np.float32)

for i in tqdm(range(550)):
    try:
        volume = np.load("/data/lechang/new_wbmri_wb_preproc2/volume_{}.npy".format(i)).astype(np.float32)
    except:
        continue
    found = None
    for j in range(volume.shape[0]):
        wbmri = match_histograms(volume[j], template_volume[31], multichannel=False).astype(np.float32)
        # wbmri = volume[j]
        # plt.imshow(wbmri)
        # plt.show()

        (tH, tW) = template.shape[:2]
        r = wbmri.shape[1] / tW


        # loop over the scales of the image
        for scale in np.linspace(0.4 * r, 0.75 * r, 10)[::-1]:
            resized = imutils.resize(template, width=int(template.shape[1] * scale))
            try:
                result = cv2.matchTemplate(wbmri, resized, cv2.TM_CCOEFF_NORMED)
            except:
                print("fail")
                continue


            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            # check to see if the iteration should be visualized
            # draw a bounding box around the detected region

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, scale, j, resized)
                # print(found[:4])
    # plt.imshow(found[4])
    # plt.show()


    # fig, ax = plt.subplots(1)
    #
    # # Display the image
    # ax.imshow(volume[found[3]])
    #
    # # Create a Rectangle patch
    # rect = patches.Rectangle((found[1][0], found[1][1]), template.shape[1] * found[2], template.shape[0] * found[2], linewidth=2, edgecolor='r', facecolor='none')
    #
    # # Add the patch to the Axes
    # ax.add_patch(rect)
    # plt.show()

    cv2.imwrite("new_wbmri/chest_wb/im_{}.png".format(i), volume[found[3],
                                                       found[1][1]: found[1][1] + found[4].shape[0],
                                                       found[1][0]: found[1][0] + found[4].shape[