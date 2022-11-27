"""
@author: Danni Beaulieu
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from matplotlib import pyplot as plt
import os
from random import sample
import random as rand
from sklearn.metrics import f1_score

image_nib = nib.load("/Users/daniellemiller/WUSTL/518A/SegProject/Subject/BRATS_030.nii").get_fdata()[:, :, :]

path_masks = '/Users/daniellemiller/WUSTL/518A/SegProject/Labels/'
list_masks_all = os.listdir(path_masks)
list_masks = [i for i in list_masks_all if i.endswith('.nii')]

segmentations = [sitk.Cast(sitk.ReadImage(path_masks + mask_name), sitk.sitkUInt16) for mask_name in list_masks]

ground_truth_mask = sitk.GetArrayFromImage(
    sitk.Cast(sitk.ReadImage("/Users/daniellemiller/WUSTL/518A/SegProject/Truth/BRATS_030.nii"), sitk.sitkUInt16))[
                    80:91, :, :]
ground_truth_mask = 1.0 * (ground_truth_mask > 0)

### Implementation of the STAPLE Algorithm alongwith applying random sampling ######
sample_indices = np.arange(0, len(segmentations), 1).tolist()
max_iter = 50

voting_dice_score_final = []
voting_dice_std = []
staple_dice_score_final = []
staple_dice_std = []


def ties(x):
    if x == 10:
        return rand.choice([0, 1])
    else:
        return x


fix_ties = np.vectorize(ties)
for i in range(0, len(sample_indices)):

    for j in range(0, max_iter):
        staple_dice_temp = []
        voting_dice_temp = []
        samples = sample(sample_indices, i + 1)
        final_img_stack = []

        for k in range(0, len(samples)):
            crowd_img = segmentations[samples[k]]
            final_img_stack.append(crowd_img)

        voting_output_sitk = sitk.LabelVoting(final_img_stack, 10)
        voting_output_mask = sitk.GetArrayFromImage(voting_output_sitk)
        voting_output_mask = fix_ties(voting_output_mask)
        voting_output_mask = 1.0 * (voting_output_mask > 0.5)
        voting_dice_sc = f1_score(ground_truth_mask.flatten(), voting_output_mask.flatten())
        voting_dice_temp.append(voting_dice_sc)

        staple_output_sitk = sitk.STAPLE(final_img_stack, 1.0)
        staple_output_mask = sitk.GetArrayFromImage(staple_output_sitk)
        staple_output_mask = 1.0 * (staple_output_mask > 0.5)
        staple_dice_sc = f1_score(ground_truth_mask.flatten(), staple_output_mask.flatten())
        staple_dice_temp.append(staple_dice_sc)

    voting_dice_score_avg = np.mean(voting_dice_temp)
    staple_dice_score_avg = np.mean(staple_dice_temp)
    voting_dice_score_final.append(voting_dice_score_avg)
    staple_dice_score_final.append(staple_dice_score_avg)
    voting_dice_std.append(np.std(voting_dice_temp))
    staple_dice_std.append(np.std(staple_dice_temp))

plt.plot(voting_dice_score_final, label="Majority Vote")
plt.plot(staple_dice_score_final, label="STAPLE")
plt.legend(loc="lower right")
plt.title("Segmentations Vs DICE Score")
plt.xlabel("Segmentations per Subject")
plt.ylabel("DICE Score")
plt.show()
