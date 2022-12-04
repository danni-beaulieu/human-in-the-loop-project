
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

staple_dice_temp = []
voting_dice_temp = []
final_img_stack = []

for k in range(0, len(segmentations)):
    crowd_img = segmentations[k]
    final_img_stack.append(crowd_img)

voting_output_sitk = sitk.LabelVoting(final_img_stack, 10)
voting_output_mask = sitk.GetArrayFromImage(voting_output_sitk)
voting_output_mask = fix_ties(voting_output_mask)
voting_output_mask = 1.0 * (voting_output_mask > 0.5)

staple_output_sitk = sitk.STAPLE(final_img_stack, 1.0)
staple_output_mask = sitk.GetArrayFromImage(staple_output_sitk)
staple_output_mask = 1.0 * (staple_output_mask > 0.5)

plt.imshow(ground_truth_mask[6,:,:], cmap='Pastel1', interpolation='none')
plt.imshow(voting_output_mask[6,:,:], cmap='Blues', interpolation='none', alpha=0.25)
plt.imshow(staple_output_mask[6,:,:], cmap='Set3', interpolation='none', alpha=0.25)
plt.title("Truth vs Majority Vote (Blue) vs STAPLE (Yellow)")
plt.show()
