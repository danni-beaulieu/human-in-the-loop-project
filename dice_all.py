
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from matplotlib import pyplot as plt, cm
import os
import seaborn as sns
from sklearn.metrics import f1_score


image_nib = nib.load("/Users/daniellemiller/WUSTL/518A/SegProject/Subject/BRATS_030.nii").get_fdata()[:, :, :]

path_masks = '/Users/daniellemiller/WUSTL/518A/SegProject/Labels/'
list_masks_all = os.listdir(path_masks)
list_masks = [i for i in list_masks_all if i.endswith('.nii')]

df_scores = pd.DataFrame(columns=['File', 'Score'])

ground_truth_mask = sitk.GetArrayFromImage(
    sitk.Cast(sitk.ReadImage("/Users/daniellemiller/WUSTL/518A/SegProject/Truth/BRATS_030.nii"), sitk.sitkUInt16))[
                    80:91, :, :]
ground_truth_mask = 1.0 * (ground_truth_mask > 0)


for i in range(len(list_masks)):

    segmentation = sitk.Cast(sitk.ReadImage(path_masks + list_masks[i]), sitk.sitkUInt16)
    output_mask = sitk.GetArrayFromImage(segmentation)
    output_mask = 1.0 * (output_mask > 0.5)

    dice_sc = f1_score(ground_truth_mask.flatten(), output_mask.flatten())

    new_score = {'File': list_masks[i], 'Score': dice_sc}
    df_scores = pd.concat([df_scores, pd.DataFrame([new_score])], ignore_index=True)



y = df_scores['Score'].tolist()

sns.distplot(y, hist=True, kde=True,
             bins=50,
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             label='DICE Score')
plt.title("DICE Score Distribution")
plt.legend()
plt.show()

