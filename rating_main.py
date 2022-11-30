
import pandas as pd
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from matplotlib import pyplot as plt, cm
import os
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import f1_score


image_nib = nib.load("/Users/daniellemiller/WUSTL/518A/SegProject/Subject/BRATS_030.nii").get_fdata()[:, :, :]

path_masks = '/Users/daniellemiller/WUSTL/518A/SegProject/Labels/'
list_masks_all = os.listdir(path_masks)
list_masks = [i for i in list_masks_all if i.endswith('.nii')]

df_ratings = pd.DataFrame(columns=['File', 'Rating', 'Score'])

ground_truth_mask = sitk.GetArrayFromImage(
    sitk.Cast(sitk.ReadImage("/Users/daniellemiller/WUSTL/518A/SegProject/Truth/BRATS_030.nii"), sitk.sitkUInt16))[
                    80:91, :, :]
ground_truth_mask = 1.0 * (ground_truth_mask > 0)

df = pd.read_csv('/Users/daniellemiller/WUSTL/518A/SegProject/Rating/DifficultyRating.csv')


for i in range(df.shape[0]):

    mask_name = df.iloc[i].loc['File']
    rating = df.iloc[i].loc['Difficulty']

    segmentation = sitk.Cast(sitk.ReadImage(path_masks + mask_name), sitk.sitkUInt16)
    output_mask = sitk.GetArrayFromImage(segmentation)
    output_mask = 1.0 * (output_mask > 0.5)

    dice_sc = f1_score(ground_truth_mask.flatten(), output_mask.flatten())

    new_rating = {'File': mask_name, 'Rating': rating, 'Score': dice_sc}
    df_ratings = pd.concat([df_ratings, pd.DataFrame([new_rating])], ignore_index=True)


rating_plot = df_ratings.plot.scatter(x='Rating', y='Score', xticks=[1,2,3,4,5], s=500, alpha=0.33)
plt.title("Difficulty Rating Vs DICE Score")
plt.xlabel("Difficulty Rating")
plt.ylabel("DICE Score")
plt.show()


x = df_ratings['Rating'].tolist()
y = df_ratings['Score'].tolist()
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
cax = ax.scatter(x, y, c=z, s=500)
clb = plt.colorbar(cax)
clb.ax.set_title('Spatial Density')
plt.title("Difficulty Rating Vs DICE Score")
plt.xlabel("Difficulty Rating")
plt.ylabel("DICE Score")
plt.show()

sns.distplot(y, hist=True, kde=True,
             bins=50,
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             label='DICE Score')
plt.title("DICE Score Distribution")
plt.legend()
plt.show()

