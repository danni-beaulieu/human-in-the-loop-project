# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:21:18 2022

@author: kaushik.dutta
"""

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from matplotlib import pyplot as plt
import os
from random import sample
from sklearn.metrics import f1_score

image_nib = nib.load("/Users/daniellemiller/WUSTL/518A/SegProject/Subject/BRATS_030.nii").get_fdata()[:,:,:]

path_masks = '/Users/daniellemiller/WUSTL/518A/SegProject/Labels/'
list_masks_all = os.listdir(path_masks)
list_masks = [i for i in list_masks_all if i.endswith('.nii')]
total_block = np.zeros((240,240,11,len(list_masks)))
for i in range(0,len(list_masks)):
    image = nib.load(path_masks+list_masks[i]).get_fdata()[:,:,:]
    total_block[:,:,:,i] = image
    
ground_truth_mask = nib.load("/Users/daniellemiller/WUSTL/518A/SegProject/BRATS_030.nii").get_fdata()[:,:,80:91]
ground_truth_mask = 1.0*(ground_truth_mask>0)

### Implementation of the STAPLE Algorithm alongwith applying random sampling ######
sample_indices = np.arange(0,total_block.shape[3],1).tolist()
max_iter = 50

dice_score_final = []
dice_std = []
for i in range(0,len(sample_indices)):
    
    for j in range(0,max_iter):
        dice_temp = []
        samples = sample(sample_indices,i+1)
        final_img_stack = []
        for k in range(0,len(samples)):
            crowd_img = total_block[:,:,:,samples[k]]
            crowd_img_sitk = sitk.GetImageFromArray(crowd_img.astype(np.int16))
            final_img_stack.append(crowd_img_sitk)
            
        staple_output_sitk = sitk.STAPLE(final_img_stack,1.0)
        staple_output_mask = sitk.GetArrayFromImage(staple_output_sitk)
        staple_output_mask = 1.0*(staple_output_mask>0.5)
        staple_output_mask_final = 1.0*(staple_output_mask>0)
        dice_sc = f1_score(ground_truth_mask.flatten(), staple_output_mask_final.flatten())
        dice_temp.append(dice_sc)
        
    dice_score_fin = np.mean(dice_temp)
    dice_score_final.append(dice_score_fin)
    dice_std.append(np.std(dice_temp))
    
plt.figure()
plt.plot(dice_score_final)
plt.show()
            
            