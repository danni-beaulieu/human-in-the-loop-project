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
import random as rand
from scipy.spatial.distance import directed_hausdorff

image_nib = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Subject/BRATS_030.nii").get_fdata()[:,:,:]

#path_masks = 'C:/Users/kaushik.dutta/Box/Brain Tumor Study/Labels/'
path_masks = 'C:/Users/kaushik.dutta/Box/Brain Tumor Study/Label_Ratings/'
list_masks = os.listdir(path_masks)
total_block = np.zeros((240,240,11,len(list_masks)))
for i in range(0,len(list_masks)):
    image = nib.load(path_masks+list_masks[i]).get_fdata()[:,:,:]
    total_block[:,:,:,i] = image
    
ground_truth_mask = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/BRATS_030.nii").get_fdata()[:,:,80:91]
ground_truth_mask = 1.0*(ground_truth_mask>0)

### Implementation of the STAPLE Algorithm alongwith applying random sampling ######
sample_indices = np.arange(0,total_block.shape[3],1).tolist()
max_iter = 100
voting_dice_score_final = []
voting_dice_std = []
staple_dice_score_final = []
staple_dice_std = []

voting_hausdorf_final = []
staple_hausdorf_final = []


def ties(x):
    if x == 10:
        return rand.choice([0, 1])
    else:
        return x
    
def hausdorff_dist(img1,img2):
    no_of_slices = img1.shape[2]
    haus = []
    for i in range(0,no_of_slices):
        haus_temp = directed_hausdorff(img1[:,:,i], img2[:,:,i])
        haus.append(haus_temp)
    haus_final = np.mean(haus)
    return haus_final
        

fix_ties = np.vectorize(ties)

dice_score_final = []
dice_std = []
for i in range(0,len(sample_indices)):
    staple_dice_temp = []
    voting_dice_temp = []
    staple_haus_temp = []
    voting_haus_temp = []
    
    for j in range(0,max_iter):
        
        samples = sample(sample_indices,i+1)
        final_img_stack = []
        final_img_stack_voting = []
        for k in range(0,len(samples)):
            crowd_img = total_block[:,:,:,samples[k]]
            crowd_img_sitk = sitk.GetImageFromArray(crowd_img.astype(np.int16))
            crowd_img_sitk_voting = sitk.Cast(crowd_img_sitk, sitk.sitkUInt16)
            final_img_stack.append(crowd_img_sitk)
            final_img_stack_voting.append(crowd_img_sitk_voting)
            
        staple_output_sitk = sitk.STAPLE(final_img_stack,1.0)
        staple_output_mask = sitk.GetArrayFromImage(staple_output_sitk)
        staple_output_mask = 1.0*(staple_output_mask>0.5)
        staple_output_mask_final = 1.0*(staple_output_mask>0)
        staple_dice_sc = f1_score(ground_truth_mask.flatten(), staple_output_mask_final.flatten())
        staple_dice_temp.append(staple_dice_sc)
        staple_haus_sc = hausdorff_dist(ground_truth_mask, staple_output_mask_final)
        staple_haus_temp.append(staple_haus_sc)
        
        voting_output_sitk = sitk.LabelVoting(final_img_stack_voting, 10)
        voting_output_mask = sitk.GetArrayFromImage(voting_output_sitk)
        voting_output_mask = fix_ties(voting_output_mask)
        voting_output_mask = 1.0 * (voting_output_mask > 0.5)
        voting_dice_sc = f1_score(ground_truth_mask.flatten(), voting_output_mask.flatten())
        voting_dice_temp.append(voting_dice_sc)
        voting_haus_sc = hausdorff_dist(ground_truth_mask, voting_output_mask)
        voting_haus_temp.append(voting_haus_sc)
        
    voting_dice_score_avg = np.mean(voting_dice_temp)
    staple_dice_score_avg = np.mean(staple_dice_temp)
    voting_dice_score_final.append(voting_dice_score_avg)
    staple_dice_score_final.append(staple_dice_score_avg)
    voting_dice_std.append(np.std(voting_dice_temp))
    staple_dice_std.append(np.std(staple_dice_temp))
    
    voting_hausdorf_final.append(np.mean(voting_haus_temp))
    staple_hausdorf_final.append(np.mean(staple_haus_temp))

fig, ax = plt.subplots(figsize =(10, 10), dpi = 300)  
ax.plot(voting_dice_score_final, label="Majority Vote")
plt.plot(staple_dice_score_final, label="STAPLE")
ax.set_title('Segmentations Vs DICE Score', fontsize = 30)
ax.set_xlabel('Number of Labellers', fontsize = 25)
ax.set_ylabel('Dice Score', fontsize = 25)
leg = ax.legend(loc="lower right",fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=25)
plt.show()

fig, ax = plt.subplots(figsize =(10, 10), dpi = 300)  
ax.plot(voting_hausdorf_final, label="Majority Vote")
plt.plot(staple_hausdorf_final, label="STAPLE")
ax.set_title('Segmentations Vs Hausdorff Distance', fontsize = 30)
ax.set_xlabel('Number of Labellers', fontsize = 25)
ax.set_ylabel('Hausdorff Score', fontsize = 25)
leg = ax.legend(loc="lower right", fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=25)
plt.show()

