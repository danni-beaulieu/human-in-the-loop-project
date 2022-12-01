# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:47:28 2022

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
import SimpleITK as sitk

################# Few User Defined Functions ###############

###### Function to break Ties ########
def ties(x):
    if x == 10:
        return rand.choice([0, 1])
    else:
        return x
    
fix_ties = np.vectorize(ties)

########### Function to implement STAPLE ###########
def calc_STAPLE(image1, image2):
    final_img_staple = []
    final_img_mv = []
    crowd_img_sitk_1 = sitk.GetImageFromArray(image1.astype(np.int16))
    crowd_img_sitk_voting_1 = sitk.Cast(crowd_img_sitk_1, sitk.sitkUInt16)
    crowd_img_sitk_2 = sitk.GetImageFromArray(image2.astype(np.int16))
    crowd_img_sitk_voting_2 = sitk.Cast(crowd_img_sitk_2, sitk.sitkUInt16)
    
    final_img_staple.append(crowd_img_sitk_1)
    final_img_staple.append(crowd_img_sitk_2)
    final_img_mv.append(crowd_img_sitk_voting_1)
    final_img_mv.append(crowd_img_sitk_voting_2)
    
    staple_output_sitk = sitk.STAPLE(final_img_staple,1.0)
    staple_output_mask = sitk.GetArrayFromImage(staple_output_sitk)
    staple_output_mask = 1.0*(staple_output_mask>0.5)
    staple_output_mask_final = 1.0*(staple_output_mask>0)
    
    voting_output_sitk = sitk.LabelVoting(final_img_mv, 10)
    voting_output_mask = sitk.GetArrayFromImage(voting_output_sitk)
    voting_output_mask = fix_ties(voting_output_mask)
    voting_output_mask = 1.0 * (voting_output_mask > 0.5)
    
    return staple_output_mask_final, voting_output_mask



############## Read the Individual Segmentation ################
segmentation_kd_1 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/Kaushik/maskBRATS_012.nii").get_fdata()[:,:,:]
segmentation_kd_2 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/Kaushik/maskBRATS_018.nii").get_fdata()[:,:,:]
segmentation_kd = np.concatenate((segmentation_kd_1[:,:,:,0], segmentation_kd_2[:,:,:,0]), axis = 2)
segmentation_kd = 1.0*(segmentation_kd>0)

segmentation_db_1 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/Danni/maskBRATS_012_DB.nii").get_fdata()[:,:,:]
segmentation_db_2 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/Danni/maskBRATS_018_DB.nii").get_fdata()[:,:,:]
segmentation_db = np.concatenate((segmentation_db_1[:,:,:,0], segmentation_db_2[:,:,:,0]), axis = 2)
segmentation_db = 1.0*(segmentation_db>0)

################ Read the Paired Masks ####################
segmentation_pair_1 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Pair Segmentation/maskBRATS_012.nii").get_fdata()[:,:,:]
segmentation_pair_2 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Pair Segmentation/maskBRATS_018.nii").get_fdata()[:,:,:]
segmentation_pair = np.concatenate((segmentation_pair_1[:,:,:,0], segmentation_pair_2[:,:,:,0]), axis = 2)
segmentation_pair = 1.0*(segmentation_pair>0)

################ Read the Ground Truth Masks ####################
ground_truth_1 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/BRATS_012.nii").get_fdata()[:,:,:]
ground_truth_2 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/BRATS_018.nii").get_fdata()[:,:,:]
ground_truth = np.concatenate((ground_truth_1, ground_truth_2), axis = 2)
ground_truth = 1.0*(ground_truth>0)

################ Applying STAPLE and Majority Voting to both the individuals ######################
segmentation_staple, segmentation_mv = calc_STAPLE(segmentation_kd, segmentation_db)

################ Calculate the Dice and Hausdorff for each slices for the subjects ##############
dice_score_kd = []
dice_score_db = []
dice_score_pair = []
dice_score_gt = []
dice_score_staple = []
dice_score_mv = []

for i in range(0, ground_truth.shape[2]):
    dice_score_kd.append(f1_score(segmentation_kd[:,:,i].flatten(), ground_truth[:,:,i].flatten()))
    dice_score_db.append(f1_score(segmentation_db[:,:,i].flatten(), ground_truth[:,:,i].flatten()))
    dice_score_pair.append(f1_score(segmentation_pair[:,:,i].flatten(), ground_truth[:,:,i].flatten()))
    dice_score_gt.append(f1_score(ground_truth[:,:,i].flatten(), ground_truth[:,:,i].flatten()))
    dice_score_staple.append(f1_score(segmentation_staple[:,:,i].flatten(), ground_truth[:,:,i].flatten()))
    dice_score_mv.append(f1_score(segmentation_mv[:,:,i].flatten(), ground_truth[:,:,i].flatten()))
    
dice_score_kd = np.array(dice_score_kd)
dice_score_db = np.array(dice_score_db)
dice_score_pair = np.array(dice_score_pair)
dice_score_gt = np.array(dice_score_gt)
dice_score_staple = np.array(dice_score_staple)
dice_score_mv = np.array(dice_score_mv)
#number_of_slice = np.count_nonzero(dice_score_gt)

dice_score_kd_final = dice_score_kd[np.nonzero(dice_score_kd)]
dice_score_db_final = dice_score_db[np.nonzero(dice_score_db)]
dice_score_pair_final = dice_score_pair[np.nonzero(dice_score_pair)]
dice_score_staple_final = dice_score_staple[np.nonzero(dice_score_staple)]
dice_score_mv_final = dice_score_mv[np.nonzero(dice_score_mv)]


data = [dice_score_kd_final, dice_score_db_final, dice_score_mv_final, dice_score_staple_final, dice_score_pair_final]
fig = plt.figure(figsize =(16, 10), dpi = 800)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_ylabel('Dice Score',fontsize=45)
ax.set_xticklabels(['Individual 1','Individual 2','Majority Vote','STAPLE','Paired'],fontsize=30)
#ax.set_yticklabels([0,0,0.2,0.4,0.6,0.8,1.0],fontsize=30)
ax.tick_params(axis='y', which='major', labelsize=30)
plt.boxplot(data)
plt.show()












