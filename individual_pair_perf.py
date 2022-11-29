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

############## Read the Individual Segmentation ################
segmentation_kd_1 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/Kaushik/maskBRATS_012.nii").get_fdata()[:,:,:]
segmentation_kd_2 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/Kaushik/maskBRATS_018.nii").get_fdata()[:,:,:]
segmentation_kd = np.concatenate((segmentation_kd_1[:,:,:,0], segmentation_kd_2[:,:,:,0]), axis = 2)

segmentation_db_1 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/Danni/maskBRATS_012_DB.nii").get_fdata()[:,:,:]
segmentation_db_2 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/Danni/maskBRATS_018_DB.nii").get_fdata()[:,:,:]
segmentation_db = np.concatenate((segmentation_db_1[:,:,:,0], segmentation_db_2[:,:,:,0]), axis = 2)

################ Read the Paired Masks ####################
segmentation_pair_1 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Pair Segmentation/maskBRATS_012.nii").get_fdata()[:,:,:]
segmentation_pair_2 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Pair Segmentation/maskBRATS_018.nii").get_fdata()[:,:,:]
segmentation_pair = np.concatenate((segmentation_pair_1[:,:,:,0], segmentation_pair_2[:,:,:,0]), axis = 2)

################ Read the Ground Truth Masks ####################
ground_truth_1 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/BRATS_012.nii").get_fdata()[:,:,:]
ground_truth_2 = nib.load("C:/Users/kaushik.dutta/Box/Brain Tumor Study/Individual Segmentation/BRATS_018.nii").get_fdata()[:,:,:]
ground_truth = np.concatenate((ground_truth_1, ground_truth_2), axis = 2)
ground_truth = 1.0*(ground_truth>0)

################ Calculate the Dice and Hausdorff for each slices for the subjects ##############
dice_score_kd = []
dice_score_db = []
dice_score_pair = []
dice_score_gt = []

for i in range(0, ground_truth.shape[2]):
    dice_score_kd.append(f1_score(segmentation_kd[:,:,i].flatten(), ground_truth[:,:,i].flatten()))
    dice_score_db.append(f1_score(segmentation_db[:,:,i].flatten(), ground_truth[:,:,i].flatten()))
    dice_score_pair.append(f1_score(segmentation_pair[:,:,i].flatten(), ground_truth[:,:,i].flatten()))
    dice_score_gt.append(f1_score(ground_truth[:,:,i].flatten(), ground_truth[:,:,i].flatten()))
    
dice_score_kd = np.array(dice_score_kd)
dice_score_db = np.array(dice_score_db)
dice_score_pair = np.array(dice_score_pair)
dice_score_gt = np.array(dice_score_gt)
number_of_slice = np.count_nonzero(dice_score_gt)

dice_score_kd_final = dice_score_kd[np.nonzero(dice_score_kd)]
dice_score_db_final = dice_score_db[np.nonzero(dice_score_db)]
dice_score_pair_final = dice_score_pair[np.nonzero(dice_score_pair)]

data = [dice_score_kd_final, dice_score_db_final, dice_score_pair_final]
fig = plt.figure(figsize =(10, 7))
plt.boxplot(data)
plt.show()
