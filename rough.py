# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:54:24 2023

@author: eesha
"""
'''
NOTE: New approach is going to be used. all of the current training images are going to be used in augmentations.

1. Just increase all of the under-represented classes to 150
2. Make all classes numbers equal.

This coupled with 10 % boundary removal could be used to boost results. 
the image ids will have annotations as well as original source mentioned. 
Also, metadata will have model prediction (from final layer) as well as confidence score. 
This should be probably performed in a new notebook. A new script here as well.
Also check out top-3 and top-5 mAP method of reporting the model accuracy. 
'''
import numpy as np
import pandas as pd
import operator

ops = {'=' : operator.eq, '<' : operator.lt, '>' : operator.gt, '>=' : operator.ge, '<=' : operator.le}

def filter_priority(df, crit_list):
    for tup in crit_list:
        col, val, op = tup
        df = df[ops[op](df[col], val)]
    return df
    


train_meta = "D:/Coding/ScriptsProjects/Pinecone/DeepDriD/regular-fundus-training.csv"
test_meta = "D:/Coding/ScriptsProjects/Pinecone/DeepDriD/regular-fundus-validation.csv"

df_train = pd.read_csv(train_meta)
df_test = pd.read_csv(test_meta)

df = pd.concat([df_train, df_test])
df = df.reset_index()

print(df.columns)

eye_dr_lvl = []
eye = 1
eye_cols = {0 : 'left_eye_DR_Level', 1 : 'right_eye_DR_Level'}
for i in range(len(df)):
    if i % 2 == 0:
        eye = 1 - eye
    col = eye_cols[eye]
    eye_dr_lvl.append(int(df.loc[i, col]))

df["eye_DR_Level"] = eye_dr_lvl
unique_eye_dr = list(set(eye_dr_lvl))

#crit_list = [("Overall quality", 1, '='), ("Clarity", 7, '>'), ("Artifact", 4, '<')]
crit_list = [("Overall quality", 1, '='), ("Clarity", 7, '>')]

df_list = []
df_train_list = []
df_valid_list = []

counts = []
valid_split = 0.1

for lvl in unique_eye_dr:
    df_lvl = df[df["eye_DR_Level"] == lvl]
    df_filtered = filter_priority(df_lvl, crit_list)
    counts.append(len(df_filtered))
    num_valid = int(counts[-1] * valid_split)
    
    valid_indices = set(np.random.randint(low = 0, high = counts[-1], size = num_valid))
    mask_valid = np.array([i in valid_indices for i in range(counts[-1])])
    
    df_valid_temp = df_filtered[mask_valid]
    df_train_temp = df_filtered[~ mask_valid]
    
    df_train_list.append(df_train_temp)
    df_valid_list.append(df_valid_temp)
    
    df_list.append(df_filtered)

df_final = pd.concat(df_list)
df_final = df_final.reset_index()
# I should probably manually add a few type 1 and type 4 images, to create another dataset, which is slightly more balanced.
# Rather than to valid as well, I should just add to train. A more balanced tree could drastically improve results. Maybe, I 
# should even remove some of the normal images. Maybe down to 160.
# or, maybe I should use image augmentation to increase the number of images. 



df_train = pd.concat(df_train_list)
df_train = df_train.reset_index()

df_valid = pd.concat(df_valid_list)
df_valid = df_valid.reset_index()

df_train_final = df_train[["patient_id", "image_id", "image_path", "eye_DR_Level"]].copy()
df_valid_final = df_valid[["patient_id", "image_id", "image_path", "eye_DR_Level"]].copy()

df_train_final.to_csv("D:/Coding/ScriptsProjects/Pinecone/DeepDriD/training/training_data.csv")
df_valid_final.to_csv("D:/Coding/ScriptsProjects/Pinecone/DeepDriD/validation/validation_data.csv")

###
img_id = list(df_valid["image_id"])

print([int(counts[i] - 0.1 * counts[i]) for i in range(len(counts))])
print([int(0.1 * counts[i]) for i in range(len(counts))])


