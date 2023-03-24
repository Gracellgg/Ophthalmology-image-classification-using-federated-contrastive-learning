import os
import pandas as pd
import cv2 as cv

root_Dir = 'C:/Users/Administrator/Desktop/simCLR/data/tumor/'
if not os.path.exists(root_Dir+'classification_img/'):
    os.makedirs(root_Dir+'classification_img/')
    os.makedirs(root_Dir + 'classification_img/' + 'CT_A/')
    os.makedirs(root_Dir + 'classification_img/' + 'CT_P/')
    os.makedirs(root_Dir + 'classification_img/' + 'mask_A/')
    os.makedirs(root_Dir + 'classification_img/' + 'mask_P/')



A_csv = pd.read_csv(root_Dir + 'AP.csv')
P_csv = pd.read_csv(root_Dir + 'PVP.csv')

# add a new column new_img_path to A_csv and P_csv
A_csv['new_img_path'] = ''
P_csv['new_img_path'] = ''


# read image path in A_csv tumor_max_dir column, save the image to classification_img/mask_A and change the image name to the index of the image
for i in range(len(A_csv)):
    img_path = A_csv['tumor_max_dir'][i]
    people = A_csv['people'][i]
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    cv.imwrite(root_Dir + 'classification_img/mask_A/' + people + str(i) + '.png', img)
    # change the 'mask' in img_path to 'img'
    CT_img_path = img_path.replace('mask', 'img')
    CT_img = cv.imread(CT_img_path, cv.IMREAD_GRAYSCALE)
    cv.imwrite(root_Dir + 'classification_img/CT_A/' + people + str(i) + '.png', CT_img)

    A_csv['new_img_path'][i] = people + str(i) + '.png'


# read image path in P_csv tumor_max_dir column, save the image to classification_img/mask__P and change the image name to the index of the image
for i in range(len(P_csv)):
    img_path = P_csv['tumor_max_dir'][i]
    people = P_csv['people'][i]
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    cv.imwrite(root_Dir + 'classification_img/mask_P/' + people + str(i) + '.png', img)
    P_csv['new_img_path'][i] = people + str(i) + '.png'
    CT_img_path = img_path.replace('mask', 'img')
    CT_img = cv.imread(CT_img_path, cv.IMREAD_GRAYSCALE)
    cv.imwrite(root_Dir + 'classification_img/CT_P/' + people + str(i) + '.png', CT_img)


# save the new csv file
A_csv.to_csv(root_Dir + 'classification_img/' + 'AP_new.csv', index=False)
P_csv.to_csv(root_Dir + 'classification_img/' + 'PVP_new.csv', index=False)

