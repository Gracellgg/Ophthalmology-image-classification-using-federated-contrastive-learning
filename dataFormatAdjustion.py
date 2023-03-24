# -*- coding: utf-8 -*-
import os
import shutil
import pandas
import pandas as pd
from sklearn.model_selection import train_test_split


def moveFile():
    # move all images under source path to destination path
    source_path = 'E:/实验室/皮质白内障数据/C_cataract/'
    dest_path = 'E:/实验室/皮质白内障数据/C_cataract_one_file/'
    csv_file = pandas.DataFrame(columns=['data_dir', 'label'])
    file_list = os.listdir(source_path)
    for small_file in file_list:
        # move all images under small file to destination path, and add the path and label to csv file
        file_path = os.path.join(source_path, small_file)
        print(file_path)
        img_list = os.listdir(file_path)
        for img in img_list:
            img_path = os.path.join(file_path, img)
            label = int(small_file.split('_')[-1])
            dest_img_path = os.path.join(dest_path, img)

            shutil.copy(img_path, dest_img_path)
            csv_file = pd.concat([csv_file, pd.DataFrame({'data_dir': [img], 'label': [label]})], ignore_index=True)

    csv_file.to_csv('C_cataract_data.csv', index=False)


def add_three_class_label():
    csv_file = pd.read_csv('C_cataract_data.csv')
    # if column five is 0, then label is 0, else if column five is 1 or 2, then label is 1, else label is 2
    csv_file['label'] = csv_file['five'].apply(lambda x: 0 if x == 0 else (1 if x == 1 or x == 2 else 2))
    csv_file.to_csv('C_cataract_data.csv', index=False)



if __name__ == '__main__':
    #add_three_class_label()
    csv = pd.read_csv('C_cataract_data.csv')
    train, test = train_test_split(csv, test_size=0.2, random_state=42)
    train.to_csv('C_cataract_train.csv', index=False)
    test.to_csv('C_cataract_test.csv', index=False)


