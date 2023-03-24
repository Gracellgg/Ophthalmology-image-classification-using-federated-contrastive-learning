import os
import pandas as pd

base_dir = 'D:/PycharmProjects/fairness/pics/Messdior dataset/'

def create_large_csv():


    # Read the csv file
    csv_name_list = ['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34']

    # concat all csv files
    large_csv = pd.DataFrame(
        columns=['Image name', 'Ophthalmologic department', 'Retinopathy grade', 'Risk of macular edema'])

    for csv_name in csv_name_list:
        csv = pd.read_excel(base_dir + 'Annotation_Base' + csv_name + '.xls', sheet_name='Feuil1')
        large_csv = pd.concat([large_csv, csv], ignore_index=True)

    # add a new column department to the csv file, which is a number representing Ophthalmologic department column
    large_csv['department'] = large_csv['Ophthalmologic department'].map(
        {'Service Ophtalmologie Lariboisière': 0, 'CHU de St Etienne': 1, 'LaTIM - CHU de BREST': 2})

    # save the csv file
    large_csv.to_csv('large_csv.csv', index=False)


# unzip the zip file under the base_dir
def unzip():
    import zipfile
    name_list = ['11', '12', '13', '14', '21', '22', '23', '24', '31', '32', '33', '34']
    for name in name_list:
        zip_name = base_dir + 'Base' + name + '.zip'
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(base_dir + 'img/')

# turn all tif images to png images
def tif_to_png():
    from PIL import Image
    import glob
    import os
    import numpy as np

    # get all tif images
    tif_list = glob.glob(base_dir + 'img/*.tif')

    # convert all tif images to png images
    for tif in tif_list:
        img = Image.open(tif)
        img = img.convert('RGB')
        img.save(tif[:-3] + 'png')

    # delete all tif images
    for tif in tif_list:
        os.remove(tif)

tif_to_png()