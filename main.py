import os
import re
import cv2
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops


from gui import *

# -------------------- Utility function ------------------------

# normalisasi label
def normalize_label(str_):
    str_ = str_.replace(" ", "")
    str_ = str_.translate(str_.maketrans("", "", "()"))
    str_ = str_.split("_")
    return ''.join(str_[:2])


def normalize_desc(folder, sub_folder):
    text = folder + " - " + sub_folder
    text = re.sub(r'\d+', '', text)
    text = text.replace(".", "")
    text = text.strip()
    return text

def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    ymin, ymax, xmin, xmax = h // 3, h * 2 // 3, w // 3, w * 2 // 3
    crop = gray[ymin:ymax, xmin:xmax]

    resize = cv2.resize(crop, (0, 0), fx=0.5, fy=0.5)

    return resize


# calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], lvl=256, sym=True,
                       norm=True):
    glcm = graycomatrix(img, distances=dists, angles=agls, levels=lvl, symmetric=sym, normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)

    if label != 0:
        feature.append(label)

    return feature


def load_dataset(dataset_dir, csv_save):
    # -------------------- Load Dataset ------------------------

    imgs = []  # list image matrix
    labels = []
    descs = []

    for folder in os.listdir(dataset_dir):
        for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
            sub_folder_files = os.listdir(os.path.join(dataset_dir, folder, sub_folder))
            len_sub_folder = len(sub_folder_files) - 1
            for i, filename in enumerate(sub_folder_files):
                img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder, filename))

                imgs.append(preprocessing(img))
                labelfix = (str(sub_folder)).replace(" ", "-")
                labels.append(labelfix)
                descs.append(normalize_desc(folder, sub_folder))

    # ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

    glcm_all_agls = []
    for img, label in zip(imgs, labels):
        glcm_all_agls.append(calc_glcm_all_agls(img, label, props=properties))
    columns = []
    angles = ['0', '45', '90', '135']

    for name in properties:
        for ang in angles:
            columns.append(name + "_" + ang)

    columns.append("label")

    # Create the pandas DataFrame for GLCM features data
    glcm_df = pd.DataFrame(glcm_all_agls, columns=columns)
    glcm_df.to_csv(csv_save, index=False)

# main function
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

# direktori file csv
csv_save = "csv/ekstraksi_fitur.csv"


# PNN classsification

# Make a prediction using csv
dataset = load_csv(csv_save)
dataset.pop(0)

for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)

# convert class column to integers and get count of class
num_class = str_column_to_int(dataset, len(dataset[0]) - 1)[0][1]

# define model parameter
n_folds = 5
num_neighbors = 3


# gui.on_click("GLCM")