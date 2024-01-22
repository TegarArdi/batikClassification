import os
import re
import cv2
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops

import tkinter as tk
from tkinter import ttk
import threading
import time


# -------------------- Untuk Loading ------------------------


# -------------------- Untuk proses GLCM------------------------
class Glcm():
    # normalisasi label
    def normalize_label(self, str_):
        str_ = str_.replace(" ", "")
        str_ = str_.translate(str_.maketrans("", "", "()"))
        str_ = str_.split("_")
        return ''.join(str_[:2])

    def normalize_desc(self, folder, sub_folder):
        text = folder + "/" + sub_folder
        # text = re.sub(r'\d+', '', text)
        # text = text.replace(".", "")
        # text = text.strip()
        return text

    # buatan
    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def cropping(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, j = img.shape
        new_width = w / 2
        new_height = h / 2

        if h > w:
            ymin, ymax, xmin, xmax = h // 2 - new_width, h // 2 + new_width, w // 2 - new_width, w // 2 + new_width
            # ymin, ymax, xmin, xmax = 0, w, 0, w
        else:
            ymin, ymax, xmin, xmax = h // 2 - new_height, h // 2 + new_height, w // 2 - new_height, w // 2 + new_height
            # ymin, ymax, xmin, xmax = 0, h, 0, h

        crop = img[int(ymin):int(ymax), int(xmin):int(xmax)]

        resize = cv2.resize(crop, (100, 100))

        return resize

    def preprocessing(self, img):

        crop = self.cropping(img)
        gray = self.grayscale(crop)

        return gray
    # buatan

    # calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135
    def calc_glcm_all_agls(self, img, label, dists, props, agls=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], lvl=256,
                           sym=True,
                           norm=True):
        glcm = graycomatrix(img, distances=[dists], angles=agls, levels=lvl, symmetric=sym, normed=norm)
        feature = []
        glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
        for item in glcm_props:
            feature.append(item)

        if label != 0:
            feature.append(label)

        return feature

    def load_dataset(self, gui_pass, dataset_dir, csv_save, dists, root):
        # -------------------- Load Dataset ------------------------

        imgs = []  # list image matrix
        labels = []
        descs = []
        total_img = 0

        for folder in os.listdir(dataset_dir):
            for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
                total_img += 1

        print(f"total img: {total_img}")



        # # loading di sini
        # for folder in os.listdir(dataset_dir):
        #     for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
        #         try:
        #             img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder))
        #             imgs.append(self.preprocessing(img))
        #             labelfix = (str(folder)).replace(" ", "-")
        #             labels.append(labelfix)
        #             descs.append(self.normalize_desc(folder, sub_folder))
        #         except:
        #             print(os.path.join(dataset_dir, folder, sub_folder))

        # ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
        properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

        glcm_all_agls = []

        def close_pbar():
            root2.destroy()

        def glcm_selesai():
            for widget in frame.winfo_children():
                widget.destroy()

        #     untuk menambahkan tombol
            lab_sel = tk.Label(frame, text = "Proses Ekstraksi GLCM Selesai")
            lab_sel.pack(pady=10)

            btn_sel = tk.Button(frame, text="OK", command=close_pbar, width=14, height=1)
            btn_sel.pack(pady=(20, 0))

        def antrian_glcm2():
            root2.grab_set()

            idx = 0
            for folder in os.listdir(dataset_dir):
                for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):

                    try:
                        img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder))
                        imgs.append(self.preprocessing(img))
                        labelfix = (str(folder)).replace(" ", "-")
                        labels.append(labelfix)
                        descs.append(self.normalize_desc(folder, sub_folder))
                    except:
                        print(os.path.join(dataset_dir, folder, sub_folder))

        def antrian_glcm():

            root2.grab_set()

            idx = 1
            for folder in os.listdir(dataset_dir):
                for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
                    try:
                        percentase = (idx / total_img) * 100
                        progress_bar['value'] = percentase
                        bawah1.config(text="file: " + self.normalize_desc(folder, sub_folder))
                        bawah2.config(text="proses: " + str(format(percentase, '.2f')) + "%")
                        img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder))
                        img = self.preprocessing(img)
                        glcm_all_agls.append(self.calc_glcm_all_agls(img, (str(folder)).replace(" ", "-"), dists, props=properties))
                        idx += 1
                    except:
                        print(os.path.join(dataset_dir, folder, sub_folder))
                        idx += 1

            # for img, label, desc in zip(imgs, labels, descs):
            #     percentase = (idx / len(imgs)) * 100
            #     progress_bar['value'] = percentase
            #     bawah1.config(text="file: " + desc)
            #     bawah2.config(text="proses: " + str(format(percentase, '.2f')) + "%")
            #     glcm_all_agls.append(self.calc_glcm_all_agls(img, label, dists, props=properties))
            #     idx += 1

            columns = []
            angles = ['0', '45', '90', '135']

            for name in properties:
                for ang in angles:
                    columns.append(name + "_" + ang)

            columns.append("label")

            # Create the pandas DataFrame for GLCM features data
            glcm_df = pd.DataFrame(glcm_all_agls, columns=columns)
            glcm_df.to_csv(csv_save, index=False)

            # selesai
            glcm_selesai()

        # buka window baru
        root2 = tk.Toplevel(root)
        root2.title("Proses perhitungan GLCM...")

        width = 410
        height = 160

        screen_width = root.winfo_screenwidth()  # Width of the screen
        screen_height = root.winfo_screenheight()  # Height of the screen

        # Calculate Starting X and Y coordinates for Window
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)

        root2.geometry('%dx%d+%d+%d' % (width, height, x, y-200))

        frame = tk.Frame(root2, padx=20, pady=20)
        frame.pack()

        lab_prog = tk.Label(frame, text="Loading...")
        lab_prog.pack(pady=(2, 6))

        progress_bar = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        progress_bar.pack(pady=(20, 0))

        fr_bawah = tk.Frame(frame)
        fr_bawah.pack(fill=tk.X, pady=(4, 0))

        bawah1 = tk.Label(fr_bawah, text="file: ")
        bawah1.pack(side=tk.LEFT, anchor='w')

        bawah2 = tk.Label(fr_bawah, text='proses: ')
        bawah2.pack(side=tk.RIGHT, anchor='w')

        loading_thread = threading.Thread(target=antrian_glcm)
        loading_thread.start()

        def check_thread():
            if loading_thread.is_alive():
                # Jika thread masih berjalan, jadwalkan pemeriksaan berikutnya
                root.after(100, check_thread)
            else:
                gui_pass.on_click("GLCM")

        root.after(100, check_thread)  # Memeriksa status thread secara berkala

    def load_dataset2(self, gui_pass, dataset_dir, csv_save, dists, root):
        # -------------------- Load Dataset ------------------------

        imgs = []  # list image matrix
        labels = []
        descs = []

        # loading di sini

        for folder in os.listdir(dataset_dir):
            for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
                try:
                    img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder))
                    imgs.append(self.preprocessing(img))
                    labelfix = (str(folder)).replace(" ", "-")
                    labels.append(labelfix)
                    descs.append(self.normalize_desc(folder, sub_folder))
                except:
                    print(os.path.join(dataset_dir, folder, sub_folder))

        # ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
        properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

        glcm_all_agls = []

        def close_pbar():
            root2.destroy()

        def glcm_selesai():
            for widget in frame.winfo_children():
                widget.destroy()

            #     untuk menambahkan tombol
            lab_sel = tk.Label(frame, text="Proses Ekstraksi GLCM Selesai")
            lab_sel.pack(pady=10)

            btn_sel = tk.Button(frame, text="OK", command=close_pbar, width=14, height=1)
            btn_sel.pack(pady=(20, 0))

        def antrian_glcm2():
            root2.grab_set()

            idx = 0

            a = len(os.listdir(dataset_dir))

            for folder in os.listdir(dataset_dir):
                for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
                    try:
                        img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder))
                        imgs.append(self.preprocessing(img))
                        labelfix = (str(folder)).replace(" ", "-")
                        labels.append(labelfix)
                        descs.append(self.normalize_desc(folder, sub_folder))
                    except:
                        print(os.path.join(dataset_dir, folder, sub_folder))

        def antrian_glcm():

            root2.grab_set()

            idx = 1
            for img, label, desc in zip(imgs, labels, descs):
                percentase = (idx / len(imgs)) * 100
                progress_bar['value'] = percentase
                bawah1.config(text="file: " + desc)
                bawah2.config(text="proses: " + str(format(percentase, '.2f')) + "%")
                glcm_all_agls.append(self.calc_glcm_all_agls(img, label, dists, props=properties))
                idx += 1

            columns = []
            angles = ['0', '45', '90', '135']

            for name in properties:
                for ang in angles:
                    columns.append(name + "_" + ang)

            columns.append("label")

            # Create the pandas DataFrame for GLCM features data
            glcm_df = pd.DataFrame(glcm_all_agls, columns=columns)
            glcm_df.to_csv(csv_save, index=False)

            # selesai
            glcm_selesai()

        # buka window baru
        root2 = tk.Toplevel(root)
        root2.title("Proses perhitungan GLCM...")

        width = 410
        height = 160

        screen_width = root.winfo_screenwidth()  # Width of the screen
        screen_height = root.winfo_screenheight()  # Height of the screen

        # Calculate Starting X and Y coordinates for Window
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)

        root2.geometry('%dx%d+%d+%d' % (width, height, x, y - 200))

        frame = tk.Frame(root2, padx=20, pady=20)
        frame.pack()

        lab_prog = tk.Label(frame, text="Loading...")
        lab_prog.pack(pady=(2, 6))

        progress_bar = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        progress_bar.pack(pady=(20, 0))

        fr_bawah = tk.Frame(frame)
        fr_bawah.pack(fill=tk.X, pady=(4, 0))

        bawah1 = tk.Label(fr_bawah, text="file: ")
        bawah1.pack(side=tk.LEFT, anchor='w')

        bawah2 = tk.Label(fr_bawah, text='proses: ')
        bawah2.pack(side=tk.RIGHT, anchor='w')

        loading_thread = threading.Thread(target=antrian_glcm)
        loading_thread.start()

        def check_thread():
            if loading_thread.is_alive():
                # Jika thread masih berjalan, jadwalkan pemeriksaan berikutnya
                root.after(100, check_thread)
            else:
                gui_pass.on_click("GLCM")

        root.after(100, check_thread)  # Memeriksa status thread secara berkala

    def glcm_avg(self, dataset_dir, csv_dir, dists, root):

        if not (os.path.exists(csv_dir)):
            self.glcm(dataset_dir, csv_dir, dists, root)
            return 0

        df = pd.read_csv(csv_dir)
        df_avg = pd.DataFrame()

        df_avg['dissiimilarity'] = df.iloc[:, 0:4].mean(axis=1)
        df_avg['correlation'] = df.iloc[:, 4:8].mean(axis=1)
        df_avg['homogeneity'] = df.iloc[:, 8:12].mean(axis=1)
        df_avg['contrast'] = df.iloc[:, 12:16].mean(axis=1)
        df_avg['ASM'] = df.iloc[:, 16:20].mean(axis=1)
        df_avg['energy'] = df.iloc[:, 20:24].mean(axis=1)
        df_avg['label'] = df.iloc[:, 24]

        return df_avg

    def glcm(self, dataset_dir, csv_dir, dists, root):
        dataset_dir = dataset_dir
        csv_save = csv_dir
        self.load_dataset(dataset_dir, csv_save, dists, root)

    def pnn(self):
        # main function
        properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

        # direktori file csv
        csv_save = "csv/ekstraksi_fitur.csv"

        # PNN classsification

        # Make a prediction using csv
        dataset = self.load_csv(csv_save)
        dataset.pop(0)

        for i in range(len(dataset[0]) - 1):
            self.str_column_to_float(dataset, i)

        # convert class column to integers and get count of class
        num_class = self.str_column_to_int(dataset, len(dataset[0]) - 1)[0][1]

        # define model parameter
        n_folds = 5
        num_neighbors = 3

