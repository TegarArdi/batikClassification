import math
import os
import tkinter as tk
from os.path import exists
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import pandas as pd
from tkinter.messagebox import showinfo

import PIL
import pandas
from CTkMessagebox import CTkMessagebox
import customtkinter
from CTkTable import *

import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from glcm import Glcm
from pnn_akurasi import PnnAk
from pnn_with_data import PnnData


class Setting:
    def __init__(self):
        self.folder = "dataset/"
        self.csv_save = "csv/ekstraksi_fitur.csv"
        self.jarak = 1
        self.n_fold = 5
        self.dataset_stat = False
        self.glcm_stat = False
        self.Pengujian_stat = False
        self.Klasifikasi_stat = False
        self.all = list()
        self.img = ""
        self.img_dir = ""
        self.num_neighbors = 5
        self.data_jarak = list()
        self.data_klasifikasi = list()
        self.img_grayscalle = ""

    #     untuk glcm
        self.glcm = []
        self.graycomatrix = []
        self.graycrop = []
        self.glcm_symm = []
        self.norm = []
        self.glcm_awal = []


    def default(self):
        self.folder = "dataset/"
        self.csv_save = "csv/ekstraksi_fitur.csv"
        self.jarak = 1
        self.n_fold = 5
        self.dataset_stat = False
        self.glcm_stat = False
        self.Pengujian_stat = False
        self.Klasifikasi_stat = False
        self.all = list()
        self.img = ""
        self.img_dir = ""
        self.num_neighbors = 5
        self.data_jarak = list()

    def set_stat(self, a, val):
        if (a == 1):
            self.dataset_stat = val
        elif (a == 2):
            self.glcm_stat = val
        elif (a == 3):
            self.pengujian_stat = val
        else:
            self.Klasifikasi_stat = val

    def get_stat(self, a):
        if (a == 1):
            return self.dataset_stat
        elif (a == 2):
            return self.glcm_stat
        elif (a == 3):
            return self.pengujian_stat
        else:
            return self.Klasifikasi_stat

    def set_folder(self, a):
        self.folder = a

    def get_folder(self):
        return self.folder


class GUI:

    def kembali(self, to):
        for widget in content_frame.winfo_children():
            widget.destroy()

        self.on_click(to)

    # fungsi hapus widget di frame
    def clear_frame(self):
        for widgets in content_frame.winfo_children():
            widgets.destroy()

    # dataset
    def Dataset(self):

        def clear_canvas():
            for widgets in f_kanan.winfo_children():
                widgets.destroy()

        def set_data():
            dt_entry.configure(state='normal')
            dt_entry.delete(0, customtkinter.END)
            dt_entry.insert(0, setting.folder)
            dt_entry.configure(state='readonly')
            show_data(setting.get_folder())


        def pilih_folder():
            folder = filedialog.askdirectory()
            dt_entry.configure(state='normal')
            dt_entry.delete(0, customtkinter.END)
            dt_entry.insert(0, folder)
            dt_entry.configure(state='readonly')
            setting.folder = folder
            show_data(folder)

        def show_data(folder):
            try:
                clear_canvas()
                folder1 = os.listdir(folder)

                for sub in folder1:
                    folder2 = os.path.join(folder, sub)

                    image_count = 0
                    columns = 10

                    img_row = customtkinter.CTkFrame(f_kanan)
                    img_row.pack(pady=10)

                    class_label = customtkinter.CTkLabel(img_row, text=sub)
                    class_label.grid(row=0, column=0, columnspan=columns, sticky='w')

                    file = os.listdir(folder2)

                    for name in file:
                        image_count += 1
                        if(image_count < 10):
                            r, c = divmod(image_count - 1, columns)
                            im = Image.open(os.path.join(folder2, name))
                            resized = im.resize((100, 100), Image.Resampling.LANCZOS)  # gambar ditampilkan secara 1:1
                            tkimage = customtkinter.CTkImage(resized, size=(100, 100))
                            myvar = customtkinter.CTkLabel(img_row, image=tkimage, text=" ")
                            myvar.image = tkimage
                            myvar.grid(row=r + 1, column=c)
            except:
                CTkMessagebox(title="Error", message="Folder dataset tidak valid, dataset diatur ke default",
                              icon="cancel")
                dt_entry.configure(state='normal')
                dt_entry.delete(0, customtkinter.END)
                dt_entry.insert(0, "dataset/")
                dt_entry.configure(state='readonly')
                setting.folder = "dataset/"
                show_data("dataset/")

        # buat 2 frame lagi
        f_kiri = customtkinter.CTkFrame(content_frame)  # pengaturan dataset
        f_kiri.pack(side=customtkinter.TOP, anchor='w', pady=(20, 0), padx=20)

        kr1 = customtkinter.CTkFrame(f_kiri, fg_color='transparent')
        kr1.pack(side=customtkinter.LEFT, anchor='n', padx=20, pady=20)

        dt_label = customtkinter.CTkLabel(kr1, text="Folder:")
        dt_label.pack(side=customtkinter.TOP, anchor='w')

        dt_entry = customtkinter.CTkEntry(kr1, width=400)
        dt_entry.pack(side=customtkinter.LEFT, anchor='w')
        # dt_entry.configure(state='readonly', readonlybackground="#fff")

        dt_btn = customtkinter.CTkButton(kr1, text="Pilih Folder", command=pilih_folder)
        dt_btn.pack(side=customtkinter.LEFT, anchor='w', padx=20)

        # bawah
        f_kanan = customtkinter.CTkScrollableFrame(content_frame)
        f_kanan.pack(fill=customtkinter.BOTH, expand=True, pady=20, padx=20)

        set_data()

    # glcm
    def GLCM(self):

        def mulai():
            j_entry.set(str(setting.jarak))

            if (os.path.exists(setting.csv_save)):
                upload_file()

            folder_entry.configure(state='normal')
            folder_entry.insert(0, setting.folder)
            folder_entry.configure(state='readonly')

        # fungsi get csv
        def upload_file():
            dists = setting.jarak

            df = glcm.glcm_avg(setting.folder, setting.csv_save, dists, root)
            l1 = list(df)  # List of column names as header
            # lab_kolom.configure(text = (str(df.shape[1])))
            # lab_baris.configure(text = (str(df.shape[0])))
            trv_refresh(df, l1)  # show Treeview

        # treeview untuk menampilkan df
        def trv_refresh(df, l1):  # Refresh the Treeview to reflect changes

            r_set = df.to_numpy().tolist()  # create list of list using rows

            trv = ttk.Treeview(f_kanan, selectmode='browse', height=20,
                               show='headings', columns=l1)
            trv.pack(pady=20, padx=20, fill=customtkinter.BOTH, expand=True, side=customtkinter.LEFT, anchor='w')

            for i in l1:
                trv.column(i, width=18, anchor='w')
                trv.heading(i, text=str(i))
            idx = 0
            for dt in r_set:
                v = [r for r in dt]
                trv.insert("", 'end', iid=idx, values=v)
                idx += 1

        def hitung_glcm():
            folder = setting.folder
            csv = setting.csv_save

            n_index = int(j_entry.get())
            setting.jarak = n_index
            jarak = n_index

            glcm.load_dataset(self, folder, csv, jarak, root)
            setting.glcm_stat = True

        def reset_glcm():
            csv = setting.csv_save

            def exit_application():
                msg_box = customtkinter.messagebox.askquestion('Reset GLCM',
                                                               'Yakin untuk mereset hasil GLCM?',
                                                               icon='warning')
                if msg_box == 'yes':
                    customtkinter.messagebox.showinfo('Return', 'Data GLCM berhasil direset')
                    if os.path.exists(csv):
                        os.remove(csv)
                else:
                    customtkinter.messagebox.showinfo('Return', 'Batal direset')

            exit_application()

            self.on_click("GLCM")

        # buat 2 frame lagi
        f_kiri = customtkinter.CTkFrame(content_frame)
        f_kiri.pack(fill=customtkinter.X, pady=(20, 0), padx=20)

        f_kanan = customtkinter.CTkScrollableFrame(content_frame)
        f_kanan.pack(fill=customtkinter.BOTH, expand=True, pady=(20, 20), padx=20)

        # frame pertama di kiri
        r1_kiri = customtkinter.CTkFrame(f_kiri, fg_color="transparent")
        r1_kiri.pack(side=customtkinter.LEFT, anchor="n", padx=20, pady=10)

        # di dalam frame kiri
        # frame untuk pengaturan folder
        fd_frame = customtkinter.CTkFrame(r1_kiri, fg_color="transparent")
        fd_frame.pack(fill=customtkinter.BOTH)

        folder_label = customtkinter.CTkLabel(fd_frame, text="Folder:")
        folder_label.pack(padx=20, pady=10, side=customtkinter.LEFT)

        folder_entry = customtkinter.CTkEntry(fd_frame, width=300)
        folder_entry.pack(padx=2, pady=5, side=customtkinter.LEFT)
        folder_entry.configure(state='readonly')

        # frame untuk bawahnya
        j_frame = customtkinter.CTkFrame(r1_kiri, bg_color="transparent", fg_color="transparent")
        j_frame.pack(fill=customtkinter.BOTH, padx=20)

        l_jarak = customtkinter.CTkLabel(j_frame, text="Jarak:")
        l_jarak.pack(side=customtkinter.LEFT)

        combobox_var = customtkinter.StringVar(value="1")
        j_entry = customtkinter.CTkComboBox(j_frame, width=300, state='readonly', values=['1', '2', '3', '4', '5'],
                                            variable=combobox_var)

        j_entry.pack(side=customtkinter.LEFT, padx=(26, 0))

        r1_kanan = customtkinter.CTkFrame(f_kiri, fg_color="transparent")
        r1_kanan.pack(side=customtkinter.LEFT, anchor='w', fill=customtkinter.X, expand=True)

        # frame untuk tombol GLCM
        t_frame = customtkinter.CTkFrame(r1_kanan, fg_color="transparent", bg_color="transparent")
        t_frame.pack(fill=customtkinter.BOTH, padx=10)

        btn_proc = customtkinter.CTkButton(t_frame, text="Hitung GLCM", command=hitung_glcm, height=66)
        btn_proc.pack(pady=20, side=customtkinter.LEFT)

        # btn_delete = customtkinter.CTkButton(t_frame, text="Reset", command=reset_glcm)
        # btn_delete.pack(pady=15, padx=30, side=customtkinter.RIGHT)

        lab_hasil = customtkinter.CTkLabel(f_kanan, text="Hasil Ekstraksi Fitur GLCM", font=("Arial", 16))
        lab_hasil.pack(pady=(10, 0))

        mulai()

        # end frame kiri

    # pengujian
    def Pengujian(self):

        def validation(n):
            msg = "Input angka 2 sampai dengan 20"

            try:
                if (int(n) > 1) and (int(n) <= 20):
                    return True
                else:
                    CTkMessagebox(title="Error", message=msg, icon="cancel")
                    return False
            except:
                CTkMessagebox(title="Error", message=msg, icon="cancel")
                return False

        def uji():
            n_fold = e_nf.get()
            if (validation(n_fold)):
                ak.hitung_akurasi(setting.csv_save, int(n_fold), root, gui, setting)

        def cek_stat():
            if exists(setting.csv_save):
                b_uji.configure(state='normal')
            else:
                b_uji.configure(state='disable')
                msg = CTkMessagebox(title="GLCM belum dilakukan!", message="Ekstraksi Fitur GLCM?",
                                    icon="warning", option_1="Ya", option_2="Tidak")
                if msg.get() == 'Ya':
                    self.on_click("GLCM")
                else:
                    CTkMessagebox(title="Error",
                                  message="Untuk pengujian akurasi harus melakukan ekstraksi GLCM terlebih dahulu!!!",
                                  icon="cancel")

            if (len(setting.all) > 0):
                hasil = setting.all

                gen_hasil(hasil)


            e_nf.insert(0, setting.n_fold)

        # buat 2 frame lagi
        f_kiri = customtkinter.CTkFrame(content_frame)
        f_kiri.pack(side=customtkinter.TOP, anchor="w", pady=20, padx=20)

        f_kanan = customtkinter.CTkFrame(content_frame)
        f_kanan.pack(pady=(0, 20), padx=20, fill=customtkinter.BOTH, expand=True)

        # untuk isi n fold
        f_nf = customtkinter.CTkFrame(f_kiri, fg_color="transparent")
        f_nf.pack(fill=customtkinter.BOTH, side=customtkinter.LEFT, pady=20, padx=10)

        l_nf = customtkinter.CTkLabel(f_nf, text="Masukkan Nilai N-Fold:")
        l_nf.pack(side=customtkinter.LEFT, padx=(10, 0))

        e_nf = customtkinter.CTkEntry(f_nf)
        e_nf.pack(side=customtkinter.LEFT, padx=(10, 0))

        b_uji = customtkinter.CTkButton(f_nf, text="Pengujian Data", command=uji)
        b_uji.pack(side=customtkinter.LEFT, padx=10)

        def gen_hasil(hasil):

            def tabel_akurasi(hasil):

                setengah = f_kanan.winfo_width() / 2
                lf_table_akurasi.configure(width=setengah)

                kolom = ('Nilai K', 'Akurasi', 'Rata-rata Akurasi')

                values = []

                values.append(kolom)

                for n in range(len(hasil[1])):
                    avg = "{:.4f}".format(np.average(hasil[1][n]))
                    values.append((f'K = {n + 1}', f'{str(hasil[1][n])}', avg))

                tb_akurasi = CTkTable(lf_table_akurasi, row=len(hasil[1]) + 1, column=3, values=values)
                tb_akurasi.pack(padx=10, pady=10, expand=True)

                tb_akurasi.edit_column(0, width=10)

            # grafik
            def plot(hasil):

                # the figure that will contain the plot
                fig = Figure(figsize=(5, 5),
                             dpi=100)

                xpoints = hasil[0]
                ypoints = list()

                for i in hasil[1]:
                    ypoints.append(np.average(i))

                # adding the subplot
                plot1 = fig.add_subplot(111)

                plot1.plot()

                plot1.set_xlabel('Nilai K')
                plot1.set_ylabel('Nilai Akurasi')

                plot1.set_title("Akurasi Berdasarkan Nilai K")
                plot1.set_ylim([0, 100])

                plot1.set_yticks([i for i in range(1, 100, 4)])
                plot1.set_yticklabels([i for i in range(1, 100, 4)])

                plot1.set_xticks([i for i in range(1, 21)])
                plot1.set_xticklabels([i for i in range(1, 21)])

                # plotting the graph
                plot1.plot(xpoints, ypoints)

                # creating the customtkinterinter canvas
                # containing the Matplotlib figure
                canvas = FigureCanvasTkAgg(fig,
                                           master=lf_grap_akurasi)
                canvas.draw()

                canvas.get_tk_widget().pack()

                canvas.get_tk_widget().pack(padx=10, pady=10)

            # isi bawah

            # hasil nilai k -> graphic akurasi
            f_b1 = customtkinter.CTkFrame(f_kanan)
            f_b1.pack(side=customtkinter.TOP, anchor='w', fill=customtkinter.X, expand=True)

            # tabel frame nilai K

            lf_table_akurasi = customtkinter.CTkScrollableFrame(f_b1, orientation="horizontal")
            lf_table_akurasi.pack(side=customtkinter.LEFT, expand=True, fill=customtkinter.BOTH, padx=0)

            lb_tabel_ak = customtkinter.CTkLabel(lf_table_akurasi, text="Tabel nilai akurasi")
            lb_tabel_ak.pack(pady=(10, 0))

            #
            lf_grap_akurasi = customtkinter.CTkFrame(f_b1)
            lf_grap_akurasi.pack(side=customtkinter.RIGHT, expand=True, fill=customtkinter.BOTH, padx=(10, 0))

            lb_graf_ak = customtkinter.CTkLabel(lf_grap_akurasi, text="Tabel nilai akurasi")
            lb_graf_ak.pack(pady=(10, 0))

            # panggil tabel
            tabel_akurasi(hasil)

            # akurasi
            plot(hasil)

        cek_stat()

    # detail glcm
    def detail_glcm(self):
        for widget in content_frame.winfo_children():
            widget.destroy()

        # title
        title = customtkinter.CTkFrame(content_frame)
        title.pack(fill=customtkinter.X, pady=20, padx=20)

        # tombol back
        back = customtkinter.CTkButton(title, text="Kembali", command=lambda: self.kembali("Klasifikasi Citra"))
        back.pack(side=customtkinter.LEFT, anchor="w", padx=20, pady=20)

        # judul
        judul = customtkinter.CTkLabel(title, text="Detail Perhitungan GLCM")
        judul.pack(side=customtkinter.RIGHT, pady=20, fill=customtkinter.X, padx=20)

        # frame untuk detail perhitungan
        frame = customtkinter.CTkScrollableFrame(content_frame)
        frame.pack(fill=customtkinter.BOTH, padx=20, pady=20, expand=True)

        # no 1 citra grayscale
        judul1 = customtkinter.CTkLabel(frame, text="1. Citra Grayscale", font=(0, 14))
        judul1.pack(anchor="w", padx=10)

        judul1_bw = customtkinter.CTkFrame(frame)
        judul1_bw.pack(anchor="w")

        image1 = customtkinter.CTkLabel(judul1_bw, width=100, height=100, fg_color="#fff", text="")
        image1.pack(padx=10, pady=20, side=customtkinter.LEFT)

        image_gr = Image.fromarray(setting.img_grayscalle)
        img3 = customtkinter.CTkImage(light_image=image_gr, size=(100, 100))

        image1.configure(image=img3)

        text_lebar = customtkinter.CTkLabel(judul1_bw, text="Lebar : 100px\n\nTinggi: 100px")
        text_lebar.pack(anchor="w", pady=20, padx=10)

        # no 2 Matriks citra

        judul2 = customtkinter.CTkLabel(frame, text="2. Matriks citra grayscale", font=(0, 14))
        judul2.pack(anchor="w", padx=10, pady=(20, 0))

        np_array = np.array(setting.img_grayscalle)

        arrr = np.array(np_array, order='c')
        a2 = np.resize(arrr, (4, 4))

        import sys
        np.set_printoptions(threshold=np.inf, linewidth=3000)


        # hori
        scf = customtkinter.CTkScrollableFrame(frame, orientation='horizontal', height=600)
        scf.pack(padx=20, fill=customtkinter.X, expand=True)

        # textbox
        ctext = customtkinter.CTkTextbox(scf, width=5000, height=600)
        ctext.pack()

        for ix in reversed(np_array):
            ctext.insert("0.0", ix)
            ctext.insert("0.1", "\n")

        # no 3 Perhitungan sudut
        judul3 = customtkinter.CTkLabel(frame, text="3. Hitung nilai matriks GLCM Berdasarkan Sudut", font=(0, 14))
        judul3.pack(anchor="w", padx=10, pady=(20, 0))

        tab3 = customtkinter.CTkTabview(frame)
        tab3.pack(padx=20, fill=tk.X, expand=True)

        sudut = ["0", "45", "90", "135"]

        for sd in sudut:
            tab3.add(sd)

            scr = customtkinter.CTkScrollableFrame(tab3.tab(sd), orientation="horizontal", height=600)
            scr.pack(fill=tk.BOTH, expand=True)

            lb = customtkinter.CTkLabel(scr, text=f"Sudut: {str(sd)}")
            lb.pack(anchor='w')

            # textbox
            ctext = customtkinter.CTkTextbox(scr, width=6000, height=600)
            ctext.pack()

            datax = []
            if(sd == "0"):
                datax = setting.glcm_awal[:, :, 0, 0]
            elif(sd == "45"):
                datax = setting.glcm_awal[:, :, 0, 1]
            elif(sd == "90"):
                datax = setting.glcm_awal[:, :, 0, 2]
            elif(sd == "135"):
                datax = setting.glcm_awal[:, :, 0, 3]

            np.set_printoptions(suppress=True)
            pd.set_option('display.float_format', lambda x: '%.10f' % x)

            for x in reversed(datax):
                ctext.insert("0.0", x)
                ctext.insert("0.0", "\n")


        # no 4 hitung symmetrical matriks
        judul4 = customtkinter.CTkLabel(frame, text="4. Hitung Simmetrical matriks", font=(0, 14))
        judul4.pack(anchor="w", padx=10, pady=(20, 0))

        tab4 = customtkinter.CTkTabview(frame)
        tab4.pack(padx=20, fill=tk.X, expand=True)

        sudut = ["0", "45", "90", "135"]

        for sd in sudut:
            tab4.add(sd)

            scr = customtkinter.CTkScrollableFrame(tab4.tab(sd), orientation="horizontal", height=600)
            scr.pack(fill=tk.BOTH, expand=True)

            lb = customtkinter.CTkLabel(scr, text=f"Sudut: {str(sd)}")
            lb.pack(anchor='w')

            # textbox
            ctext = customtkinter.CTkTextbox(scr, width=6000, height=600)
            ctext.pack()

            datax = []
            if (sd == "0"):
                datax = setting.glcm_symm[:, :, 0, 0]
            elif (sd == "45"):
                datax = setting.glcm_symm[:, :, 0, 1]
            elif (sd == "90"):
                datax = setting.glcm_symm[:, :, 0, 2]
            elif (sd == "135"):
                datax = setting.glcm_symm[:, :, 0, 3]

            np.set_printoptions(suppress=True)
            pd.set_option('display.float_format', lambda x: '%.10f' % x)

            for x in reversed(datax):
                ctext.insert("0.0", x)
                ctext.insert("0.0", "\n")

        # no 5 hitung normalisasi matriks
        judul5 = customtkinter.CTkLabel(frame, text="5. Normalisasi matriks", font=(0, 14))
        judul5.pack(anchor="w", padx=10, pady=(20, 0))

        tab5 = customtkinter.CTkTabview(frame)
        tab5.pack(padx=20, fill=tk.X, expand=True)

        sudut = ["0", "45", "90", "135"]

        for sd in sudut:
            tab5.add(sd)

            scr = customtkinter.CTkScrollableFrame(tab5.tab(sd), orientation="horizontal", height=600)
            scr.pack(fill=tk.BOTH, expand=True)

            lb = customtkinter.CTkLabel(scr, text=f"Sudut: {str(sd)}")
            lb.pack(anchor='w')

            # textbox
            ctext = customtkinter.CTkTextbox(scr, width=14000, height=600)
            ctext.pack()

            datax = []
            if (sd == "0"):
                datax = setting.glcm[:, :, 0, 0]
            elif (sd == "45"):
                datax = setting.glcm[:, :, 0, 1]
            elif (sd == "90"):
                datax = setting.glcm[:, :, 0, 2]
            elif (sd == "135"):
                datax = setting.glcm[:, :, 0, 3]

            np.set_printoptions(suppress=True)
            pd.set_option('display.float_format', lambda x: '%.10f' % x)

            for x in reversed(datax):
                ctext.insert("0.0", x)
                ctext.insert("0.0", "\n")



        # no 6 hitung energy dkk
        judul6 = customtkinter.CTkLabel(frame,
                                        text="6. Hitung Energy, Homogeneity, Constrast, ASM, Correlation, dan Dissimiliarity",
                                        font=(0, 14))
        judul6.pack(anchor="w", padx=10, pady=(20, 0))

        tab6 = customtkinter.CTkTabview(frame)
        tab6.pack(padx=20, fill=tk.X, expand=True)

        sudut = ["0", "45", "90", "135"]

        for i, sd in enumerate(sudut):
            tab6.add(sd)

            frame_prop = customtkinter.CTkFrame(tab6.tab(sd))
            frame_prop.pack(fill=tk.X, expand=True)

            # dissimilarity
            dis_frame = customtkinter.CTkFrame(frame_prop)
            dis_frame.pack(fill=tk.X, expand=True, pady=10, padx=10)

            text1 = customtkinter.CTkLabel(dis_frame, text="a. Dissimilarity")
            text1.pack(anchor="w", padx=10)

            text2 = customtkinter.CTkLabel(dis_frame, text="              = ∑P(i,j) |i - j|     => i,j = 0 sampai 255 (level = 256)")
            text2.pack(anchor="w", padx=20)

            text3 = customtkinter.CTkLabel(dis_frame, text="              = P(0,0)*|0 - 0| + P(0,1)*|0 - 1| + ... + P(255,255)*|255 - 255|")
            text3.pack(anchor="w", padx=20)

            text4 = customtkinter.CTkLabel(dis_frame, text=f"              = {setting.glcm[:, :, 0, i][0][0]} + {setting.glcm[:, :, 0, i][0][1]}  + ... + {setting.glcm[:, :, 0, i][0][255]}")
            text4.pack(anchor="w", padx=20)

            text5 = customtkinter.CTkLabel(dis_frame, text=f"              = {setting.data_klasifikasi[0][i]}")
            text5.pack(anchor="w", padx=20)

            # correlation
            cor_frame = customtkinter.CTkFrame(frame_prop)
            cor_frame.pack(fill=tk.X, expand=True, pady=10, padx=10)

            text1 = customtkinter.CTkLabel(cor_frame, text="b. Correlation")
            text1.pack(anchor="w", padx=10)

            # mencari miu
            data = setting.glcm[:, :, 0, i]
            itot = 0
            jtot = 0
            sigmai = 0
            sigmaj = 0
            for ii, ix in enumerate(data):
                for j, jx in enumerate(ix):
                    itot += ii * jx
                    jtot += j * jx

                    sigmai += ((ii - (ii * jx))**2) * jx
                    sigmaj += ((ii - (j * jx)) ** 2) * jx

            sigmai = math.sqrt(sigmai)
            sigmaj = math.sqrt(sigmaj)

            itot = round(itot, 4)
            jtot = round(jtot, 4)
            sigmai = round(sigmai, 4)
            sigmaj = round(sigmaj, 4)

            text2jdl = customtkinter.CTkLabel(cor_frame, text="* Menghitung nilai μ")
            text2jdl.pack(anchor="w", padx=20)
            text2a = customtkinter.CTkLabel(cor_frame, text="μi = SUM(i * P(i))")
            text2a.pack(anchor="w", padx=30)
            text2b = customtkinter.CTkLabel(cor_frame, text=f"μi = {itot}")
            text2b.pack(anchor="w", padx=30)

            text2c = customtkinter.CTkLabel(cor_frame, text="\nμj = SUM(j * P(j))")
            text2c.pack(anchor="w", padx=30)
            text2d = customtkinter.CTkLabel(cor_frame, text=f"μj = {jtot}")
            text2d.pack(anchor="w", padx=30)

            text3jdl = customtkinter.CTkLabel(cor_frame, text="* Menghitung nilai σ")
            text3jdl.pack(anchor="w", padx=20)
            text3a = customtkinter.CTkLabel(cor_frame, text="σi = (i - μi)^2 * Pi")
            text3a.pack(anchor="w", padx=30)
            text3b = customtkinter.CTkLabel(cor_frame, text=f"σi = {sigmai}")
            text3b.pack(anchor="w", padx=30)

            text3c = customtkinter.CTkLabel(cor_frame, text="\nσj = (j - μj)^2 * Pj")
            text3c.pack(anchor="w", padx=30)
            text3d = customtkinter.CTkLabel(cor_frame, text=f"σj = {sigmaj}")
            text3d.pack(anchor="w", padx=30)

            text2 = customtkinter.CTkLabel(cor_frame,
                                           text="Correlation = ∑P(i,j) * (((i - μi)(j - μj)) / (sqrt((σi^2)(σj^2))))       => i,j = 0 sampai 255 (level = 256)")
            text2.pack(anchor="w", padx=20)


            text3 = customtkinter.CTkLabel(cor_frame,
                                           text=f"             = {setting.glcm[:, :, 0, i][0][0]} * (((0 - {itot})(0 - {jtot})) / (sqrt(({sigmai}^2)({sigmaj}^2)))) + ... + {setting.glcm[:, :, 0, i][0][255]} * (((255 - {itot})(255 - {jtot})) / (sqrt(({sigmai}^2)({sigmaj}^2))))")
            text3.pack(anchor="w", padx=10)

            text4 = customtkinter.CTkLabel(cor_frame,
                                           text=f"              = {setting.glcm[:, :, 0, i][0][0]} + {setting.glcm[:, :, 0, i][0][1]}  + ... + {setting.glcm[:, :, 0, i][0][255]}")
            text4.pack(anchor="w", padx=10)

            text5 = customtkinter.CTkLabel(cor_frame, text=f"              = {setting.data_klasifikasi[0][i+4]}")
            text5.pack(anchor="w", padx=10)

            #homogeneity
            hom_frame = customtkinter.CTkFrame(frame_prop)
            hom_frame.pack(fill=tk.X, expand=True, pady=10, padx=10)

            text1 = customtkinter.CTkLabel(hom_frame, text="c. Homogeneity")
            text1.pack(anchor="w", padx=10)

            text2 = customtkinter.CTkLabel(hom_frame,
                                           text="              = ∑P(i,j) / (1+(i-j)^2)     => i,j = 0 sampai 255 (level = 256)")
            text2.pack(anchor="w", padx=20)

            text3 = customtkinter.CTkLabel(hom_frame,
                                           text=f"              = ({setting.glcm[:, :, 0, i][0][0]} / (1+(0-0)^2)) + ({setting.glcm[:, :, 0, i][0][1]} / (1+(0-1)^2)) + ... + ({setting.glcm[:, :, 0, i][0][255]} / (1+(255-255)^2))")
            text3.pack(anchor="w", padx=20)

            text4 = customtkinter.CTkLabel(hom_frame,
                                           text=f"              = ({setting.glcm[:, :, 0, i][0][0] / (1+(0-0)^2)}) + ({setting.glcm[:, :, 0, i][0][1] / (1+(0-1)^2)}) + ... + ({setting.glcm[:, :, 0, i][0][255] / (1+(255-255)^2)})")
            text4.pack(anchor="w", padx=20)

            text5 = customtkinter.CTkLabel(hom_frame,
                                           text=f"              = {setting.data_klasifikasi[0][i+8]}")
            text5.pack(anchor="w", padx=20)

            #Contrast
            con_frame = customtkinter.CTkFrame(frame_prop)
            con_frame.pack(fill=tk.X, expand=True, pady=10, padx=10)

            text1 = customtkinter.CTkLabel(con_frame, text="d. Contrast")
            text1.pack(anchor="w", padx=10)

            text2 = customtkinter.CTkLabel(con_frame,
                                           text="              = ∑Pij (i,j)^2      => i,j = 0 sampai 255 (level = 256)")
            text2.pack(anchor="w", padx=20)

            text3 = customtkinter.CTkLabel(con_frame,
                                           text=f"              = ({setting.glcm[:, :, 0, i][0][0]} (0-0)^2) + ({setting.glcm[:, :, 0, i][0][0]} (0-1)^2) + ... + ({setting.glcm[:, :, 0, i][0][255]} (255-255)^2)")
            text3.pack(anchor="w", padx=20)

            text4 = customtkinter.CTkLabel(con_frame,
                                           text=f"              = {setting.glcm[:, :, 0, i][0][0]*((0-0)^2)} + {(setting.glcm[:, :, 0, i][0][0])*((0-1)^2)} + ... + {(setting.glcm[:, :, 0, i][0][255])*((255-255)^2)}")
            text4.pack(anchor="w", padx=20)

            text5 = customtkinter.CTkLabel(con_frame,
                                           text=f"              = {setting.data_klasifikasi[0][i+12]}")
            text5.pack(anchor="w", padx=20)

            # ASM
            asm_frame = customtkinter.CTkFrame(frame_prop)
            asm_frame.pack(fill=tk.X, expand=True, pady=10, padx=10)

            text1 = customtkinter.CTkLabel(asm_frame, text="e. ASM")
            text1.pack(anchor="w", padx=10)

            text2 = customtkinter.CTkLabel(asm_frame,
                                           text="              = ∑Pij^2      => i,j = 0 sampai 255 (level = 256)")
            text2.pack(anchor="w", padx=20)

            text3 = customtkinter.CTkLabel(asm_frame,
                                           text=f"              = ({setting.glcm[:, :, 0, i][0][0]})^2 + ({setting.glcm[:, :, 0, i][0][0]})^2 + ... + ({setting.glcm[:, :, 0, i][0][255]})^2")
            text3.pack(anchor="w", padx=20)

            text4 = customtkinter.CTkLabel(asm_frame,
                                           text=f"              = {setting.glcm[:, :, 0, i][0][0]**2} + {setting.glcm[:, :, 0, i][0][0]**2} + ... + {setting.glcm[:, :, 0, i][0][255]**2}")
            text4.pack(anchor="w", padx=20)

            text5 = customtkinter.CTkLabel(asm_frame,
                                           text=f"              = {setting.data_klasifikasi[0][i+16]}")
            text5.pack(anchor="w", padx=20)

            # energy
            ene_frame = customtkinter.CTkFrame(frame_prop)
            ene_frame.pack(fill=tk.X, expand=True, pady=10, padx=10)

            text1 = customtkinter.CTkLabel(ene_frame, text="f. Energy")
            text1.pack(anchor="w", padx=10)

            text2 = customtkinter.CTkLabel(ene_frame,
                                           text="              = SQRT(ASM)")
            text2.pack(anchor="w", padx=20)

            text3 = customtkinter.CTkLabel(ene_frame,
                                           text=f"              = {setting.data_klasifikasi[0][i+20]}")
            text3.pack(anchor="w", padx=20)

    def detail_pnn(self):
        for widget in content_frame.winfo_children():
            widget.destroy()

        # title
        title = customtkinter.CTkFrame(content_frame)
        title.pack(fill=customtkinter.X, pady=(20, 0), padx=20)

        # tombol back
        back = customtkinter.CTkButton(title, text="Kembali", command=lambda: self.kembali("Klasifikasi Citra"))
        back.pack(side=customtkinter.LEFT, anchor="w", padx=20, pady=20)

        # judul
        judul = customtkinter.CTkLabel(title, text="Detail Perhitungan Pseudo Nearest Neighbor")
        judul.pack(side=customtkinter.RIGHT, pady=20, fill=customtkinter.X, padx=20)

        # frame untuk detail perhitungan
        frame = customtkinter.CTkScrollableFrame(content_frame)
        frame.pack(fill=customtkinter.BOTH, padx=20, pady=20, expand=True)

        # no 1 citra grayscale
        judul1 = customtkinter.CTkLabel(frame, text="1. Hasil Ekstraksi GLCM", font=(0, 14))
        judul1.pack(anchor="w", padx=10)

        data_glcm = []

        data_glcm.append(("Sudut", "Dissimilarity", "Correlation", "Homogeneity", "Contrast", "ASM", "Energy"))

        if (len(setting.data_klasifikasi) > 0):
            hasil_glcm = setting.data_klasifikasi[0]

            sudut = 0
            for i in range(4):
                data_glcm.append((str(sudut), hasil_glcm[i], hasil_glcm[i + 4], hasil_glcm[i + 8],
                                  hasil_glcm[i + 12], hasil_glcm[i + 16], hasil_glcm[i + 2]))
                sudut += 45

        else:
            sudut = 0
            for i in range(4):
                data_glcm.append((str(sudut), '0', '0', '0', '0', '0', '0'))
                sudut += 45

        glcm_tb = CTkTable(frame, row=5, column=7, values=data_glcm, header_color="light blue")
        glcm_tb.pack(padx=20, pady=10)

        # no 2 Matriks citra

        judul2 = customtkinter.CTkLabel(frame, text="2. Data Ekstraksi GLCM dari Dataset", font=(0, 14))
        judul2.pack(anchor="w", padx=10, pady=(20, 0))

        df = pd.read_csv(setting.csv_save)

        l1 = list(df)  # List of column names as header

        r_set = df.to_numpy().tolist()

        trv = ttk.Treeview(frame, selectmode='browse', height=10,
                           show='headings', columns=l1)
        trv.pack(pady=20, padx=20, fill=customtkinter.BOTH, expand=True, side=customtkinter.TOP, anchor='w')

        for i in l1:
            trv.column(i, width=18, anchor='w')
            trv.heading(i, text=str(i))
        idx = 0
        for dt in r_set:
            v = [r for r in dt]
            trv.insert("", 'end', iid=idx, values=v)
            idx += 1

        # no 3 matriks ordo 255*255
        judul3 = customtkinter.CTkLabel(frame, text="3. Pisah Data Berdasarkan Kelas", font=(0, 14))
        judul3.pack(anchor="w", padx=10, pady=(20, 0))

        fr3 = customtkinter.CTkFrame(frame)
        fr3.pack(anchor="w", padx=10, fill=customtkinter.X, expand=True)

        tab3 = customtkinter.CTkTabview(fr3)
        tab3.pack(anchor="w", padx=10, pady=(20, 0), fill=customtkinter.X, expand=True)

        for i in range(len(setting.data_jarak)):
            tab3.add(setting.data_jarak[i][0])

            lb = customtkinter.CTkLabel(master=tab3.tab(setting.data_jarak[i][0]),
                                        text=f"Data {setting.data_jarak[i][0]}")
            lb.pack()

            df = pd.read_csv(setting.csv_save)

            x = setting.data_jarak[i][0].replace(" ", "-")

            grouped = df.groupby(["label"])
            group = grouped.get_group(x)

            l1 = list(df)  # List of column names as header

            r_set = group.to_numpy().tolist()

            trv = ttk.Treeview(tab3.tab(setting.data_jarak[i][0]), selectmode='browse', height=10,
                               show='headings', columns=l1)
            trv.pack(pady=20, padx=20, fill=customtkinter.BOTH, expand=True, side=customtkinter.TOP, anchor='w')

            for i in l1:
                trv.column(i, width=18, anchor='w')
                trv.heading(i, text=str(i))
            idx = 0
            for dt in r_set:
                v = [r for r in dt]
                trv.insert("", 'end', iid=idx, values=v)
                idx += 1

        # no 4 Perhitungan sudut
        judul4 = customtkinter.CTkLabel(frame, text="4. Hitung Jarak Euclidean Masing-Masing Kelas", font=(0, 14))
        judul4.pack(anchor="w", padx=10, pady=(20, 0))

        fr3 = customtkinter.CTkFrame(frame)
        fr3.pack(anchor="w", padx=10, fill=customtkinter.X, expand=True)

        tab4 = customtkinter.CTkTabview(fr3)
        tab4.pack(anchor="w", padx=10, pady=(20, 0), fill=customtkinter.X, expand=True)

        # rows, cols = (len(df), len(list(df)))
        # data_euc = [[0] * rows] * cols

        data_euc_all = []

        euc_index = 0
        for i in range(len(setting.data_jarak)):
            tab4.add(setting.data_jarak[i][0])

            lb = customtkinter.CTkLabel(master=tab4.tab(setting.data_jarak[i][0]),
                                        text=f"Data {setting.data_jarak[i][0]}")
            lb.pack()

            df = pd.read_csv(setting.csv_save)

            x = setting.data_jarak[i][0].replace(" ", "-")

            grouped = df.groupby(["label"])
            group = grouped.get_group(x)

            l1 = ("#1", "#2")

            r_set = group.to_numpy().tolist()

            trv = ttk.Treeview(tab4.tab(setting.data_jarak[i][0]), selectmode='browse', height=10,
                               show='headings', columns=l1)
            trv.pack(pady=20, padx=20, fill=customtkinter.BOTH, expand=True, side=customtkinter.TOP, anchor='w')

            trv.column("#1", width=60, stretch=customtkinter.NO)
            trv.heading("#2", text="Perhitungan Jarak Euclidean")
            trv.heading("#1", text="Hasil")

            dk = setting.data_klasifikasi[0]

            index_ec = 0
            data_euc = []
            for dt in r_set:
                ec = f"sqrt(({dt[0]} - {dk[0]})^2 + ({dt[1]} - {dk[1]})^2 + ({dt[2]} - {dk[2]})^2 + ({dt[3]} - {dk[3]})^2 + ({dt[4]} - {dk[4]})^2 + ({dt[5]} - {dk[5]})^2 + ({dt[6]} - {dk[6]})^2 + ({dt[7]} - {dk[7]})^2 + ({dt[8]} - {dk[8]})^2 + ({dt[9]} - {dk[9]})^2 + ({dt[10]} - {dk[10]})^2 + ({dt[11]} - {dk[11]})^2 + ({dt[12]} - {dk[12]})^2 + ({dt[13]} - {dk[13]})^2 + ({dt[14]} - {dk[14]})^2 + ({dt[15]} - {dk[15]})^2 + ({dt[16]} - {dk[16]})^2 + ({dt[17]} - {dk[17]})^2 + ({dt[18]} - {dk[18]})^2 + ({dt[19]} - {dk[19]})^2 + ({dt[20]} - {dk[20]})^2 + ({dt[21]} - {dk[21]})^2 + ({dt[22]} - {dk[22]})^2 + ({dt[23]} - {dk[23]})^2)"
                hasil = math.sqrt(
                    (dt[0] - dk[0]) ** 2 + (dt[1] - dk[1]) ** 2 + (dt[2] - dk[2]) ** 2 + (dt[3] - dk[3]) ** 2 + (
                            dt[4] - dk[4]) ** 2 + (dt[5] - dk[5]) ** 2 + (dt[6] - dk[6]) ** 2 + (
                            dt[7] - dk[7]) ** 2 + (dt[8] - dk[8]) ** 2 + (dt[9] - dk[9]) ** 2 + (
                            dt[10] - dk[10]) ** 2 + (dt[11] - dk[11]) ** 2 + (dt[12] - dk[12]) ** 2 + (
                            dt[13] - dk[13]) ** 2 + (dt[14] - dk[14]) ** 2 + (dt[15] - dk[15]) ** 2 + (
                            dt[16] - dk[16]) ** 2 + (dt[17] - dk[17]) ** 2 + (dt[18] - dk[18]) ** 2 + (
                            dt[19] - dk[19]) ** 2 + (dt[20] - dk[20]) ** 2 + (dt[21] - dk[21]) ** 2 + (
                            dt[22] - dk[22]) ** 2 + (dt[23] - dk[23]) ** 2)

                data_euc.append(hasil)
                index_ec += 1
                trv.insert("", 'end', iid=idx, values=(hasil, ec))
                idx += 1

            euc_index += 1
            data_euc_all.append(data_euc)


        # no 5 hitung symmetrical matriks
        judul4 = customtkinter.CTkLabel(frame, text="5. Urutkan Rata-rata Jarak Euclidean Masing-masing Kelas",
                                        font=(0, 14))
        judul4.pack(anchor="w", padx=10, pady=(20, 0))


        fr3 = customtkinter.CTkFrame(frame)
        fr3.pack(anchor="w", padx=10, fill=customtkinter.X, expand=True)

        tab4 = customtkinter.CTkTabview(fr3)
        tab4.pack(anchor="w", padx=10, pady=(20, 0), fill=customtkinter.X, expand=True)

        rata_k = []

        for i in range(len(setting.data_jarak)):
            tab4.add(setting.data_jarak[i][0])

            lb = customtkinter.CTkLabel(master=tab4.tab(setting.data_jarak[i][0]),
                                        text=f"Data {setting.data_jarak[i][0]}")
            lb.pack()

            l1 = ("#1", "#2")

            trv = ttk.Treeview(tab4.tab(setting.data_jarak[i][0]), selectmode='browse', height=10,
                               show='headings', columns=l1)
            trv.pack(pady=20, padx=20, side=customtkinter.LEFT, anchor='n')

            trv.column("#1", width=60, stretch=customtkinter.NO)
            trv.heading("#2", text="Jarak Euclidean")
            trv.heading("#1", text="No")

            # urutkan array
            data_euc_all[i].sort()

            id_k = 0
            rata_kk = []
            all_jarak = ""
            for j in range(len(data_euc_all[i])):
                trv.insert("", 'end', iid=j, values=(j+1, data_euc_all[i][j]))
                if(id_k < int(setting.num_neighbors)):
                    if(id_k == 0):
                        all_jarak = str(data_euc_all[i][j])
                    else:
                        all_jarak = all_jarak + " + " + str(round(data_euc_all[i][j], 2))
                    rata_kk.append(data_euc_all[i][j])
                id_k += 1

            rata_k.append(rata_kk)

            # kanan
            f_kanan = customtkinter.CTkFrame(tab4.tab(setting.data_jarak[i][0]))
            f_kanan.pack(side=customtkinter.LEFT, anchor='n', fill=customtkinter.X, expand=True, pady=20, padx=(0, 20))

            # kanan
            k_label = customtkinter.CTkLabel(f_kanan, text=f"Nilai K: {setting.num_neighbors}", justify="left")
            k_label.pack(side=customtkinter.TOP, anchor='w', padx=20, pady=(20, 0))


            r_label = customtkinter.CTkLabel(f_kanan, text=f"Rata-rata jarak: ({all_jarak})/{setting.num_neighbors}", justify="left")
            r_label.pack(side=customtkinter.TOP, anchor='w', padx=20)

            r_label2 = customtkinter.CTkLabel(f_kanan, text=f"Rata-rata jarak: {setting.data_jarak[i][1]}", justify="left")
            r_label2.pack(side=customtkinter.TOP, anchor='w', padx=20, pady=(0, 20))


        # no 6 hitung normalisasi matriks
        judul4 = customtkinter.CTkLabel(frame, text="6. Hasil Klasifikasi ", font=(0, 14))
        judul4.pack(anchor="w", padx=10, pady=(20, 0))

        frame_6 = customtkinter.CTkFrame(frame)
        frame_6.pack()

        l1 = ("#1", "#2", "#3")

        trv = ttk.Treeview(frame_6, height=10, show='headings', columns=l1)
        trv.pack(pady=20, padx=20, side=customtkinter.LEFT, anchor='n')

        trv.column("#1", width=40, stretch=customtkinter.NO)
        trv.heading("#1", text="No")
        trv.heading("#2", text="Nama Batik")
        trv.heading("#3", text="Rata-rata Jarak Euclidean")


        for i, r_jarak in enumerate(setting.data_jarak):
            trv.insert("", 'end', iid=i, values=(str(i+1), str(r_jarak[0]), str(r_jarak[1])))


    # klasifikasi
    def Klasifikasi(self):

        def cek():
            #
            if(setting.Klasifikasi_stat == True):
                show_img(setting.img_dir)
                update_hasil()

            citra_entry.insert(0, setting.img_dir)
            citra_entry.configure(state='readonly')
            k_entry.insert(0, setting.num_neighbors)

            # panggil fungsi untuk tabel
            table_glcm()
            tabel_jarak()

        def show_img(file_path):
            try:
                if file_path:
                    image = cv2.imread(file_path)
                    setting.img = image

                    # perkecil skala 0.5
                    width = int(image.shape[1] * 0.5)
                    height = int(image.shape[0] * 0.5)

                    # keep aspect ratio
                    if (width > height):
                        ratio = height / width
                        new_width = 100
                        new_height = int(ratio * new_width)
                        dim = (new_width, new_height)
                    else:
                        ratio = width / height
                        new_height = 100
                        new_width = int(ratio * new_height)
                        dim = (new_width, new_height)

                    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                    img_cr = glcm.cropping(image)
                    img_gr = glcm.grayscale(img_cr)

                    # get img gr untuk disimpan di setting
                    setting.img_dir = file_path
                    setting.img_grayscalle = img_gr

                    image_cr = Image.fromarray(img_cr)
                    image_gr = Image.fromarray(img_gr)
                    image_customtkinter = Image.fromarray(resized)

                    citra_entry.delete(0, 'end')
                    citra_entry.configure(state='normal')
                    citra_entry.insert(0, str(file_path))
                    setting.img_dir = str(file_path)
                    citra_entry.configure(state='readonly')

                    img1 = customtkinter.CTkImage(light_image=image_customtkinter, size=dim)
                    img2 = customtkinter.CTkImage(light_image=image_cr, size=(100, 100))
                    img3 = customtkinter.CTkImage(light_image=image_gr, size=(100, 100))

                    image_label2.configure(image=img1)
                    image_label3.configure(image=img2)
                    image_label4.configure(image=img3)


            except Exception as error:
                CTkMessagebox(title="Error",
                              message=error,
                              icon="cancel")


        # fungsi untuk memilih gambar
        def open_image():
            file_path = filedialog.askopenfilename()
            show_img(file_path)

        def klasifikasi():

            try:
                setting.num_neighbors = k_entry.get()

                csv_save = setting.csv_save
                num_neighbors = k_entry.get()

                img = glcm.preprocessing(setting.img)

                klas.proses_utama(csv_save, int(num_neighbors), img, setting)

                setting.Klasifikasi_stat = True

                # update tabel
                table_glcm()
                tabel_jarak()
                update_hasil()

            except:
                if (setting.img == ""):
                    msg = "Silahkan pilih citra terlebih dahulu"
                else:
                    msg = "Nilai k tidak valid"

                CTkMessagebox(title="Error",
                              message=msg,
                              icon="cancel")

        def update_hasil():
            # clear img_hasil
            for widget in fr2_rb1.winfo_children():
                widget.destroy()

            hasil = setting.data_klasifikasi[1]
            sub = hasil.replace(" ", "-")
            path = os.path.join(setting.folder, sub)
            file = os.listdir(path)

            index = 0
            for img in file:
                index += 1

                img_path = os.path.join(path, img)

                image = cv2.imread(img_path)

                # perkecil skala 0.5
                width = int(image.shape[1] * 0.5)
                height = int(image.shape[0] * 0.5)

                # keep aspect ratio
                if (width > height):
                    ratio = height / width
                    new_width = 100
                    new_height = int(ratio * new_width)
                    dim = (new_width, new_height)
                else:
                    ratio = width / height
                    new_height = 100
                    new_width = int(ratio * new_height)
                    dim = (new_width, new_height)

                resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                imagef = Image.fromarray(resized)

                img1 = customtkinter.CTkImage(light_image=imagef, size=dim)

                image1 = customtkinter.CTkLabel(fr2_rb1, width=100, height=100, fg_color="#fff", text=" ")
                image1.pack(padx=10, pady=20, side=customtkinter.LEFT)

                image1.configure(image=img1)

                if index == 3:
                    break

            h_label2.configure(text=setting.data_klasifikasi[1])


        # kiri / atas
        f_kiri = customtkinter.CTkFrame(content_frame, fg_color="transparent")
        f_kiri.pack(side=customtkinter.TOP, fill=customtkinter.X, pady=20, padx=20)

        # frame col 1
        r1 = customtkinter.CTkFrame(f_kiri)
        r1.pack(side=customtkinter.LEFT, anchor='n', padx=(0, 10))

        # frame untuk pack
        inp_frame = customtkinter.CTkFrame(r1, fg_color="transparent")
        inp_frame.pack(side=customtkinter.LEFT, anchor='n', fill=customtkinter.X, padx=10, pady=10)

        # frame untuk pengaturan folder
        fd_frame = customtkinter.CTkFrame(inp_frame, fg_color="transparent")
        fd_frame.pack(fill=customtkinter.BOTH)

        citra_label = customtkinter.CTkLabel(fd_frame, text="Citra:")
        citra_label.pack(padx=(20, 0), pady=10, side=customtkinter.LEFT)

        citra_entry = customtkinter.CTkEntry(fd_frame, width=230)
        citra_entry.pack(padx=(20, 10), pady=5, side=customtkinter.LEFT)

        label2 = customtkinter.CTkButton(fd_frame, text="Pilih Citra", width=40, command=open_image)
        label2.pack(pady=10, padx=10, side=customtkinter.LEFT)

        # frame untuk bawahnya
        k_frame = customtkinter.CTkFrame(inp_frame, fg_color="transparent")
        k_frame.pack(fill=customtkinter.BOTH, padx=20)

        k_jarak = customtkinter.CTkLabel(k_frame, text="Nilai K:")
        k_jarak.pack(side=customtkinter.LEFT)

        k_entry = customtkinter.CTkEntry(k_frame, width=320)
        k_entry.pack(side=customtkinter.LEFT, anchor='w', padx=(10, 0))

        # frame untuk tombol GLCM
        t_frame = customtkinter.CTkFrame(inp_frame, fg_color="transparent")
        t_frame.pack(fill=customtkinter.BOTH, padx=20)

        btn_proc = customtkinter.CTkButton(t_frame, text="Proses Klasifikasi", command=klasifikasi)
        btn_proc.pack(pady=(14, 10), side=customtkinter.RIGHT, fill=customtkinter.X)

        # col 2 -> citra asli
        r2 = customtkinter.CTkFrame(f_kiri, fg_color="transparent")
        r2.pack(side=customtkinter.LEFT, anchor='n', padx=(30, 20))

        kl_frame2 = customtkinter.CTkFrame(r2)
        kl_frame2.pack(anchor='w', side=customtkinter.LEFT)

        image_label2 = customtkinter.CTkLabel(kl_frame2, width=100, height=100, fg_color="#fff", text="")
        image_label2.pack(padx=30, pady=(20, 0))

        lb_prep2 = customtkinter.CTkLabel(kl_frame2, text="Citra asli")
        lb_prep2.pack()

        # col 3 -> citra grayscale
        r3 = customtkinter.CTkFrame(f_kiri, fg_color="transparent")
        r3.pack(side=customtkinter.LEFT, anchor='n', padx=(20, 20))

        kl_frame3 = customtkinter.CTkFrame(r3)
        kl_frame3.pack(anchor='w', side=customtkinter.LEFT)

        image_label3 = customtkinter.CTkLabel(kl_frame3, width=100, height=100, fg_color="#fff", text="")
        image_label3.pack(padx=30, pady=(20, 0))

        lb_prep3 = customtkinter.CTkLabel(kl_frame3, text="Citra cropping")
        lb_prep3.pack()

        # col 4 -> citra asli
        r4 = customtkinter.CTkFrame(f_kiri, fg_color="transparent")
        r4.pack(side=customtkinter.RIGHT, anchor='n', padx=(20, 0))

        kl_frame4 = customtkinter.CTkFrame(r4)
        kl_frame4.pack(anchor='w', side=customtkinter.LEFT)

        image_label4 = customtkinter.CTkLabel(kl_frame4, width=100, height=100, fg_color="#fff", text="")
        image_label4.pack(padx=30, pady=(20, 0))

        lb_prep4 = customtkinter.CTkLabel(kl_frame4, text="Citra grayscale")
        lb_prep4.pack()

        def table_glcm():

            for widget in fr2.winfo_children():
                widget.destroy()

            data_glcm = []

            data_glcm.append(("Sudut", "Dissimilarity", "Correlation", "Homogeneity", "Contrast", "ASM", "Energy"))

            if (len(setting.data_klasifikasi) > 0):
                hasil_glcm = setting.data_klasifikasi[0]

                sudut = 0
                for i in range(4):
                    data_glcm.append((str(sudut), hasil_glcm[i], hasil_glcm[i + 4], hasil_glcm[i + 8],
                                      hasil_glcm[i + 12], hasil_glcm[i + 16], hasil_glcm[i + 20]))
                    sudut += 45

            else:
                sudut = 0
                for i in range(4):
                    data_glcm.append((str(sudut), '0', '0', '0', '0', '0', '0'))
                    sudut += 45

            glcm_tb = CTkTable(fr2, row=5, column=7, values=data_glcm, header_color="light blue")
            glcm_tb.pack(padx=20, pady=10)

        # grafik
        def tabel_jarak():

            for widget in rb2.winfo_children():
                widget.destroy()

            # frame untuk tombol detail
            pnn_fr = customtkinter.CTkFrame(rb2)
            pnn_fr.pack(padx=20, pady=(20, 0), side=customtkinter.TOP, expand=True, anchor="e")

            btn_pnn = customtkinter.CTkButton(pnn_fr, text="Detail Perhitungan PNN", command=lambda: self.detail_pnn())
            btn_pnn.pack(side=customtkinter.RIGHT)

            # frame scroll untuk tabel jarak
            scr_jarak = customtkinter.CTkScrollableFrame(rb2)
            scr_jarak.pack(padx=20, pady=(10, 20), fill=customtkinter.BOTH, expand=True)

            data_glcm = []
            data_glcm.append(("No", "Nama", "Jarak"))

            if (len(setting.data_jarak) > 0):

                setting.data_jarak.sort(key=lambda x: x[1])

                for i in range(len(setting.data_jarak)):
                    data_glcm.append((i + 1, setting.data_jarak[i][0], setting.data_jarak[i][1]))

            else:
                data_glcm.append((1, 'mega mendung', 0))
                data_glcm.append((2, 'kawung', 0))
                data_glcm.append((3, 'parang rusak', 0))
                data_glcm.append((4, 'bukan batik', 0))
                data_glcm.append((5, 'insang', 0))
                data_glcm.append((6, 'dayak', 0))
                data_glcm.append((7, 'poleng', 0))
                data_glcm.append((8, 'ikat celup', 0))

            # for data in data_glcm:
            #     trv_glcm.insert('', customtkinter.END, values=data)
            #
            # trv_glcm.pack(pady=20, padx=20, fill=customtkinter.X)

            tb_jarak = CTkTable(scr_jarak, row=len(data_glcm), column=3, values=data_glcm)
            tb_jarak.pack(padx=20, pady=20)

        # frame untuk tabel
        f_kanan = customtkinter.CTkFrame(content_frame)
        f_kanan.pack(anchor='w', fill=customtkinter.X, padx=20, pady=(0, 0))

        fr1 = customtkinter.CTkFrame(f_kanan, fg_color="transparent")
        fr1.pack(side=customtkinter.TOP, fill=customtkinter.X, padx=20, pady=(20, 0))

        glcm_label = customtkinter.CTkLabel(fr1, text="Hasil ekstraksi GLCM")
        glcm_label.pack(side=customtkinter.LEFT)

        glcm_btn = customtkinter.CTkButton(fr1, text="Detail Perhitungan GLCM", command=lambda: self.detail_glcm())
        glcm_btn.pack(side=customtkinter.RIGHT)

        # frame untuk tabel

        fr2 = customtkinter.CTkFrame(f_kanan, fg_color="transparent", bg_color="transparent")
        fr2.pack(side=customtkinter.TOP)

        # end untuk tabel glcm

        # untuk frame bawah
        f_bawah = customtkinter.CTkFrame(content_frame, fg_color="transparent", bg_color="transparent")
        f_bawah.pack(padx=20, pady=20, fill=customtkinter.BOTH)

        # bawah kiri
        rb1 = customtkinter.CTkFrame(f_bawah)
        rb1.pack(side=customtkinter.LEFT, fill=customtkinter.BOTH)

        fr1_rb1 = customtkinter.CTkFrame(rb1, fg_color="transparent")
        fr1_rb1.pack(side=customtkinter.TOP, pady=20, padx=20, fill=customtkinter.X)

        h_label = customtkinter.CTkLabel(fr1_rb1, text="Hasil Klasifikasi")
        h_label.pack(side=customtkinter.LEFT)

        hasl_fr = customtkinter.CTkFrame(fr1_rb1, fg_color="blue")
        hasl_fr.pack(side=customtkinter.RIGHT)

        h_label2 = customtkinter.CTkLabel(hasl_fr, text="Batik belum diklasifikasi", fg_color="#fff")
        h_label2.pack(padx=20)


        # frame 2
        fr2_rb1 = customtkinter.CTkFrame(rb1, fg_color="transparent")
        fr2_rb1.pack(padx=20)

        image1 = customtkinter.CTkLabel(fr2_rb1, width=100, height=100, fg_color="#fff", text="")
        image1.pack(padx=10, pady=20, side=customtkinter.LEFT)

        image1 = customtkinter.CTkLabel(fr2_rb1, width=100, height=100, fg_color="#fff", text="")
        image1.pack(padx=5, pady=20, side=customtkinter.LEFT)

        image1 = customtkinter.CTkLabel(fr2_rb1, width=100, height=100, fg_color="#fff", text="")
        image1.pack(padx=10, pady=20, side=customtkinter.LEFT)

        # bawah kanan
        rb2 = customtkinter.CTkFrame(f_bawah)
        rb2.pack(side=customtkinter.RIGHT, expand=True, fill=customtkinter.BOTH, padx=(20, 0))

        cek()

    def on_click(self, CTkButton_name):
        self.clear_frame()
        if (CTkButton_name == "Dataset"):
            self.Dataset()
        elif (CTkButton_name == "Cropping"):
            self.Cropping()
        elif (CTkButton_name == "Grayscalling"):
            self.Grayscalling()
        elif (CTkButton_name == "GLCM"):
            self.GLCM()
        elif (CTkButton_name == "Pengujian Akurasi"):
            self.Pengujian()
        else:
            self.Klasifikasi()

    def create_button(self, text):
        return customtkinter.CTkButton(sidebar_frame, height=40, width=200, text=text,
                                       command=lambda: self.on_click(text))

    def create_sidebar(self):
        Buttons = ["Dataset", "GLCM", "Pengujian Akurasi", "Klasifikasi Citra"]
        for button_name in Buttons:
            button = self.create_button(button_name)
            button.pack(pady=(20, 0), padx=20)

    def constr(self):
        # CTkFrame utama untuk konten utama
        main_frame = customtkinter.CTkFrame(root)
        main_frame.pack(fill=customtkinter.BOTH, side=customtkinter.LEFT, expand=True)

        # CTkFrame untuk sidebar
        sidebar_frame = customtkinter.CTkFrame(main_frame, width=400)
        sidebar_frame.pack(side=customtkinter.LEFT, fill=customtkinter.BOTH, pady=20, padx=20)

        # CTkFrame untuk konten utama
        content_frame = customtkinter.CTkFrame(main_frame)
        content_frame.pack(side=customtkinter.LEFT, fill=customtkinter.BOTH, expand=True, padx=(0, 20), pady=20)

        def create_button(text):
            return customtkinter.CTkButton(sidebar_frame, height=40, width=200, text=text,
                                           command=lambda: self.on_click(text))

        def create_sidebar():
            Buttons = ["Dataset", "GLCM", "K-Fold Cross Validation", "Pengujian Akurasi", "Klasifikasi Citra"]
            for button_name in Buttons:
                button = create_button(button_name)
                button.pack(pady=(20, 0), padx=20)

        create_sidebar()


# custom customtkinterinter
customtkinter.set_appearance_mode("system")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

# Membuat instance customtkinterinter
root = customtkinter.CTk()
root.title("GLCM -> Pseudo Nearest Neighbor")
root.state('zoomed')
root.after(0, lambda: root.state('zoomed'))

# CTkFrame utama untuk konten utama
main_frame = customtkinter.CTkFrame(root)
main_frame.pack(fill=customtkinter.BOTH, side=customtkinter.LEFT, expand=True)

# CTkFrame untuk sidebar
sidebar_frame = customtkinter.CTkFrame(main_frame, width=400)
sidebar_frame.pack(side=customtkinter.LEFT, fill=customtkinter.BOTH, pady=20, padx=20)

# CTkFrame untuk konten utama
content_frame = customtkinter.CTkFrame(main_frame)
content_frame.pack(side=customtkinter.LEFT, fill=customtkinter.BOTH, expand=True, padx=(0, 20), pady=20)

setting = Setting()
gui = GUI()
glcm = Glcm()
ak = PnnAk()
klas = PnnData()

# Membuat sidebar dengan tombol-tombolnya
gui.create_sidebar()

# start
gui.on_click("Dataset")

# Menjalankan loop customtkinterinter
root.mainloop()
