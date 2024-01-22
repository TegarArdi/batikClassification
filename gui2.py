import os
import tkinter as tk
from os.path import exists
from tkinter import filedialog, ttk
from tkinter.messagebox import showinfo
import customtkinter

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
        self.num_neighbors = 5
        self.data_jarak = list()
        self.data_klasifikasi = list()

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

    # fungsi hapus widget di frame

    def clear_frame(self):
        for widgets in content_frame.winfo_children():
            widgets.destroy()

    # dataset
    def Dataset(self):

        # def clear_canvas():
        #     for widgets in frame_container.winfo_children():
        #         widgets.destroy()

        def set_data():
            if (setting.get_stat('1') == False):
                dt_entry.configure(state='normal')
                dt_entry.delete(0, customtkinter.END)
                dt_entry.insert(0, setting.folder)
                dt_entry.configure(state='readonly')
                # show_data(setting.get_folder())

        def pilih_folder():
            folder = filedialog.askdirectory()
            dt_entry.configure(state='normal')
            dt_entry.delete(0, customtkinter.END)
            dt_entry.insert(0, folder)
            dt_entry.configure(state='readonly')
            setting.folder = folder
            # show_data(folder)

        # def show_data(folder):
        #     clear_canvas()
        #     folder1 = os.listdir(folder)
        #
        #     for sub in folder1:
        #         folder2 = os.path.join(folder, sub)
        #
        #         image_count = 0
        #         columns = 19
        #
        #         img_row = customtkinter.CTkLabel(frame_container, text=sub)
        #         img_row.pack()
        #
        #         file = os.listdir(folder2)
        #
        #         for name in file:
        #             image_count += 1
        #             r, c = divmod(image_count - 1, columns)
        #             im = Image.open(os.path.join(folder2, name))
        #             resized = im.resize((50, 50), Image.Resampling.LANCZOS)
        #             tkimage = ImageTk.PhotoImage(resized)
        #             myvar = customtkinter.CTkLabel(img_row, image=tkimage)
        #             myvar.image = tkimage
        #             myvar.grid(row=r, column=c)

        # def on_frame_configure(canvas):
        #     canvas.configure(scrollregion=canvas.bbox("all"))
        #
        # def on_mousewheel(event):
        #     canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        # buat 2 frame lagi
        f_kiri = customtkinter.CTkLabel(content_frame)
        f_kiri.grid(column=0, row=0, pady=(20, 0), padx=10)

        kr1 = customtkinter.CTkFrame(f_kiri)
        kr1.grid(column=0, row=0, padx=10, pady=10)

        dt_label = customtkinter.CTkLabel(kr1, text="Folder:")
        dt_label.grid(column=0, row=0)

        dt_entry = customtkinter.CTkEntry(kr1, width=40)
        dt_entry.grid(column=0, row=1)

        dt_entry.configure(state='readonly')

        dt_btn = customtkinter.CTkButton(kr1, text="Pilih Folder", command=pilih_folder)
        dt_btn.grid(column=1, row=1, padx=20)

        # bawah
        f_kanan = customtkinter.CTkFrame(content_frame)
        f_kanan.grid(column=0, row=1, pady=(20, 0), padx=10)

        # canvas = tk.Canvas(f_kanan)
        # canvas.grid(column=0, row=0)

        # scrollbar = tk.Scrollbar(f_kanan, command=canvas.yview)
        # scrollbar.grid()

        # canvas.configure(yscrollcommand=scrollbar.set)

        # frame_container = customtkinter.CTkFrame(canvas)
        # canvas.create_window((0, 0), window=frame_container, anchor="nw")
        #
        # frame_container.bind("<Configure>", lambda event, canvas=canvas: on_frame_configure(canvas))
        #
        # set_data()

    # # glcm
    # def GLCM(self):
    #
    #     def mulai():
    #         j_entry.current(setting.jarak - 1)
    #         if (setting.glcm_stat == False):
    #             pass
    #             # for widgets in f_kanan.winfo_children():
    #             #     widgets.destroy()
    #         else:
    #             upload_file()
    #
    #         folder_entry.config(state='normal')
    #         folder_entry.insert(0, setting.folder)
    #         folder_entry.config(state='readonly')
    #
    #     # fungsi get csv
    #     def upload_file():
    #         dists = setting.jarak
    #
    #         df = glcm.glcm_avg(setting.folder, setting.csv_save, dists, root)
    #         l1 = list(df)  # List of column names as header
    #         lab_kolom.config(text=str(df.shape[1]))
    #         lab_baris.config(text=str(df.shape[0]))
    #         trv_refresh(df, l1)  # show Treeview
    #
    #     # treeview untuk menampilkan df
    #     def trv_refresh(df, l1):  # Refresh the Treeview to reflect changes
    #         r_set = df.to_numpy().tolist()  # create list of list using rows
    #         trv = ttk.Treeview(f_kanan, selectmode='browse', height=20,
    #                            show='headings', columns=l1)
    #         trv.pack(pady=(0, 10), padx=10, fill=customtkinter.BOTH, expand=True, side=customtkinter.LEFT, anchor='w')
    #
    #         for i in l1:
    #             trv.column(i, width=18, anchor='w')
    #             trv.heading(i, text=str(i))
    #         idx = 0
    #         for dt in r_set:
    #             v = [r for r in dt]
    #             trv.insert("", 'end', iid=idx, values=v)
    #             idx += 1
    #
    #     def hitung_glcm():
    #         folder = setting.folder
    #         csv = setting.csv_save
    #
    #         n_index = int(j_entry.get())
    #         setting.jarak = n_index
    #         jarak = n_index
    #
    #         glcm.load_dataset(self, folder, csv, jarak, root)
    #         setting.glcm_stat = True
    #
    #     def reset_glcm():
    #         csv = setting.csv_save
    #
    #         def exit_application():
    #             msg_box = customtkinter.messagebox.askquestion('Reset GLCM',
    #                                                 'Yakin untuk mereset hasil GLCM?',
    #                                                 icon='warning')
    #             if msg_box == 'yes':
    #                 customtkinter.messagebox.showinfo('Return', 'Data GLCM berhasil direset')
    #                 if os.path.exists(csv):
    #                     os.remove(csv)
    #             else:
    #                 customtkinter.messagebox.showinfo('Return', 'Batal direset')
    #
    #         exit_application()
    #
    #         self.on_click("GLCM")
    #
    #     # buat 2 frame lagi
    #     f_kiri = customtkinter.CTkLabel(content_frame, text="Ekstraksi Fitur Tekstur GLCM")
    #     f_kiri.pack(fill=customtkinter.X, pady=(20, 0), padx=10)
    #
    #     f_kanan = customtkinter.CTkLabel(content_frame, text="Hasil Ekstraksi Fitur Tekstur GLCM")
    #     f_kanan.pack(fill=customtkinter.BOTH, expand=True, pady=(20, 0), padx=10)
    #
    #     # buat 2 frame untuk tombol tertentu
    #     r1_kiri = customtkinter.CTkFrame(f_kiri)
    #     r1_kiri.pack(side=customtkinter.LEFT, anchor='w', pady=(0, 20))
    #
    #     r1_kanan = customtkinter.CTkFrame(f_kiri)
    #     r1_kanan.pack(side=customtkinter.LEFT, anchor='w', fill=customtkinter.X, expand=True)
    #
    #     # di dalam frame kiri
    #     # frame untuk pengaturan folder
    #     fd_frame = customtkinter.CTkFrame(r1_kiri)
    #     fd_frame.pack(fill=customtkinter.BOTH, pady=(10, 0))
    #
    #     folder_label = customtkinter.CTkLabel(fd_frame, text="Folder:")
    #     folder_label.pack(padx=20, pady=10, side=customtkinter.LEFT)
    #     folder_entry = customtkinter.CTkEntry(fd_frame, width=50)
    #     folder_entry.pack(padx=2, pady=5, side=customtkinter.LEFT)
    #     folder_entry.config(readonlybackground='#fff', state='readonly')
    #
    #     # frame untuk bawahnya
    #     j_frame = customtkinter.CTkFrame(r1_kiri)
    #     j_frame.pack(fill=customtkinter.BOTH, padx=20)
    #
    #     l_jarak = customtkinter.CTkLabel(j_frame, text="Jarak:")
    #     l_jarak.pack(side=customtkinter.LEFT)
    #
    #     j_entry = ttk.Combobox(j_frame, width=20, state='readonly')
    #     j_entry['values'] = ('1', '2', '3', '4', '5')
    #     j_entry.pack(side=customtkinter.LEFT, padx=(28, 0))
    #
    #     # frame untuk tombol GLCM
    #     t_frame = customtkinter.CTkFrame(r1_kanan)
    #     t_frame.pack(fill=customtkinter.BOTH, expand=True)
    #
    #     btn_proc = customtkinter.CTkButton(t_frame, text="Hitung GLCM", width=12, height=3, command=hitung_glcm)
    #     btn_proc.pack(pady=15, padx=10, side=customtkinter.LEFT)
    #
    #     btn_delete = customtkinter.CTkButton(t_frame, text="Reset", width=12, height=3, command=reset_glcm)
    #     btn_delete.pack(pady=15, padx=10, side=customtkinter.RIGHT)
    #
    #     lab_hasil = customtkinter.CTkLabel(f_kanan, text="Hasil Ekstraksi Fitur GLCM", font=("Arial", 16))
    #     lab_hasil.pack(pady=(10, 0))
    #
    #     # buat frame untuk keterangan
    #     ket_frame = customtkinter.CTkFrame(f_kanan)
    #     ket_frame.pack(pady=(0, 10), padx=10, fill=customtkinter.BOTH)
    #
    #     # kolom
    #     kol_lframe = customtkinter.CTkLabel(ket_frame, text="Kolom")
    #     kol_lframe.pack(pady=10, side=customtkinter.LEFT)
    #     lab_kolom = customtkinter.CTkLabel(kol_lframe, text="0")
    #     lab_kolom.pack(pady=10)
    #
    #     # baris
    #     bar_lframe = customtkinter.CTkLabel(ket_frame, text="Baris")
    #     bar_lframe.pack(pady=10, padx=10, side=customtkinter.LEFT)
    #     lab_baris = customtkinter.CTkLabel(bar_lframe, text="0")
    #     lab_baris.pack(pady=10)
    #
    #     mulai()
    #
    #     # end frame kiri
    #
    # # pengujian
    # def Pengujian(self):
    #
    #     def validation(n):
    #         msg = "Input angka 2 sampai dengan 20"
    #         if (n > 1) and (n <= 20):
    #             return True
    #         else:
    #             customtkinter.messagebox.showerror('input tidak valid', msg)
    #             return False
    #
    #     def uji():
    #         n_fold = int(e_nf.get())
    #         if (validation(n_fold)):
    #             ak.hitung_akurasi(setting.csv_save, n_fold, root, gui, setting)
    #
    #     def cek_stat():
    #         if exists(setting.csv_save):
    #             b_uji.config(state='normal')
    #         else:
    #             b_uji.config(state='disable')
    #             msg_box = customtkinter.messagebox.askquestion('GLCM belum dilakukan', 'Lakukan ekstraksi GLCM?',
    #                                                 icon='warning')
    #             if msg_box == 'yes':
    #                 self.on_click("GLCM")
    #             else:
    #                 customtkinter.messagebox.showinfo('Return',
    #                                        'Untuk pengujian akurasi harus melakukan ekstraksi GLCM terlebih dahulu')
    #
    #         if (len(setting.all) > 0):
    #             hasil = setting.all
    #
    #             # panggil tabel
    #             tabel_akurasi(hasil)
    #
    #             # akurasi
    #             plot(hasil)
    #
    #         e_nf.insert(0, setting.n_fold)
    #
    #     # buat 2 frame lagi
    #     f_kiri = customtkinter.CTkLabel(content_frame, text="Pengujian Data")
    #     f_kiri.pack(side=customtkinter.TOP, anchor="w", pady=(20, 0), padx=10)
    #
    #     f_kanan = customtkinter.CTkLabel(content_frame)
    #     f_kanan.pack(fill=customtkinter.BOTH, expand=True, pady=(20, 0), padx=10)
    #
    #     # untuk isi n fold
    #     f_nf = customtkinter.CTkFrame(f_kiri)
    #     f_nf.pack(fill=customtkinter.BOTH, side=customtkinter.LEFT, pady=10)
    #
    #     l_nf = customtkinter.CTkLabel(f_nf, text="Masukkan Nilai N-Fold:")
    #     l_nf.pack(side=customtkinter.LEFT, padx=(10, 0))
    #
    #     e_nf = customtkinter.CTkEntry(f_nf)
    #     e_nf.pack(side=customtkinter.LEFT, padx=(10, 0))
    #
    #     b_uji = customtkinter.CTkButton(f_nf, text="Pengujian Data", height=2, command=uji)
    #     b_uji.pack(side=customtkinter.LEFT, padx=10)
    #
    #     # isi bawah
    #
    #     # hasil nilai k -> graphic akurasi
    #     f_b1 = customtkinter.CTkFrame(f_kanan)
    #     f_b1.pack(side=customtkinter.TOP, anchor='w', fill=customtkinter.X, expand=True)
    #
    #     # label frame nilai K
    #     lf_table_akurasi = customtkinter.CTkLabel(f_b1, text="Tabel Nilai K")
    #     lf_table_akurasi.pack(side=customtkinter.LEFT, expand=True, fill=customtkinter.BOTH, padx=0)
    #
    #     lf_grap_akurasi = customtkinter.CTkLabel(f_b1, text="Grafik K")
    #     lf_grap_akurasi.pack(side=customtkinter.RIGHT, expand=True, fill=customtkinter.BOTH, padx=(10, 0))
    #
    #     def tabel_akurasi(hasil):
    #         kolom = ('k', 'nilai_ak', 'rata_ak')
    #         tree = ttk.Treeview(lf_table_akurasi, columns=kolom, show='headings', height=20)
    #         tree.heading('k', text='Nilai K')
    #         tree.column('k', width=50)
    #         tree.heading('nilai_ak', text='Nilai Akurasi')
    #         tree.heading('rata_ak', text='Rata-rata')
    #         tree.column('rata_ak', width=80)
    #
    #         values = []
    #
    #         for n in range(len(hasil[1])):
    #             avg = "{:.4f}".format(np.average(hasil[1][n]))
    #             values.append((f'K = {n + 1}', f'{str(hasil[1][n])}', avg))
    #
    #         for value in values:
    #             tree.insert('', customtkinter.END, values=value)
    #
    #         # def item_selected(event):
    #         #     for selected_item in tree.selection():
    #         #         item = tree.item(selected_item)
    #         #         record = item['values']
    #         #         # show a message
    #         #         showinfo(title='Information', message=','.join(record))
    #         #
    #         # tree.bind('<<TreeviewSelect>>', item_selected)
    #         tree.pack(padx=10, pady=10, expand=True)
    #
    #     # grafik
    #     def plot(hasil):
    #
    #         # the figure that will contain the plot
    #         fig = Figure(figsize=(5, 5),
    #                      dpi=100)
    #
    #         xpoints = hasil[0]
    #         ypoints = list()
    #
    #         for i in hasil[1]:
    #             ypoints.append(np.average(i))
    #
    #         # adding the subplot
    #         plot1 = fig.add_subplot(111)
    #
    #         plot1.plot()
    #
    #         plot1.set_xlabel('Nilai K')
    #         plot1.set_ylabel('Nilai Akurasi')
    #
    #         plot1.set_title("Akurasi Berdasarkan Nilai K")
    #
    #         # plotting the graph
    #         plot1.plot(xpoints, ypoints)
    #
    #         # creating the customtkinterinter canvas
    #         # containing the Matplotlib figure
    #         canvas = FigureCanvasTkAgg(fig,
    #                                    master=lf_grap_akurasi)
    #         canvas.draw()
    #
    #         # placing the canvas on the customtkinterinter window
    #         canvas.get_customtkinter_widget().pack()
    #
    #         # creating the Matplotlib toolbar
    #         # toolbar = NavigationToolbar2customtkinter(canvas,
    #         #                                lf_grap_akurasi)
    #         # toolbar.update()
    #
    #         # placing the toolbar on the customtkinterinter window
    #         canvas.get_customtkinter_widget().pack(padx=10, pady=10)
    #
    #     cek_stat()
    #
    # def Klasifikasi(self):
    #
    #     def cek():
    #
    #         # citra_entry.insert(0, setting.img)
    #         k_entry.insert(0, setting.num_neighbors)
    #
    #         # panggil fungsi untuk tabel
    #         table_glcm()
    #         tabel_jarak()
    #
    #     # fungsi untuk memilih gambar
    #     def open_image():
    #         file_path = filedialog.askopenfilename()
    #         if file_path:
    #             image = cv2.imread(file_path)
    #             setting.img = image
    #
    #             # perkecil skala 0.5
    #             width = int(image.shape[1] * 0.5)
    #             height = int(image.shape[0] * 0.5)
    #             dim = (width, height)
    #
    #             resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #             resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    #
    #             img_cr = glcm.cropping(image)
    #             img_gr = glcm.grayscale(img_cr)
    #
    #             image_cr = ImageTk.PhotoImage(Image.fromarray(img_cr))
    #             image_gr = ImageTk.PhotoImage(Image.fromarray(img_gr))
    #             image_customtkinter = ImageTk.PhotoImage(Image.fromarray(resized))
    #
    #             citra_entry.delete(0, 'end')
    #             citra_entry.insert(customtkinter.END, str(file_path))
    #
    #             image_label.config(image=image_customtkinter, width=120, height=120)
    #             image_crop.config(image=image_cr, width=120, height=120)
    #             img_prep.config(image=image_gr, width=120, height=120)
    #
    #             image_crop.image = image_cr
    #             image_label.image = image_customtkinter
    #             img_prep.image = image_gr
    #
    #     def klasifikasi():
    #
    #         setting.num_neighbors = k_entry.get()
    #
    #         csv_save = setting.csv_save
    #         num_neighbors = k_entry.get()
    #
    #         img = glcm.preprocessing(setting.img)
    #
    #         klas.proses_utama(csv_save, int(num_neighbors), img, setting)
    #
    #         # update tabel
    #         table_glcm()
    #         tabel_jarak()
    #         update_hasil()
    #
    #     def update_hasil():
    #         # clear img_hasil
    #         for widget in img_frame.winfo_children():
    #             widget.destroy()
    #
    #         hasil = setting.data_klasifikasi[1]
    #         sub = hasil.replace(" ", "-")
    #         path = os.path.join(setting.folder, sub)
    #         file = os.listdir(path)
    #
    #         index = 0
    #         for img in file:
    #             index += 1
    #
    #             img_path = os.path.join(path, img)
    #
    #             image = cv2.imread(img_path)
    #
    #             # perkecil skala 0.5
    #             width = int(image.shape[1] * 0.5)
    #             height = int(image.shape[0] * 0.5)
    #             dim = (width, height)
    #
    #             resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #             resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    #
    #             image_customtkinter = ImageTk.PhotoImage(Image.fromarray(resized))
    #
    #             img_hasil = customtkinter.CTkLabel(img_frame, width=18, height=8, bg='#fff')
    #             img_hasil.pack(padx=10, side=customtkinter.LEFT)
    #
    #             img_hasil.config(image=image_customtkinter, width=120, height=120)
    #             img_hasil.image = image_customtkinter
    #
    #             if index == 3:
    #                 break
    #
    #         hs_label1.config(text="Hasil Klasifikasi: " + setting.data_klasifikasi[1])
    #
    #     # kiri / atas
    #     f_kiri = customtkinter.CTkLabel(content_frame, text="Klasifikasi Citra Motif Batik")
    #     f_kiri.pack(side=customtkinter.TOP, fill=customtkinter.X, pady=(20, 0), padx=10)
    #
    #     # frame row 1
    #
    #     r1 = customtkinter.CTkFrame(f_kiri)
    #     r1.pack(fill=customtkinter.X)
    #
    #     # frame untuk pack
    #     inp_frame = customtkinter.CTkFrame(r1)
    #     inp_frame.pack(side=customtkinter.LEFT, anchor='n', fill=customtkinter.X, padx=(0, 30), pady=10)
    #
    #     # frame untuk pengaturan folder
    #     fd_frame = customtkinter.CTkFrame(inp_frame, bg="")
    #     fd_frame.pack(fill=customtkinter.BOTH)
    #
    #     citra_label = customtkinter.CTkLabel(fd_frame, text="Citra:")
    #     citra_label.pack(padx=(20, 0), pady=10, side=customtkinter.LEFT)
    #
    #     citra_entry = customtkinter.CTkEntry(fd_frame, width=36)
    #     citra_entry.pack(padx=(20, 10), pady=5, side=customtkinter.LEFT)
    #
    #     label2 = customtkinter.CTkButton(fd_frame, text="Pilih Citra", command=open_image)
    #     label2.pack(pady=10, padx=10, side=customtkinter.LEFT)
    #
    #     # frame untuk bawahnya
    #     k_frame = customtkinter.CTkFrame(inp_frame)
    #     k_frame.pack(fill=customtkinter.BOTH, padx=20)
    #
    #     k_jarak = customtkinter.CTkLabel(k_frame, text="Nilai K:")
    #     k_jarak.pack(side=customtkinter.LEFT)
    #
    #     k_entry = customtkinter.CTkEntry(k_frame, width=50)
    #     k_entry.pack(side=customtkinter.LEFT, anchor='w', padx=(10, 0))
    #
    #     # frame untuk tombol GLCM
    #     t_frame = customtkinter.CTkFrame(inp_frame)
    #     t_frame.pack(fill=customtkinter.BOTH, padx=20)
    #
    #     btn_proc = customtkinter.CTkButton(t_frame, text="Proses Klasifikasi", width=50, height=2, command=klasifikasi)
    #     btn_proc.pack(pady=(30, 40), side=customtkinter.RIGHT, fill=customtkinter.X)
    #
    #     # untuk citra masukkan
    #     kl_frame = customtkinter.CTkFrame(r1)
    #     kl_frame.pack(padx=(30, 110), pady=10, anchor='w', side=customtkinter.LEFT)
    #
    #     image_label = customtkinter.CTkLabel(kl_frame, width=16, height=8, bg='#fff')
    #     image_label.pack()
    #
    #     lb_prep = customtkinter.CTkLabel(kl_frame, text="Citra asli")
    #     lb_prep.pack(pady=(10, 0))
    #
    #     # hasil cropping
    #     cr_frame = customtkinter.CTkFrame(r1)
    #     cr_frame.pack(side=customtkinter.LEFT, anchor='w', padx=20, pady=20)
    #
    #     image_crop = customtkinter.CTkLabel(cr_frame, width=16, height=8, bg='#fff')
    #     image_crop.pack()
    #
    #     lb_crop = customtkinter.CTkLabel(cr_frame, text="Citra cropping")
    #     lb_crop.pack(pady=(10, 0))
    #
    #
    #     # hasil preprocessing
    #     rsz_frame = customtkinter.CTkFrame(r1)
    #     rsz_frame.pack(side=customtkinter.LEFT, anchor='n', pady=20, padx=20)
    #
    #     img_prep = customtkinter.CTkLabel(rsz_frame, width=16, height=8, bg='#fff')
    #     img_prep.pack()
    #
    #     lb_prep = customtkinter.CTkLabel(rsz_frame, text="Citra Grayscale")
    #     lb_prep.pack(pady=(10, 0))
    #
    #
    #     def table_glcm():
    #
    #         for widget in kr1.winfo_children():
    #             widget.destroy()
    #
    #         # CTkButton untuk detail perhitungan
    #         btn_det = customtkinter.CTkButton(kr1, text="Detail Perhitungan")
    #         btn_det.pack(pady=(10, 0), padx=20, side=customtkinter.TOP, anchor='e')
    #
    #         columns = ('ara', 'dis', 'cor', 'hom', 'con', 'asm', 'ene')
    #         trv_glcm = ttk.Treeview(kr1, columns=columns, show='headings', height=4)
    #
    #         trv_glcm.heading('ara', text='Sudut')
    #         trv_glcm.heading('dis', text='Dissimilarity')
    #         trv_glcm.heading('cor', text='Correlation')
    #         trv_glcm.heading('hom', text='Homogeneity')
    #         trv_glcm.heading('con', text='Contrast')
    #         trv_glcm.heading('asm', text='ASM')
    #         trv_glcm.heading('ene', text='Energy')
    #
    #         trv_glcm.column('ara', width=40)
    #         trv_glcm.column('dis', width=160)
    #         trv_glcm.column('cor', width=160)
    #         trv_glcm.column('hom', width=160)
    #         trv_glcm.column('con', width=160)
    #         trv_glcm.column('asm', width=160)
    #         trv_glcm.column('ene', width=160)
    #
    #         data_glcm = []
    #
    #         if (len(setting.data_klasifikasi) > 0):
    #             hasil_glcm = setting.data_klasifikasi[0]
    #
    #             sudut = 0
    #             for i in range(4):
    #                 data_glcm.append((str(sudut), hasil_glcm[i], hasil_glcm[i + 4], hasil_glcm[i + 8],
    #                                   hasil_glcm[i + 12], hasil_glcm[i + 16], hasil_glcm[i + 2]))
    #                 sudut += 45
    #
    #         else:
    #             sudut = 0
    #             for i in range(4):
    #                 data_glcm.append((str(sudut), '0', '0', '0', '0', '0', '0'))
    #                 sudut += 45
    #
    #         for data in data_glcm:
    #             trv_glcm.insert('', customtkinter.END, values=data)
    #
    #         trv_glcm.pack(pady=20)
    #
    #     # grafik
    #     def tabel_jarak():
    #
    #         for widget in hasil_frame2.winfo_children():
    #             widget.destroy()
    #
    #         columns = ('no', 'nama', 'jarak')
    #         trv_glcm = ttk.Treeview(hasil_frame2, columns=columns, show='headings', height=8)
    #
    #         trv_glcm.heading('no', text='No')
    #         trv_glcm.heading('nama', text='Nama Batik')
    #         trv_glcm.heading('jarak', text='Jarak')
    #
    #         trv_glcm.column('no', width=60)
    #
    #         data_glcm = []
    #
    #         if (len(setting.data_jarak) > 0):
    #
    #             setting.data_jarak.sort(key=lambda x: x[1])
    #
    #             for i in range(len(setting.data_jarak)):
    #                 data_glcm.append((i + 1, setting.data_jarak[i][0], setting.data_jarak[i][1]))
    #
    #         else:
    #             data_glcm.append((1, 'mega mendung', 0))
    #             data_glcm.append((2, 'kawung', 0))
    #             data_glcm.append((3, 'parang rusak', 0))
    #             data_glcm.append((4, 'bukan batik', 0))
    #             data_glcm.append((5, 'insang', 0))
    #             data_glcm.append((6, 'dayak', 0))
    #             data_glcm.append((7, 'poleng', 0))
    #             data_glcm.append((8, 'ikat celup', 0))
    #
    #         for data in data_glcm:
    #             trv_glcm.insert('', customtkinter.END, values=data)
    #
    #         trv_glcm.pack(pady=20, padx=20, fill=customtkinter.X)
    #
    #     # frame bawahnya atau kanan
    #     f_kanan = customtkinter.CTkLabel(content_frame)
    #     f_kanan.pack(anchor='w', fill=customtkinter.X, padx=10, pady=(20, 0))
    #
    #     kr1 = customtkinter.CTkLabel(f_kanan, text="Hasil ekstraksi fitur tekstur GLCM")
    #     kr1.pack(expand=True, fill=customtkinter.BOTH)
    #
    #
    #     kr2 = customtkinter.CTkFrame(f_kanan)
    #     kr2.pack(anchor='w', fill=customtkinter.X, pady=(20, 0))
    #
    #     hasil_frame = customtkinter.CTkLabel(kr2, text="Hasil klasifikasi")
    #     hasil_frame.pack(fill=customtkinter.BOTH, side=customtkinter.LEFT, anchor='w', expand=True)
    #
    #     img_frame = customtkinter.CTkFrame(hasil_frame)
    #     img_frame.pack(pady=10, padx=10)
    #
    #     img_hasil = customtkinter.CTkLabel(img_frame, width=18, height=8, bg='#fff')
    #     img_hasil.pack(padx=10, side=customtkinter.LEFT)
    #
    #     img_hasil = customtkinter.CTkLabel(img_frame, width=18, height=8, bg='#fff')
    #     img_hasil.pack(padx=10, side=customtkinter.LEFT)
    #
    #     img_hasil = customtkinter.CTkLabel(img_frame, width=18, height=8, bg='#fff')
    #     img_hasil.pack(padx=10, side=customtkinter.LEFT)
    #
    #     hs_label1 = customtkinter.CTkLabel(hasil_frame, text="Hasil Klasifikasi: -", font=25, bg="#666")
    #     hs_label1.pack(side=customtkinter.TOP, anchor='w', pady=10, padx=10, fill=customtkinter.X)
    #
    #     # tabel jarak
    #
    #     hasil_frame2 = customtkinter.CTkLabel(kr2, text="Tabel Jarak")
    #     hasil_frame2.pack(fill=customtkinter.BOTH, side=customtkinter.LEFT, anchor='w', expand=True, padx=(10, 0))
    #
    #     # panggil fungsi untuk tabel dan plot
    #
    #     cek()

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
        elif (CTkButton_name == "K-Fold Cross Validation"):
            self.KFolds()
        elif (CTkButton_name == "Pengujian Akurasi"):
            self.Pengujian()
        else:
            self.Klasifikasi()

    def create_CTkButton(self, text):
        return customtkinter.CTkButton(sidebar_frame, text=text, width=30, height=2, command=lambda: self.on_click(text))

    def create_sidebar(self):
        CTkButtons = ["Dataset", "Cropping", "Grayscaling", "GLCM", "K-Fold Cross Validation", "Pengujian Akurasi", "Klasifikasi Citra"]
        for CTkButton_name in CTkButtons:
            CTkButton = self.create_CTkButton(CTkButton_name)
            CTkButton.pack(pady=10, padx=20)


# custom customtkinterinter
customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

# Membuat instance customtkinterinter
root = customtkinter.CTk()
root.title("GLCM -> Pseudo Nearest Neighbor")
root.state('zoomed')
root.after(0, lambda:root.state('zoomed'))

# CTkFrame utama untuk konten utama
main_frame = customtkinter.CTkFrame(root, bg_color='red')
main_frame.grid(row=0, column=0, sticky='nsew')
# main_frame.pack(fill=customtkinter.BOTH, side=customtkinter.LEFT, expand=True)

# CTkFrame untuk sidebar
sidebar_frame = customtkinter.CTkFrame(main_frame, width=400)
sidebar_frame.grid(column=0, row=0)

# CTkFrame untuk konten utama
content_frame = customtkinter.CTkFrame(main_frame, width=2000)
content_frame.grid(column=1, row=0, sticky='e')

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
