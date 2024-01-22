import time

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import threading
import tkinter as tk
import cv2
from tkinter import filedialog, ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import os
from os.path import exists

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# class setting
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

    def default(self):
        self.folder = "dataset/"
        self.csv_save = "csv/ekstraksi_fitur.csv"
        self.jarak = 1
        self.n_fold = 5
        self.dataset_stat = False
        self.glcm_stat = False
        self.Pengujian_stat = False
        self.Klasifikasi_stat = False

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

    def preprocessing(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        ymin, ymax, xmin, xmax = h // 2 - 50, h // 2 + 50, w // 2 - 50, w // 2 + 50
        crop = gray[ymin:ymax, xmin:xmax]

        resize = cv2.resize(crop, (100, 100))

        return resize

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

    def load_dataset(self, dataset_dir, csv_save, dists, root):
        # -------------------- Load Dataset ------------------------

        imgs = []  # list image matrix
        labels = []
        descs = []

        for folder in os.listdir(dataset_dir):
            for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
                img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder))

                imgs.append(self.preprocessing(img))
                labelfix = (str(folder)).replace(" ", "-")
                labels.append(labelfix)
                descs.append(self.normalize_desc(folder, sub_folder))

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

    def glcm_avg(self, dataset_dir, csv_dir):

        if not (os.path.exists(csv_dir)):
            self.glcm(dataset_dir, csv_dir)
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

    def glcm(self, dataset_dir, csv_dir, root):
        dataset_dir = dataset_dir
        csv_save = csv_dir
        self.load_dataset(dataset_dir, csv_save, root)

class GUI:

    # fungsi hapus widget di frame

    def clear_frame(self):
        for widgets in content_frame.winfo_children():
            widgets.destroy()

    # dataset
    def Dataset(self):

        def clear_canvas():
            for widgets in frame_container.winfo_children():
                widgets.destroy()

        def set_data():
            if (setting.get_stat('1') == False):
                dt_entry.config(state='normal')
                dt_entry.delete(0, tk.END)
                dt_entry.insert(0, setting.folder)
                dt_entry.config(state='disable')
                show_data(setting.get_folder())

        def pilih_folder():
            folder = filedialog.askdirectory()
            dt_entry.config(state='normal')
            dt_entry.delete(0, tk.END)
            dt_entry.insert(0, folder)
            dt_entry.config(state='disable')
            setting.folder = folder
            show_data(folder)

        def show_data(folder):
            clear_canvas()
            folder1 = os.listdir(folder)

            for sub in folder1:
                folder2 = os.path.join(folder, sub)

                image_count = 0
                columns = 19

                img_row = tk.LabelFrame(frame_container, text=sub)
                img_row.pack()

                file = os.listdir(folder2)

                for name in file:
                    image_count += 1
                    r, c = divmod(image_count - 1, columns)
                    im = Image.open(os.path.join(folder2, name))
                    resized = im.resize((50, 50), Image.Resampling.LANCZOS)
                    tkimage = ImageTk.PhotoImage(resized)
                    myvar = tk.Label(img_row, image=tkimage)
                    myvar.image = tkimage
                    myvar.grid(row=r, column=c)

        def on_frame_configure(canvas):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_mousewheel(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        # buat 2 frame lagi
        f_kiri = tk.LabelFrame(content_frame, text="Pengaturan Dataset")
        f_kiri.pack(side=tk.TOP, anchor='w', pady=(20, 0), padx=10)

        kr1 = tk.Label(f_kiri)
        kr1.pack(side=tk.LEFT, anchor='n', padx=10, pady=10)

        dt_label = tk.Label(kr1, text="Folder:")
        dt_label.pack(side=tk.TOP, anchor='w')

        dt_entry = tk.Entry(kr1, width=40)
        dt_entry.pack(side=tk.LEFT, anchor='w')
        dt_entry.config(state='disable')

        dt_btn = tk.Button(kr1, text="Pilih Folder", command=pilih_folder)
        dt_btn.pack(side=tk.LEFT, anchor='w', padx=20)

        # bawah
        f_kanan = tk.Frame(content_frame)
        f_kanan.pack(fill=tk.BOTH, expand=True, pady=(20, 0), padx=10)

        canvas = tk.Canvas(f_kanan)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(f_kanan, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)

        frame_container = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame_container, anchor="nw")

        frame_container.bind("<Configure>", lambda event, canvas=canvas: on_frame_configure(canvas))

        set_data()


    def GLCM(self):

        def mulai():
            if(setting.glcm_stat == False):
                pass
                # for widgets in f_kanan.winfo_children():
                #     widgets.destroy()
            else:
                upload_file()

            folder_entry.config(state='normal')
            folder_entry.insert(0, setting.folder)
            folder_entry.config(state='readonly')

        # fungsi get csv
        def upload_file():

            df = glcm.glcm_avg(setting.folder, setting.csv_save)
            l1 = list(df)  # List of column names as header
            lab_kolom.config(text=str(df.shape[1]))
            lab_baris.config(text=str(df.shape[0]))
            trv_refresh(df, l1)  # show Treeview

        # treeview untuk menampilkan df
        def trv_refresh(df, l1):  # Refresh the Treeview to reflect changes
            r_set = df.to_numpy().tolist()  # create list of list using rows
            trv = ttk.Treeview(f_kanan, selectmode='browse', height=20,
                               show='headings', columns=l1)
            trv.pack(pady=(0, 10), padx=10, fill=tk.BOTH, expand=True, side=tk.LEFT, anchor='w')

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

            glcm.load_dataset(folder, csv, jarak, root)

            # if glcm.load_dataset(folder, csv, jarak, root):
            #     setting.glcm_stat = True
            #     upload_file()
            # else:
            #     print("Terjadi kesalahan")



        # buat 2 frame lagi
        f_kiri = tk.LabelFrame(content_frame, text="Ekstraksi Fitur Tekstur GLCM")
        f_kiri.pack(fill=tk.X, pady=(20, 0), padx=10)

        f_kanan = tk.LabelFrame(content_frame, text="Hasil Ekstraksi Fitur Tekstur GLCM")
        f_kanan.pack(fill=tk.BOTH, expand=True, pady=(20, 0), padx=10)

        # buat 2 frame untuk tombol tertentu
        r1_kiri = tk.Frame(f_kiri)
        r1_kiri.pack(side=tk.LEFT, anchor='w', pady=(0, 20))

        r1_kanan = tk.Frame(f_kiri)
        r1_kanan.pack(side=tk.LEFT, anchor='w')

        # di dalam frame kiri
        # frame untuk pengaturan folder
        fd_frame = tk.Frame(r1_kiri)
        fd_frame.pack(fill=tk.BOTH, pady=(10, 0))

        folder_label = tk.Label(fd_frame, text="Folder:")
        folder_label.pack(padx=20, pady=10, side=tk.LEFT)
        folder_entry = tk.Entry(fd_frame, width=50)
        folder_entry.pack(padx=2, pady=5, side=tk.LEFT)
        folder_entry.config(readonlybackground='#fff', state='readonly')

        # folder = "datasetBaru/train"
        # label2 = tk.Button(fd_frame, text="Pilih Folder", command=select_folder)
        # label2.pack(pady=10, padx=10, side=tk.LEFT)

        # frame untuk bawahnya
        j_frame = tk.Frame(r1_kiri)
        j_frame.pack(fill=tk.BOTH, padx=20)

        l_jarak = tk.Label(j_frame, text="Jarak:")
        l_jarak.pack(side=tk.LEFT)

        n = tk.StringVar()
        j_entry = ttk.Combobox(j_frame, width=20, textvariable=n, state='readonly')
        j_entry['values'] = ('1', '2', '3', '4', '5')
        j_entry.pack(side=tk.LEFT, padx=(28, 0))
        j_entry.current(1)

        # j_lab = tk.Label(j_frame, text="(range 1-5)")
        # j_lab.pack(side=tk.LEFT, padx=10)

        # frame untuk tombol GLCM
        t_frame = tk.Frame(r1_kanan)
        t_frame.pack(fill=tk.BOTH)

        btn_proc = tk.Button(t_frame, text="Hitung GLCM", width=12, height=3, command=hitung_glcm)
        btn_proc.pack(pady=15, padx=10, side=tk.LEFT)

        lab_hasil = tk.Label(f_kanan, text="Hasil Ekstraksi Fitur GLCM", font=("Arial", 16))
        lab_hasil.pack(pady=(10, 0))

        # buat frame untuk keterangan
        ket_frame = tk.Frame(f_kanan)
        ket_frame.pack(pady=(0, 10), padx=10, fill=tk.BOTH)

        # kolom
        kol_lframe = tk.LabelFrame(ket_frame, text="Kolom")
        kol_lframe.pack(pady=10, side=tk.LEFT)
        lab_kolom = tk.Label(kol_lframe, text="0")
        lab_kolom.pack(pady=10)

        # baris
        bar_lframe = tk.LabelFrame(ket_frame, text="Baris")
        bar_lframe.pack(pady=10, padx=10, side=tk.LEFT)
        lab_baris = tk.Label(bar_lframe, text="0")
        lab_baris.pack(pady=10)

        # membuat tabel untuk menampilkan csv

        # file1 = "csv/ekstraksi_fitur.csv"
        # df = pd.read_csv(file1)
        # str1 = "Baris:" + str(df.shape[0]) + " , Kolom:" + str(df.shape[1])
        # lab_hasil.config(text=str1)
        # l1 = []

        mulai()

        # end frame kiri

    # pengujian
    def Pengujian(self):

        # cek csv file
        if (exists("csv/ekstraksi_fitur.csv")):
            print("Ekstraksi fitur tekstur sudah dilakukan")
        else:
            print("Ekstraksi fitur tekstur belum dilakukan")

        def tabel_akurasi():
            kolom = ('k', 'nilai_ak', 'rata_ak')
            tree = ttk.Treeview(lf_table_akurasi, columns=kolom, show='headings')
            tree.heading('k', text='Nilai K')
            tree.column('k', width=50)
            tree.heading('nilai_ak', text='Nilai Akurasi')
            tree.heading('rata_ak', text='Rata-rata Akurasi')

            skors = []

            for n in range(1, 20):
                skors.append((f'K = {n}', f'last {n}', f'email{n}@example.com'))

            for skor in skors:
                tree.insert('', tk.END, values=skor)

            def item_selected(event):
                for selected_item in tree.selection():
                    item = tree.item(selected_item)
                    record = item['values']
                    # show a message
                    showinfo(title='Information', message=','.join(record))

            tree.bind('<<TreeviewSelect>>', item_selected)
            tree.pack(padx=10, pady=10)

        # buat 2 frame lagi
        f_kiri = tk.LabelFrame(content_frame, text="Pengujian Data")
        f_kiri.pack(side=tk.TOP, anchor="w", pady=(20, 0), padx=10)

        f_kanan = tk.Label(content_frame)
        f_kanan.pack(fill=tk.BOTH, expand=True, pady=(20, 0), padx=10)

        # untuk isi n fold
        f_nf = tk.Frame(f_kiri)
        f_nf.pack(fill=tk.BOTH, side=tk.LEFT, pady=10)

        l_nf = tk.Label(f_nf, text="Masukkan Nilai N-Fold:")
        l_nf.pack(side=tk.LEFT, padx=(10, 0))

        e_nf = tk.Entry(f_nf)
        e_nf.pack(side=tk.LEFT, padx=(10, 0))

        b_uji = tk.Button(f_nf, text="Pengujian Data", height=2)
        b_uji.pack(side=tk.LEFT, padx=10)

        # isi bawah

        # hasil nilai k -> graphic akurasi
        f_b1 = tk.Frame(f_kanan)
        f_b1.pack(side=tk.TOP, anchor='w')

        # label frame nilai K
        lf_table_akurasi = tk.LabelFrame(f_b1, text="Tabel Nilai K")
        lf_table_akurasi.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=0)

        # panggil tabel
        tabel_akurasi()

        # grafik
        def plot():

            # the figure that will contain the plot
            fig = Figure(figsize=(5, 2),
                         dpi=100)

            # list of squares
            y = [i ** 2 for i in range(101)]

            # adding the subplot
            plot1 = fig.add_subplot(111)

            # plotting the graph
            plot1.plot(y)

            # creating the Tkinter canvas
            # containing the Matplotlib figure
            canvas = FigureCanvasTkAgg(fig,
                                       master=lf_grap_akurasi)
            canvas.draw()

            # placing the canvas on the Tkinter window
            canvas.get_tk_widget().pack()

            # creating the Matplotlib toolbar
            # toolbar = NavigationToolbar2Tk(canvas,
            #                                lf_grap_akurasi)
            # toolbar.update()

            # placing the toolbar on the Tkinter window
            canvas.get_tk_widget().pack(padx=10, pady=10)

        lf_grap_akurasi = tk.LabelFrame(f_b1, text="Grafik K")
        lf_grap_akurasi.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(10, 0))

        # akurasi
        plot()

    def Klasifikasi(self):
        # fungsi untuk memilih gambar
        def open_image():
            file_path = filedialog.askopenfilename()
            if file_path:
                image = cv2.imread(file_path)

                # perkecil skala 0.5
                width = int(image.shape[1] * 0.5)
                height = int(image.shape[0] * 0.5)
                dim = (width, height)

                resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

                kikiki = Glcm()
                img_gr = kikiki.preprocessing(image)

                image_gr = ImageTk.PhotoImage(Image.fromarray(img_gr))
                image_tk = ImageTk.PhotoImage(Image.fromarray(resized))

                citra_entry.delete(0, 'end')
                citra_entry.insert(tk.END, str(file_path))
                image_label.config(image=image_tk, width=120, height=120)
                img_prep.config(image=image_gr, width=120, height=120)
                image_label.image = image_tk
                img_prep.image = image_gr

        # kiri / atas
        f_kiri = tk.LabelFrame(content_frame, text="Klasifikasi Citra Motif Batik")
        f_kiri.pack(side=tk.TOP, fill=tk.X, pady=(20, 0), padx=10)

        # frame row 1

        r1 = tk.Frame(f_kiri)
        r1.pack(fill=tk.X)

        # frame untuk pack
        inp_frame = tk.Frame(r1)
        inp_frame.pack(side=tk.LEFT, anchor='n', fill=tk.X, padx=(0, 30), pady=10)

        # frame untuk pengaturan folder
        fd_frame = tk.Frame(inp_frame, bg="")
        fd_frame.pack(fill=tk.BOTH)

        citra_label = tk.Label(fd_frame, text="Citra:")
        citra_label.pack(padx=(20, 0), pady=10, side=tk.LEFT)

        citra_entry = tk.Entry(fd_frame, width=36)
        citra_entry.pack(padx=(20, 10), pady=5, side=tk.LEFT)

        label2 = tk.Button(fd_frame, text="Pilih Citra", command=open_image)
        label2.pack(pady=10, padx=10, side=tk.LEFT)

        # frame untuk bawahnya
        k_frame = tk.Frame(inp_frame)
        k_frame.pack(fill=tk.BOTH, padx=20)

        k_jarak = tk.Label(k_frame, text="Nilai K:")
        k_jarak.pack(side=tk.LEFT)

        k_entry = tk.Entry(k_frame, width=50)
        k_entry.pack(side=tk.LEFT, anchor='w', padx=(10, 0))

        # frame untuk tombol GLCM
        t_frame = tk.Frame(inp_frame)
        t_frame.pack(fill=tk.BOTH, padx=20)

        btn_proc = tk.Button(t_frame, text="Proses Klasifikasi")
        btn_proc.pack(pady=(15, 50), side=tk.LEFT, fill=tk.X)

        # untuk citra masukkan
        kl_frame = tk.Frame(r1)
        kl_frame.pack(padx=(320, 10), pady=10, anchor='w', side=tk.LEFT)

        image_label = tk.Label(kl_frame, width=16, height=8, bg='#fff')
        image_label.pack()

        lb_prep = tk.Label(kl_frame, text="Citra asli")
        lb_prep.pack(pady=(10, 0))

        # hasil preprocessing
        rsz_frame = tk.Frame(r1)
        rsz_frame.pack(side=tk.RIGHT, anchor='n', pady=10, padx=(0, 30))

        img_prep = tk.Label(rsz_frame, width=16, height=8, bg='#fff')
        img_prep.pack()

        lb_prep = tk.Label(rsz_frame, text="Hasil Preprocessing")
        lb_prep.pack(pady=(10, 0))

        def table_glcm():
            columns = ('ara', 'dis', 'cor', 'hom', 'con', 'asm', 'ene')
            trv_glcm = ttk.Treeview(kr1, columns=columns, show='headings', height=4)

            trv_glcm.heading('ara', text='Sudut')
            trv_glcm.heading('dis', text='Dissimilarity')
            trv_glcm.heading('cor', text='Correlation')
            trv_glcm.heading('hom', text='Homogeneity')
            trv_glcm.heading('con', text='Contrast')
            trv_glcm.heading('asm', text='ASM')
            trv_glcm.heading('ene', text='Energy')

            trv_glcm.column('ara', width=40)
            trv_glcm.column('dis', width=160)
            trv_glcm.column('cor', width=160)
            trv_glcm.column('hom', width=160)
            trv_glcm.column('con', width=160)
            trv_glcm.column('asm', width=160)
            trv_glcm.column('ene', width=160)

            data_glcm = []
            data_glcm.append(('0', 1, 2, 3, 4, 5, 6))
            data_glcm.append(('45', 1, 2, 3, 4, 5, 6))
            data_glcm.append(('90', 1, 2, 3, 4, 5, 6))
            data_glcm.append(('135', 1, 2, 3, 4, 5, 6))

            for data in data_glcm:
                trv_glcm.insert('', tk.END, values=data)

            trv_glcm.pack(pady=20)

        # grafik
        def tabel_jarak():

            columns = ('no', 'nama', 'jarak')
            trv_glcm = ttk.Treeview(hasil_frame2, columns=columns, show='headings', height=8)

            trv_glcm.heading('no', text='No')
            trv_glcm.heading('nama', text='Nama Batik')
            trv_glcm.heading('jarak', text='Jarak')

            trv_glcm.column('no', width=60)

            data_glcm = []
            data_glcm.append((1, 'mega mendung', 1494.5963388344826))
            data_glcm.append((2, 'kawung', 2285.6628520018307))
            data_glcm.append((3, 'parang rusak', 2979.1464280963555))
            data_glcm.append((4, 'bukan batik', 6910.107110408299))
            data_glcm.append((5, 'insang', 5632.092564730484))
            data_glcm.append((6, 'dayak', 1550.2962266189309))
            data_glcm.append((7, 'poleng', 3173.732295914358))
            data_glcm.append((8, 'ikat celup', 5181.175126537936))

            for data in data_glcm:
                trv_glcm.insert('', tk.END, values=data)

            trv_glcm.pack(pady=10, padx=10, fill=tk.X)

        # frame bawahnya atau kanan
        f_kanan = tk.Label(content_frame)
        f_kanan.pack(anchor='w', fill=tk.X, padx=10, pady=(20, 0))

        kr1 = tk.LabelFrame(f_kanan, text="Hasil ekstraksi fitur tekstur GLCM")
        kr1.pack(expand=True, fill=tk.BOTH)

        kr2 = tk.Frame(f_kanan)
        kr2.pack(anchor='w', fill=tk.X, pady=(20, 0))

        hasil_frame = tk.LabelFrame(kr2, text="Hasil klasifikasi")
        hasil_frame.pack(fill=tk.BOTH, side=tk.LEFT, anchor='w', expand=True)

        img_hasil = tk.Label(hasil_frame, width=18, height=8, bg='#fff')
        img_hasil.pack(pady=10, padx=10)

        hs_label1 = tk.Label(hasil_frame, text="Hasil Klasifikasi: Batik Mega Mendung", font=(25), bg="#666")
        hs_label1.pack(side=tk.TOP, anchor='w', pady=10, padx=10)

        # tabel jarak

        hasil_frame2 = tk.LabelFrame(kr2, text="Tabel Jarak")
        hasil_frame2.pack(fill=tk.BOTH, side=tk.LEFT, anchor='w', expand=True, padx=(10, 0))

        table_glcm()
        tabel_jarak()

    def on_click(self, button_name):
        self.clear_frame()
        if (button_name == "Dataset"):
            self.Dataset()
        elif (button_name == "GLCM"):
            self.GLCM()
        elif (button_name == "Pengujian Akurasi"):
            self.Pengujian()
        else:
            self.Klasifikasi()

    def create_button(self, text):
        return tk.Button(sidebar_frame, text=text, width=30, height=2, command=lambda: self.on_click(text))

    def create_sidebar(self):
        buttons = ["Dataset", "GLCM", "Pengujian Akurasi", "Klasifikasi Citra"]
        for button_name in buttons:
            button = self.create_button(button_name)
            button.pack(pady=10, padx=20)


# Membuat instance Tkinter
root = tk.Tk()
root.title("GLCM -> Pseudo Nearest Neighbor")
root.state('zoomed')

# Frame utama untuk konten utama
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

# Frame untuk sidebar
sidebar_frame = tk.Frame(main_frame, bg="gray", width=400)
sidebar_frame.pack(side=tk.LEFT, fill=tk.BOTH)

# Frame untuk konten utama
content_frame = tk.Frame(main_frame)
content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20)

setting = Setting()
gui = GUI()
glcm = Glcm()

# Membuat sidebar dengan tombol-tombolnya
gui.create_sidebar()

# start
gui.on_click("Dataset")

# Menjalankan loop Tkinter
root.mainloop()

