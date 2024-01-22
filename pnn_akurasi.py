import threading
import time
from random import randrange
from csv import reader
from math import sqrt
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, ttk
from tkinter.messagebox import showinfo
import matplotlib.pyplot as plt


class PnnAk:
    data_jarak = list()

    # Load a CSV file
    def load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset


    # Convert string column to float
    def str_column_to_float(self, dataset, column):
        for row in dataset[1:]:
            row[column] = float(row[column].strip())


    # Convert string column to integer
    def str_column_to_int(self, dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        ret = list()
        for i, value in enumerate(unique):
            lookup[value] = i
            # print('[%s] => %d' % (value, i))
        for row in dataset:
            row[column] = lookup[row[column]]
        ret.append([lookup, i+1])
        return ret


    # Find the min and max values for each column
    def dataset_minmax(self, dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax


    # Rescale dataset columns to the range 0-1
    def normalize_dataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


    # Split a dataset into k folds
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)

        return dataset_split


    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, pros, pb, i, idx, algorithm, n_folds, *args):
        folds = self.cross_validation_split(dataset, n_folds)

        jml_proses = 0
        for fold in folds:
            jml_proses += len(fold)

        scores = list()
        idx = 0
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None

                n = (idx/jml_proses) * 100
                pros.configure(text="Progress : " + str(round(n, 2)) + "%")
                pb.set(n/100)
                idx += 1

            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            accuracy = "{:.4f}".format(accuracy)
            scores.append(float(accuracy))

        return scores


    # Calculate the Euclidean distance between two vectors
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (float(row1[i]) - float(row2[i])) ** 2
        return sqrt(distance)


    # Locate the most similar neighbors
    def get_neighbors(self, train, test_row, num_neighbors, num_class):
        distances = list()
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)

            distances.append((train_row, dist))

        distances.sort(key=lambda tup: tup[1])

        neighbors = list()
        for c in range(num_class):
            a1 = 0
            total_jarak = 0.0
            for x in range(len(distances)):
                if (distances[x][0][24] == c) and (a1 < num_neighbors):
                    total_jarak += distances[x][1]
                    # neighbors.append([distances[x][0][24], distances[x][1]])
                    a1 += 1
            neighbors.append(total_jarak / num_neighbors)

        return neighbors


    # Make a prediction with neighbors
    def predict_classification(self, train, test_row, num_neighbors, num_class):
        neighbors = self.get_neighbors(train, test_row, num_neighbors, num_class)

        # dapatkan index dari nilai terkecil
        tmp = min(neighbors)
        index = neighbors.index(tmp)

        return index

    # PNN algorithm
    def pnn(self, train, test, num_neighbors, num_class):
        predictions = list()

        for row in test:
            output = self.predict_classification(train, row, num_neighbors, num_class)
            predictions.append(output)

        return (predictions)


    def hitung_akurasi(self, filename, n_folds, rot, gui, setting):

        dataset = self.load_csv(filename)
        dataset.pop(0)

        for i in range(len(dataset[0]) - 1):
            self.str_column_to_float(dataset, i)

        # convert class column to integers and get count of class
        num_class = self.str_column_to_int(dataset, len(dataset[0]) - 1)[0][1]

        # scores = evaluate_algorithm(dataset, pnn, n_folds, num_neighbor, num_class)
        #
        # print(f'K: {num_neighbor} => Scores: %s' % scores)

        nilai_k = list()
        nilai_ak = list()
        all = list()

        # root window
        root2 = ctk.CTkToplevel(rot)
        root2.title('Silahkan tunggu')

        width = 410
        height = 160

        screen_width = root2.winfo_screenwidth()  # Width of the screen
        screen_height = root2.winfo_screenheight()  # Height of the screen

        # Calculate Starting X and Y coordinates for Window
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)

        root2.geometry('%dx%d+%d+%d' % (width, height, x, y-200))

        root2.grab_set()


        # label
        value_label = ctk.CTkLabel(root2, text="Menghitung akurasi PNN...")
        value_label.pack(pady=(20, 0))

        # progressbar
        pb = ctk.CTkProgressBar(
            root2,
            orientation='horizontal',
            mode='determinate',
            width=390,
            height=16
        )
        # place the progressbar
        pb.pack(padx=10, pady=20, fill=tk.X)

        fr2 = ctk.CTkFrame(root2, bg_color='transparent', fg_color='transparent')
        fr2.pack(fill=ctk.X, padx=10)

        value_step = ctk.CTkLabel(fr2, text="K = 1 / 20")
        value_step.pack(side=ctk.LEFT)

        # label proses
        pros = ctk.CTkLabel(fr2, text="Progress : ")
        pros.pack(side=ctk.RIGHT)

        def close_pbar():
            root2.destroy()
            setting.n_fold = n_folds
            gui.on_click("Pengujian Akurasi")


        def selesai():
            for widget in root2.winfo_children():
                widget.destroy()

            #     untuk menambahkan tombol
            lab_sel = ctk.CTkLabel(root2, text="Proses Pengujian Akurasi Selesai")
            lab_sel.pack(fill=tk.X, pady=(20, 0))

            btn_frame = ctk.CTkFrame(root2)
            btn_frame.pack(pady=(20, 0), side=ctk.TOP)

            btn_sel = ctk.CTkButton(btn_frame, text="OK", command=close_pbar)
            btn_sel.pack()


        idx = 0
        def proses():
            for i in range(20):
                n = (i/20) * 100
                pros.configure(text="Progress : " + str(round(n, 2)) + "%")
                pb.set(i/20)
                value_step.configure(text=f"K = {i+1} / 20")
                scores = self.evaluate_algorithm(dataset, pros, pb, i, idx, self.pnn, n_folds, i+1, num_class)
                print(f'K: {i+1} => Scores: %s' % scores)
                nilai_k.append(i)
                nilai_ak.append(scores)

            all.append(nilai_k)
            all.append(nilai_ak)

            setting.all = all

            selesai()

        def progress():
            loading_thread = threading.Thread(target=proses)
            loading_thread.start()


        root2.after(100, progress)
        root2.mainloop()



































            # print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

        # xpoints = nilai_k
        # ypoints = nilai_ak
        # plt.xlabel('Nilai K')
        # plt.ylabel('Nilai Akurasi')
        #
        # plt.title("Akurasi Berdasarkan Nilai K")
        #
        #
        # plt.plot(xpoints, ypoints)
        # plt.show()