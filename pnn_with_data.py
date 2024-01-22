import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from csv import reader
from math import sqrt


class PnnData:

    def __init__(self):
        # define global variable
        self.data_jarak = list()
        self.class2 = list()

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
        for row in dataset:
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
            self.class2.append(value)
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


    # Calculate the Euclidean distance between two vectors
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)


    # Locate the most similar neighbors
    def get_neighbors(self, train, test_row, num_neighbors, num_class):
        distances = list()
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)

            distances.append((train_row, dist))

        distances.sort(key=lambda tup: tup[1])

        neighbors = list()

        # kosongkan data jarak
        self.data_jarak.clear()

        for c in range(num_class):
            a1 = 0
            total_jarak = 0.0
            for x in range(len(distances)):
                if (distances[x][0][24] == c) and (a1 < num_neighbors):
                    total_jarak += distances[x][1]
                    # neighbors.append([distances[x][0][24], distances[x][1]])
                    a1 += 1
            neighbors.append(total_jarak/num_neighbors)
            self.data_jarak.append([self.generate_hasil(c), (total_jarak/num_neighbors)])


        return neighbors

    # generate hasil
    def generate_hasil(self, x):
        hasil = self.class2[x].replace("-", " ")

        return hasil


    # Make a prediction with neighbors
    def predict_classification(self, train, test_row, num_neighbors, num_class):
        neighbors = self.get_neighbors(train, test_row, num_neighbors, num_class)

        # dapatkan index dari nilai terkecil
        tmp = min(neighbors)
        index = neighbors.index(tmp)

        print(f"neighbors: {tmp}")

        # jika lebih dari 1000 maka dianggap bukan batik
        if(tmp > 2000):
            hasil_akhir = "bukan-batik"
        else:
            hasil_akhir = self.generate_hasil(index)

        return hasil_akhir

    # calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135
    def calc_glcm_all_agls(self, setting, img, label, dists, props, agls=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], lvl=256,
                           sym=True,
                           norm=True):
        glcm_awal = graycomatrix(img, distances=[dists], angles=agls, levels=lvl)
        glcm_symm = graycomatrix(img, distances=[dists], angles=agls, levels=lvl, symmetric=True)
        glcm = graycomatrix(img, distances=[dists], angles=agls, levels=lvl, symmetric=sym, normed=norm)
        feature = []
        glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
        for item in glcm_props:
            feature.append(item)

        if label != 0:
            feature.append(label)

        setting.glcm = glcm
        setting.glcm_awal = glcm_awal
        setting.glcm_symm = glcm_symm

        print(f"feature: {feature}")

        return feature

    def proses_utama(self, csv_save, num_neighbors, img, setting):

        self.class2 = list()
        # main function
        properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

        # PNN classsification
        # Make a prediction using csv
        dataset = self.load_csv(csv_save)
        dataset.pop(0)

        for i in range(len(dataset[0]) - 1):
            self.str_column_to_float(dataset, i)

        # convert class column to integers and get count of class
        num_class = self.str_column_to_int(dataset, len(dataset[0]) - 1)[0][1]

        row = self.calc_glcm_all_agls(setting, img, 0, setting.jarak, props=properties)

        # predict the label
        label = self.predict_classification(dataset, row, num_neighbors, num_class)
        # print('Data = %s, \nPredicted: %s' % (row, label))

        #append data klasifikasi, row = perhitungan glcm, label = hasil klasifikasi
        setting.data_klasifikasi.clear()
        setting.data_klasifikasi.append(row)
        setting.data_klasifikasi.append(label)

        # data jarak
        # setting.data_jarak.clear()
        setting.data_jarak = self.data_jarak








