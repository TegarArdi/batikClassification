import numpy as np
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import sys

np.set_printoptions(threshold=sys.maxsize, linewidth=1000)

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.10f' % x)



class Gui:

    def __init__(self):
        self.glcm = []
        self.glcm_symm = []
        self.glcm_norm = []
        self.glcm_props = []

    def preprocessing(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        h, w = gray.shape

        if h > w:
            ymin, ymax, xmin, xmax = 0, w, 0, w
        else:
            ymin, ymax, xmin, xmax = 0, h, 0, h

        crop = gray[ymin:ymax, xmin:xmax]

        resize = cv2.resize(crop, (100, 100))

        return resize

    def symmetric_matriks(self, a):
        symm = np.zeros((256, 256), dtype=float)
        symm = a + a.T - np.diag(a.diagonal())

        return symm

    # transpose matriks
    def transpose(self, mat, tr, N):
        for i in range(N):
            for j in range(N):
                tr[i][j] = mat[j][i]

    # cek simetrik
    def isSymmetric(self, mat, N):
        tr = [[0 for j in range(len(mat[0]))] for i in range(len(mat))]
        self.transpose(mat, tr, N)
        for i in range(N):
            for j in range(N):
                if (mat[i][j] != tr[i][j]):
                    return False
        return True

    def normalisasi_matriks(self, matriks):

        if (self.isSymmetric(matriks, 4)):
            print("yes")
        else:
            print("No")

        glcm_norm = np.zeros((4, 4), dtype=float)

        # cari nilai matriks
        sum_matriks = sum(map(sum, matriks))
        print(f"sum: {sum_matriks}")

        for i in range(len(matriks[0])):
            for j in range(len(matriks[i])):
                glcm_norm[i][j] = round((1 / sum_matriks) * matriks[i][j], 8)

        return glcm_norm

    # dissimilarity
    def diss(self, matriks):
        # diss = np.zeros((256, 256), dtype=np.float32)
        diss = 0
        for i in range(256):
            for j in range(256):
                diss += matriks[i, j] * np.abs(i - j)
                # print(f"{matriks[i, j]} * {np.abs(i - j)}")
        return diss

    # homogeneity
    def homogeneity(self, matriks):
        homo = np.zeros((255, 255), dtype=np.float32)
        for i in range(len(matriks[0])):
            for j in range(len(matriks[i])):
                homo += matriks[i, j] / (1. + (i - j) ** 2)

        return homo

    def calc_prop(self):
        pass

    def vectorized_glcm(self, image, distance, direction):
        img_crop = self.cropping(image)
        img_gr = self.grayscale(img_crop)
        img = np.array(img_gr)

        # print(f"gambar: {img}")

        glcm = np.zeros((4, 4), dtype=int)

        if direction == 1:  # 0
            first = img[:, :-distance]
            second = img[:, distance:]
        elif direction == 2:  # 45
            first = img[distance:, :-distance]
            second = img[:-distance, distance:]
        elif direction == 3:  # 90
            first = img[distance:, :]
            second = img[:-distance, :]
        elif direction == 4:  # 135
            first = img[:-distance, :-distance]
            second = img[distance:, distance:]

        # print(f"first: {first}")

        for i, j in zip(first.ravel(), second.ravel()):
            glcm[i, j] += 1

        return glcm

    def glcm_all_agls(self, image, distance):
        glcm_all = []
        for i in range(1, 5):
            glcm_all.append(self.vectorized_glcm(image, distance, i))

        return glcm_all

    def all_props(self, norm_matriks):
        diss = []
        corr = []
        hom = []
        cont = []
        asm = []
        energy = []

        feature = []
        for i in norm_matriks:
            diss.append(self.dissimilarity_feature(i))
            corr.append(self.correlation_feature(i))
            hom.append(self.homogeneity_feature(i))
            cont.append(self.contrast_feature(i))
            asm.append(self.asm_feature(i))
            energy.append(self.energy_feature(i))

        feature = diss + corr + hom + cont + asm + energy

        # props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
        #
        # feature = []
        # glcm_props = [propery for name in props for propery in graycoprops(norm_matriks, name)[0]]
        #
        # for item in glcm_props:
        #     feature.append(item)

        return feature


    def main(self):
        # img = cv2.imread("dataset/batik-dayak/1.jpg")
        # img = self.preprocessing(img)

        img = np.array([[2, 0, 1, 1],
                          [0, 3, 1, 1],
                          [2, 2, 1, 2],
                          [2, 0, 3, 3]], dtype=np.uint8)

        result = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=4, symmetric=True, normed=True)
        print(f"HASIL: \n{result[:, :, 0, 0]}")

        print(f"contrast: {graycoprops(result, 'contrast')}")
        print(f"correlation: {graycoprops(result, 'correlation')}")
        print(f"ASM: {graycoprops(result, 'ASM')}")
        print(f"Energy: {graycoprops(result, 'energy')}")
        print(f"Homogeneity: {graycoprops(result, 'homogeneity')}")
        print(f"Dissimilarity: {graycoprops(result, 'dissimilarity')}")


        # glcm_matriks = self.glcm_all_agls(img, 1)
        #
        # symm_matriks = [self.symmetric_matriks(i) for i in glcm_matriks]
        # norm_matriks = [self.normalisasi_matriks(i) for i in symm_matriks]
        #
        # print(f"GLCM: {self.vectorized_glcm(img, 1, 1)}")
        # print(f"symm: {symm_matriks[0]}")
        # print(f"norm: {norm_matriks[0]}")

        # print(f"dissiilarity: {self.homogeneity(norm_matriks[0])}")

        # print(f"akhir: {np.shape(norm_matriks)}")
        #
        #
        #
        # print(f"feature: {self.all_props(norm_matriks)}")


        # cv2.imshow("hehe", img)
        # cv2.waitKey(0)


if __name__ == '__main__':
    obj = Gui()
    obj.main()

