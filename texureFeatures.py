# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import measurements
import math
from Utils import normalize
import cv2 as cv
import os
import pandas as pd
from scipy import signal

df = pd.DataFrame(columns=('data_dir',
                           'grades',
                           'uniformity',
                           'entropy',
                           'dissimilarity',
                           'contrast',
                           'homogeneity',
                           'inverse_difference_moment',
                           'maximum_probability',
                           'small_area_emp',
                           'large_area_emp',
                           'low_intensity_emp',
                           'high_intensity_emp',
                           'intensity_variability',
                           'size_zone_variability',
                           'zone_percentage',
                           'low_intensity_small_area_emp',
                           'high_intensity_small_area_emp',
                           'low_intensity_large_area_emp',
                           'high_intensity_large_area_emp',
                           'coarseness',
                           'busyness',
                           'complexity',
                           'strength',
                           ))

class TextureFeatures:
    """
    Gray-Level Co-occurrence Matrix
    """

    def __init__(self, img, name, grade_small, theta=[0, 1], level_min=1, level_max=256, threshold=None):
        """
        initialize
        :param img: normalized image
        :param theta: definition of neighbor
        :param level_min: min intensity of normalized image
        :param level_max: max intensity of normalized image
        :param threshold: threshold of the minimal value
        """

        assert len(img.shape) == 2, 'image must be 2D'
        self.img, self.slope, self.intercept = \
            normalize(img, level_min, level_max, threshold)
        self.name = name
        self.n_level = (level_max - level_min) + 1
        self.level_min = level_min
        self.level_max = level_max
        self.theta = theta
        self.matrixGLCM = self._construct_matrixGLCM()

        self.matrixGLSZM, self.zone_sizes = self._construct_matrixGLSZM()
        self.d = 1
        self.s, self.p, self.ng, self.n2 = self._construct_matrixNGTDM()
        self.grade = grade_small
        self.features = self.calc_features()

    def calc_features(self):
        """
        calculate feature values
        :return: feature values
        """

        global df
        I, J = np.ogrid[self.level_min:self.level_max + 1,
               self.level_min:self.level_max + 1]
        matGLCM = np.array(self.matrixGLCM)
        uniformity = (matGLCM ** 2).sum()
        entropy = -(matGLCM[matGLCM > 0] * np.log(matGLCM[matGLCM > 0])).sum()
        dissimilarity = (matGLCM * np.abs(I - J)).sum()
        contrast = (matGLCM * ((I - J) ** 2)).sum()
        homogeneity = (matGLCM / (1 + np.abs(I - J))).sum()
        inverse_difference_moment = (matGLCM / (1 + (I - J) ** 2)).sum()
        maximum_probability = matGLCM.max() * 100

        matGLSZM = self.matrixGLSZM
        zone_sizes = self.zone_sizes
        omega = matGLSZM.flatten().sum()
        min_size = zone_sizes.min()
        max_size = zone_sizes.max()
        j = np.array(range(min_size, max_size + 1))[np.newaxis, :]
        j = np.vstack((j,) * matGLSZM.shape[0])
        i = np.array(range(self.level_min, self.level_max + 1))[:, np.newaxis]
        i = np.hstack((i,) * matGLSZM.shape[1])
        small_area_emp = (matGLSZM / (j ** 2)).sum() / omega
        large_area_emp = (matGLSZM * (j ** 2)).sum() / omega
        low_intensity_emp = (matGLSZM / (i ** 2)).sum() / omega
        high_intensity_emp = (matGLSZM * (i ** 2)).sum() / omega
        intensity_variability = ((matGLSZM / (i ** 2)).sum(axis=1) ** 2).sum() / omega
        size_zone_variability = ((matGLSZM / (j ** 2)).sum(axis=0) ** 2).sum() / omega  # ?
        zone_percentage = omega / (matGLSZM * (j ** 2)).sum()
        low_intensity_small_area_emp = (matGLSZM / (i ** 2) / (j ** 2)).sum() / omega
        high_intensity_small_area_emp = (matGLSZM * (i ** 2) * (j ** 2)).sum() / omega
        low_intensity_large_area_emp = (matGLSZM * (j ** 2) / (i ** 2)).sum() / omega
        high_intensity_large_area_emp = (matGLSZM * (i ** 2) / (j ** 2)).sum() / omega

        I, J = np.ogrid[self.level_min:self.level_max + 1,
               self.level_min:self.level_max + 1]
        pi = np.hstack((self.p[:, np.newaxis],) * len(self.p))
        pj = np.vstack((self.p[np.newaxis, :],) * len(self.p))

        ipi = np.hstack(
            ((self.p * np.arange(1, len(self.p) + 1))[:, np.newaxis],) * len(self.p))
        jpj = np.vstack(
            ((self.p * np.arange(1, len(self.p) + 1))[np.newaxis, :],) * len(self.p))
        pisi = pi * np.hstack((self.s[:, np.newaxis],) * len(self.p))
        pjsj = pj * np.vstack((self.s[np.newaxis, :],) * len(self.p))
        fcos = 1.0 / (1e-6 + (self.p * self.s).sum())
        # fcon = 1.0 / (self.ng * (self.ng - 1)) * (pi * pj * (I - J) ** 2).sum() * (self.s.sum() / self.n2)
        mask1 = np.logical_and(pi > 0, pj > 0)
        mask2 = self.p > 0
        if (np.abs(ipi[mask1] - jpj[mask1])).sum() == 0:
            fbus = np.inf
        else:
            fbus = (self.p * self.s)[mask2].sum() / (np.abs(ipi[mask1] - jpj[mask1])).sum()
        fcom = (np.abs(I - J)[mask1] / (self.n2 * (pi + pj)[mask1]) * (pisi + pjsj)[mask1]).sum()
        fstr = ((pi + pj) * (I - J) ** 2).sum() / (1e-6 + self.s.sum())

        angular_second_moment = 0
        name = self.name
        grade = self.grade
        df = df.append({'data_dir': name,
                        'grades': grade,
                        'uniformity': uniformity,
                        'entropy': entropy,
                        'dissimilarity': dissimilarity,
                        'contrast': contrast,
                        'homogeneity': homogeneity,
                        'inverse_difference_moment': inverse_difference_moment,
                        'maximum_probability': maximum_probability,

                        'small_area_emp': small_area_emp,
                        'large_area_emp': large_area_emp,
                        'low_intensity_emp': low_intensity_emp,
                        'high_intensity_emp': high_intensity_emp,
                        'intensity_variability': intensity_variability,
                        'size_zone_variability': size_zone_variability,
                        'zone_percentage': zone_percentage,
                        'low_intensity_small_area_emp': low_intensity_small_area_emp,
                        'high_intensity_small_area_emp': high_intensity_small_area_emp,
                        'low_intensity_large_area_emp': low_intensity_large_area_emp,
                        'high_intensity_large_area_emp': high_intensity_large_area_emp,

                        'coarseness': fcos,
                        'busyness': fbus,
                        'complexity': fcom,
                        'strength': fstr,
                        }, ignore_index=True)
        df.to_csv('GLCM.csv', index=False)
        return df

    def _construct_matrixGLCM(self):
        """
        construct GLC-Matrix
        :return: GLC-Matrix
        """

        mat = np.zeros((self.n_level, self.n_level)).astype(np.float64)
        unique = np.unique(self.img)
        width = self.img.shape[1]
        height = self.img.shape[0]

        for uni in unique:
            if uni < self.level_min:
                continue
            indices = np.argwhere(self.img == uni)
            for idx in indices:
                pos = np.array(idx + self.theta, dtype=np.int32)
                if 0 <= pos[0] < height and 0 <= pos[1] < width:
                    neighbor_value = self.img[pos[0], pos[1]]
                    if neighbor_value >= self.level_min:
                        mat[self.img[idx[0], idx[1]] - self.level_min,
                            self.img[pos[0], pos[1]] - self.level_min] += 1
                pos = idx + (self.theta * -1)
                if 0 <= pos[0] < height and 0 <= pos[1] < width:
                    neighbor_value = self.img[pos[0], pos[1]]
                    if neighbor_value >= self.level_min:
                        mat[self.img[idx[0], idx[1]] - self.level_min,
                            self.img[pos[0], pos[1]] - self.level_min] += 1
        sum = mat.sum()
        for i in range(0, mat.shape[0]):
            for j in range(0, mat.shape[1]):
                mat[i][j] /= sum
        return mat

    def _construct_matrixGLSZM(self):
        """
        construct GLSZ-Matrix
        :return: GLSZ-Matrix
        """
        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        elements = []
        for i in range(self.level_min, self.level_max):
            assert i >= 0, 'level mast be positive value or 0.'
            tmp_img = np.array(self.img)
            tmp_img = (tmp_img == i)
            labeled_array, num_features = measurements.label(tmp_img,
                                                             structure=s)
            for label in range(1, num_features + 1):
                size = (labeled_array.flatten() == label).sum()
                elements.append([i, size])

        elements = np.array(elements)
        min_element_size = elements[:, 1].min()
        rows = (self.level_max - self.level_min) + 1
        cols = elements[:, 1].max() - min_element_size + 1
        mat = np.zeros((rows, cols), dtype=np.float64)
        zone_sizes = np.unique(elements[:, 1])
        for element in elements:
            mat[element[0], element[1] - min_element_size] += 1

        return mat, zone_sizes

    def _construct_matrixNGTDM(self):
        """
        construct NGTD-Matrix
        :return: NGTD-Matrix
        """

        assert self.d > 0, 'd must be grater than 1'
        assert self.level_min > 0, 'lower level must be greater than 0'
        # w = (2 * self.d + 1)**2
        kernel = np.ones((2 * self.d + 1, 2 * self.d + 1))
        kernel[self.d, self.d] = 0
        h, w = self.img.shape
        A = signal.convolve2d(self.img, kernel, mode='valid')
        A *= (1 / ((2 * self.d + 1) ** 2 - 1))
        s = np.zeros(self.n_level)
        p = np.zeros_like(s)
        crop_img = np.array(self.img[self.d:h - self.d, self.d:w - self.d])
        for i in range(self.level_min, self.level_max + 1):
            indices = np.argwhere(crop_img == i)
            s[i - self.level_min] = np.abs(i - A[indices[:, 0], indices[:, 1]]).sum()
            p[i - self.level_min] = float(len(indices)) / np.prod(crop_img.shape)
        ng = np.sum(np.unique(crop_img) >= 0)
        n2 = np.prod(crop_img.shape)
        return s, p, ng, n2


def compress(src, max):
    cnt = math.ceil(256 / max)
    bins = np.array(range(max, 257, cnt))
    #   bins = np.array([0,26,52,78,104,130,156,182,208])
    x = np.digitize(src, bins)
    return x

def increaseContrast(img):
    height, width = img.shape
    enhanced_image = np.zeros(img.shape, dtype=np.float64)
    original_alpha = 1.5
    original_pixel = 30
    for r in range(height):
        for c in range(width):
            B = img[r, c]
            enhanced_image[r, c] = B * original_alpha + original_pixel

    enhanced_image[enhanced_image > 255] = 255  # 对大于255的值截断为255
    enhanced_image = np.round(enhanced_image)  # 取整
    enhanced_image = enhanced_image.astype(np.uint8)  # 最后转化为0~255之间
    return enhanced_image

if __name__ == '__main__':

    csv_file = pd.read_csv('C_cataract_data.csv')

    path = 'D:/PycharmProjects/fairness/pics/cataract/C_cataract_one_file/'
    for i in range(0, len(csv_file)):
        img_name = csv_file['data_dir'][i]
        img_path = path + img_name
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        # resize image to 128*128
        img = cv.resize(img, (128, 128))
        grade = csv_file['label'][i]
        img = increaseContrast(img)
        img = compress(img, 16)
        texture_features = TextureFeatures(img, img_name, grade, theta=np.array([0, 1]), level_min=1, level_max=16,
                                           threshold=None)
        print(img_path)


df.to_csv('C_cataract_data_texture.csv', index=False)