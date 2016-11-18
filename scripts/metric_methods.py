import numpy as np


def euclidean_dist(col_one, col_two):
    return np.sqrt((col_one[0] - col_two[0])**2 + (col_one[1] - col_two[1])**2 + (col_one[2] - col_two[2])**2)