import numpy as np


def extract_features(file_path):
    file = open(file_path, 'r')
    number_of_features = file.readline(0)
    number_of_key_points = file.readline(1)

    for line in file:
        elements = line.split(' ')
        cordinates = np.array(elements[0:1], dtype=np.float)
        features = np.array(elements[5:], dtype=np.float)

    file.close()
