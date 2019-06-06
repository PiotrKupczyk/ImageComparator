# from .image_extractor import extract_features
import numpy as np


def extract_features(file_path):
    file = open(file_path, 'r')

    coordinates = []
    features = []
    counter = 0
    for line in file:
        if counter >= 2:
            elements = line.split(' ')
            coordinates.append(np.array(elements[0:2], dtype=np.float))
            features.append(np.array(elements[5:], dtype=np.int))
        counter += 1
    file.close()

    return np.array(coordinates), np.array(features)


def find_nearest_neighbours(single_features, all_features):
    all_neighbours_distances = np.linalg.norm(single_features - all_features, axis=1)
    nearest_neighbour_index = all_neighbours_distances.argmin()
    return nearest_neighbour_index, all_features[nearest_neighbour_index], np.argsort(all_neighbours_distances)


# returns index of neighbour, it's value, and sorted distances to all neighbours


def find_key_points(image1_features, image2_features):
    temp_neighbours = np.zeros(shape=[image1_features.shape[0], ])
    feature_iterator = 0
    for feature in image1_features:
        neighbour_index, _, _ = find_nearest_neighbours(feature, image2_features)
        temp_neighbours[feature_iterator] = neighbour_index
        feature_iterator += 1

    image1_neighbours = dict(zip(range(0, temp_neighbours.size), temp_neighbours))

    values = []
    indexes = []
    image1_to_image2 = []
    image2_to_image1 = []

    feature_iterator = 0
    for feature in image2_features:
        neighbour_index, _, _ = find_nearest_neighbours(feature, image1_features)
        if neighbour_index in image1_neighbours:
            if image1_neighbours[neighbour_index] == feature_iterator:
                indexes.append([neighbour_index, feature_iterator])
                values.append([image1_features[neighbour_index], image2_features[feature_iterator]])
        feature_iterator += 1

    return np.array(values), np.array(indexes)


def find_consistent_points(key_points_features, key_point_indexes, key_point_of_first_image, key_point_of_second_image,
                           n):
    old_shape = key_points_features.shape
    image1_consistent_points = find_nearest_neighbours(key_point_of_first_image,
                                                       key_points_features.reshape(
                                                           (old_shape[1], old_shape[0], old_shape[2]))[0]
                                                       )[2][0:n]

    image2_consistent_points = find_nearest_neighbours(key_point_of_second_image,
                                                       key_points_features.reshape(
                                                           (old_shape[1], old_shape[0], old_shape[2]))[0]
                                                       )[2][0:n]

    some_list = key_points_indexes.tolist()
    other_list = list(key_points_indexes)
    return np.array(
        list(filter(lambda pair: pair[0] in image1_consistent_points and pair[1] in image2_consistent_points,
                    key_points_indexes)))


def get_most_center_point(image_cordinates):
    center_point = np.mean(image_cordinates, axis=0)
    return find_nearest_neighbours(center_point, image_cordinates)


def draw_lines():
    blabla = ''


if __name__ == '__main__':
    # array = np.arange(10*2*15).reshape((10, 2, 15)).reshape((2, 10, 15))
    coordinates_image1, features_image1 = extract_features('../files/image1_features.haraff.sift')
    coordinates_image2, features_image2 = extract_features('../files/image2_features.haraff.sift')
    key_points_features, key_points_indexes = find_key_points(features_image1, features_image2)
    img1_center_index = get_most_center_point(coordinates_image1)[0]
    img2_center_index = get_most_center_point(coordinates_image2)[0]
    consistent_points = find_consistent_points(key_points_features,
                                               key_points_indexes,
                                               features_image1[img1_center_index],
                                               features_image2[img2_center_index],
                                               int(key_points_features.shape[0] / 2))

