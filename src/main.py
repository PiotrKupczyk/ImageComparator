# from .image_extractor import extract_features
import numpy as np
from PIL import Image, ImageDraw
import consistent_points as cp
import transformations_models as tm
import ransac


def extract_files(file_path):
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


def find_key_points(image1_features, image2_features):
    image1_neighbours = np.array(list(map(lambda image1_feature:
                                          cp.find_nearest_neighbours(image1_feature, image2_features)[0],
                                          image1_features)))

    image2_neighbours = np.array(list(map(lambda image2_feature:
                                          cp.find_nearest_neighbours(image2_feature, image1_features)[0],
                                          image2_features)))

    enumerated = list(enumerate(image1_neighbours))  # (index, neighbour_index)

    indexes = np.array(list(filter(lambda img1_neighbour_index:
                                   img1_neighbour_index[0] == image2_neighbours[img1_neighbour_index[1]],
                                   enumerated)))
    return np.array([image1_features[indexes[:, 0]], image2_features[indexes[:, 1]]]), indexes


def draw_lines(coordinates_img1, coordinates_img2, A = None):
    img1 = Image.open(first_image_path)
    img2 = Image.open(second_image_path)
    # img2 = img2.transform(img1.size, Image.AFFINE, A[0:2].flatten())

    concat = Image.fromarray(np.hstack((np.array(img1), np.array(img2))))
    d = ImageDraw.Draw(concat)
    for i in range(0, len(coordinates_img2)):
        d.line([coordinates_img1[i][0],
                coordinates_img1[i][1],
                coordinates_img2[i][0] + img1.size[0],
                coordinates_img2[i][1]])
    del d
    concat.show()


def error_fn(pairs, transformed_pairs, min_inliners):
    res = np.linalg.norm(pairs - transformed_pairs, axis=1)
    return np.nonzero(res <= min_inliners)[0]


def transformation_fn(points, transformation_params):
    A = tm.affine_transformation(transformation_params)
    return A, (A @ np.insert(points.transpose(), 2, 0, axis=0))[0:2].transpose()


first_image_path = '../files/extracted/img1.png'
second_image_path = '../files/extracted/img2.png'

if __name__ == '__main__':
    coordinates_image1, features_image1 = extract_files(first_image_path + '.haraff.sift')
    coordinates_image2, features_image2 = extract_files(second_image_path + '.haraff.sift')
    # swap elements to prevent IndexError
    # always image1 will be the one with more points
    if coordinates_image1.size < coordinates_image2.size:
        temp = coordinates_image1, features_image1
        coordinates_image1, features_image1 = coordinates_image2, features_image2
        coordinates_image2, features_image2 = temp
        temp = first_image_path
        first_image_path = second_image_path
        second_image_path = temp

    key_points_features, key_points_indexes = find_key_points(features_image1, features_image2)
    key_points_coordinates = np.array([
        coordinates_image1[key_points_indexes[:, 0]], coordinates_image2[key_points_indexes[:, 1]]])

    # draw before algorithms
    draw_lines(
        key_points_coordinates[0],
        key_points_coordinates[1]
    )


    number_of_neighbours = int(15 * np.ceil(key_points_features.shape[0] / 100))  # 5% of all key points
    min_compability = int(np.ceil(30 / 100 * number_of_neighbours))  # 70% of neigbours must be compatible


    # calculate ransac
    best_model, transformed_points = ransac.ransac(key_points_coordinates, transformation_fn, error_fn)

    draw_lines(
        transformed_points[0].tolist(),
        transformed_points[1].tolist(),
        best_model
    )


