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

    return coordinates, features


def find_nearest_neighbour(point, coordinates):
    distances = np.zeros(shape=[len(coordinates), ])
    counter = 0
    for coordinate in coordinates:
        distances[counter] = np.hypot(point[0]-coordinate[0], point[1]-coordinate[1])
        counter += 1

    return coordinates[distances.argmin()]


def find_key_points(image1_coordinates, image2_coordinates):
    image1_neighbours = dict()

    for image1_coord in image1_coordinates:
        image1_neighbours.update(
            {str(find_nearest_neighbour(image1_coord, image2_coordinates)): image1_coord}
        )
    result = []

    for image2_cord in image2_coordinates:
        image2_cord_neighbour = find_nearest_neighbour(image2_cord, image1_coordinates)
        if str(image2_cord) in image1_neighbours:
            result.append([image2_cord_neighbour, image2_cord])

    return result

if __name__ == '__main__':
    coordinates_image1, _ = extract_features('../files/image1_features.haraff.sift')
    coordinates_image2, _ = extract_features('../files/image2_features.haraff.sift')
    res = find_key_points(coordinates_image1, coordinates_image2)
    string = ''
