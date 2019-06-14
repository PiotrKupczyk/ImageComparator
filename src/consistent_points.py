import numpy as np


def find_nearest_neighbours(single_features, all_features):
    all_neighbours_distances = np.linalg.norm(single_features - all_features, axis=1)
    nearest_neighbour_index = all_neighbours_distances.argmin()
    return nearest_neighbour_index, all_features[nearest_neighbour_index], np.argsort(all_neighbours_distances)


# returns index of neighbour, it's value, and sorted distances to all neighbours


def find_consistent_points(key_points_coordinates,
                           min_compatibility,
                           number_of_neighbours):
    result = []
    first_point_neighbours = np.array(list(map(lambda key_point:
                                               find_nearest_neighbours(key_point, key_points_coordinates[0])[2][
                                               0:number_of_neighbours],
                                               key_points_coordinates[0].tolist())))

    second_point_neighbours = np.array(list(map(lambda key_point:
                                                find_nearest_neighbours(key_point, key_points_coordinates[1])[2][
                                                0:number_of_neighbours],
                                                key_points_coordinates[1].tolist())))

    for i in range(key_points_coordinates.shape[1]):
        if np.intersect1d(first_point_neighbours[i], second_point_neighbours[1]).size >= min_compatibility:
            result.append(key_points_coordinates[:, i])

    return np.array(result)
