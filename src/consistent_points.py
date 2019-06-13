import numpy as np


def find_nearest_neighbours(single_features, all_features):
    all_neighbours_distances = np.linalg.norm(single_features - all_features, axis=1)
    nearest_neighbour_index = all_neighbours_distances.argmin()
    return nearest_neighbour_index, all_features[nearest_neighbour_index], np.argsort(all_neighbours_distances)


# returns index of neighbour, it's value, and sorted distances to all neighbours

def find_consistent_points(key_points_features,
                           key_point_indexes,
                           min_compatibility,
                           number_of_neighbours):
    reshaped_features = key_points_features.reshape(
        (key_points_features.shape[1],
         key_points_features.shape[0],
         key_points_features.shape[2]))
    result = []
    iterator = 0
    for key_points_pair in key_points_features:
        first_point_neighbours = find_nearest_neighbours(key_points_pair[0],
                                                         reshaped_features[0]
                                                         )[2][0:number_of_neighbours]
        second_point_neighbours = find_nearest_neighbours(key_points_pair[1],
                                                          reshaped_features[1]
                                                          )[2][0:number_of_neighbours]
        counter = np.intersect1d(first_point_neighbours, second_point_neighbours)
        if np.intersect1d(first_point_neighbours, second_point_neighbours).size > min_compatibility:
            result.append(key_points_features[iterator])
        # for index in first_point_neighbours:
        #     if np.isin(np.int(index), second_point_neighbours):
        #         counter += 1
        # if counter >= min_compatibility:
        #     result.append(key_point_indexes[iterator])
        iterator += 1
    return np.array(result)
