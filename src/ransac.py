import numpy as np


def ransac(key_points, transform_fn, evaluate_fn, max_iters=40000, samples_to_fit=3, inlier_threshold=0.1,
           min_inliers=100):
    best_model = None, None
    best_model_performance = 0

    for i in range(max_iters):
        sample = random_partition(samples_to_fit, key_points)
        model_params, transformed_points = transform_fn(key_points[0], sample)
        model_performance = evaluate_fn(key_points[1], transformed_points, min_inliers)

        if model_performance.size > best_model_performance:
            best_model = model_params, np.array([key_points[0][model_performance], key_points[1][model_performance]])
            best_model_performance = model_performance.size

    return best_model


def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    random_indexes = np.arange(0, n_data.shape[1])
    np.random.shuffle(random_indexes)
    return np.array([n_data[0][random_indexes[:n]], n_data[1][random_indexes[:n]]])
