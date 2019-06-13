import numpy as np
# params is shape nx1
def affine_transformation(cords_pairs):

    x1, y1 = cords_pairs[0][0]
    x2, y2 = cords_pairs[0][1]
    x3, y3 = cords_pairs[0][2]
    u1, v1 = cords_pairs[1][0]
    u2, v2 = cords_pairs[1][1]
    u3, v3 = cords_pairs[1][2]

    matrix = np.array([
        [x1, y1, 1, 0, 0, 0],
        [x2, y2, 1, 0, 0, 0],
        [x3, y3, 1, 0, 0, 0],
        [0, 0, 0, x1, y1, 1],
        [0, 0, 0, x2, y2, 1],
        [0, 0, 0, x3 ,y3, 1]
    ])
    if np.linalg.det(matrix) != 0:
        res_params = np.linalg.inv(matrix) @ np.array([u1, u2, u3, v1, v2, v3])
        A = np.array([
            [res_params[0], res_params[1], res_params[2]],
            [res_params[3], res_params[4], res_params[5]],
            [0, 0, 1]])
    else:
        A = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1]])
    return A
