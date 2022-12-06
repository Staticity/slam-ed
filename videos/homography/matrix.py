import numpy as np

def center_homography(H: np.array, x: float, y: float):
    """
    This function will center the transformation to a new origin (x, y) than
    the default (0, 0).
    """
    H_center = np.array([
        [1, 0, -x],
        [0, 1, -y],
        [0, 0, 1]
    ])
    H_center_inv = np.array([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ])

    return np.linalg.multi_dot([H_center_inv, H, H_center]).astype(float)
