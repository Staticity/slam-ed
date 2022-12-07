import numpy as np
from scipy.linalg import expm, logm
from typing import List, Tuple

from probability import Gaussian


class Euclidean:
    @staticmethod
    def generators() -> List[np.array]:
        return [
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
        ]


class Similarity:
    @staticmethod
    def generators() -> List[np.array]:
        return Euclidean.generators() + [np.array([[0, 0, 0], [0, 0, 0], [0, 0, -1]])]


class Affine:
    @staticmethod
    def generators() -> List[np.array]:
        return Euclidean.generators() + [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        ]


class Homography:
    @staticmethod
    def generators() -> List[np.array]:
        return Euclidean.generators() + [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        ]


class LieGroup:
    @staticmethod
    def exp(group, v: np.array):
        gens = group.generators()
        N = len(gens)
        assert len(v) == N
        return np.real(expm(sum(v[i] * gens[i] for i in range(N))))

    @staticmethod
    def interpolate(A, B, t) -> np.array:
        Ainv = np.linalg.inv(A)
        return A.dot(expm(np.real(logm(Ainv.dot(B)) * t)))

    @staticmethod
    def random(group, gaussians: List[Gaussian] = None) -> np.array:
        gens = group.generators()
        N = len(gens)
        if gaussians is None:
            r = np.random.normal(0, 1, N)
        else:
            assert len(gaussians) == N
            r = np.array([g.sample() for g in gaussians])

        return np.real(expm(sum(r[i] * gens[i] for i in range(N))))
