import cv2
import numpy as np

from manim import *
from lie_group import *
from matrix import center_homography


def pad_image(image: np.array, pad_scale: float) -> np.array:
    assert pad_scale >= 0
    h, w, c = image.shape

    dw = int(w * (1 + pad_scale))
    dh = int(h * (1 + pad_scale))

    x = (dw - w) // 2
    y = (dh - h) // 2

    base_image = np.zeros((dh, dw, c), dtype=np.uint8)
    base_image[y : y + h, x : x + w, :] = image

    return base_image


class LieGroupAnimation:
    def __init__(
        self,
        group,
        user_image: np.array,
        Hs: List[np.array],
        text_color: str = WHITE,
        **kwargs
    ):
        h, w = user_image.shape[:2]

        def update_image(obj, alpha: float):
            n = len(Hs)
            f = n - 2 if alpha == 1 else alpha * (n - 1)
            i = int(f)
            alpha = f - i

            H = LieGroup.interpolate(Hs[i], Hs[i + 1], alpha)
            obj.become(
                ImageMobject(cv2.warpPerspective(self.user_image, H, (w, h))),
                match_height=True,
                match_width=True,
                match_depth=True,
                match_center=True,
                stretch=True,
            )

        self.user_image = user_image
        self.label = Text(group.__name__.upper(), color=text_color, font_size=40)
        self.image = ImageMobject(user_image)
        self.animation = UpdateFromAlphaFunc(self.image, update_image, **kwargs)


class ImageAnimation(Scene):
    def construct(self):
        def noise(sigma: float):
            return Gaussian(0, sigma)

        gaussians = {}
        gaussians["Euclidean"] = np.array([noise(0.5), noise(50), noise(50)])
        gaussians["Similarity"] = np.array([noise(0), noise(0), noise(0), noise(0.2)])
        gaussians["Affine"] = np.array(
            [noise(0), noise(0), noise(0), noise(0), noise(0.1), noise(0.1)]
        )
        gaussians["Homography"] = np.array(
            [
                noise(0),
                noise(0),
                noise(0),
                noise(0),
                noise(0),
                noise(0),
                noise(0.0005),
                noise(0.0005),
            ]
        )

        colors = {
            "Euclidean": GREEN,
            "Similarity": BLUE,
            "Affine": ORANGE,
            "Homography": PURPLE,
        }

        user_image = pad_image(cv2.imread("assets/checkerboard.jpg"), 0.25)
        np.random.seed(1234)
        w, h = user_image.shape[:2]
        lgas = []
        animations = []
        for group_name, gs in gaussians.items():
            group = globals()[group_name]
            N = len(group.generators())
            Hs = [np.identity(3)]
            for i in range(3):
                Hs.append(
                    center_homography(LieGroup.random(group, gs[:N]), w / 2, h / 2)
                )

            i = len(Hs) - 2
            while i >= 0:
                Hs.append(Hs[i])
                i -= 1

            lga = LieGroupAnimation(
                group, user_image, Hs, colors[group_name], run_time=5
            )
            self.add(lga.image, lga.label)
            lgas.append(lga)
            animations.append(lga.animation)

        subgroups = []
        for lga in lgas:
            subgroups.append(Group(lga.label, lga.image).arrange(DOWN, buff=0.3))
        group = Group(*subgroups).arrange(RIGHT, buff=1.0)
        group.scale(0.5)

        self.add(group)
        self.play(*animations)
