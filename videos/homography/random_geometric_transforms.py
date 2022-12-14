import cv2
import numpy as np
import os
import click

from matrix import center_homography
from probability import Gaussian
from lie_group import *


@click.command()
@click.option(
    "--group-name",
    type=click.Choice(["Euclidean", "Similarity", "Affine", "Homography"]),
    help="The Lie Group to visualize.",
    required=True,
)
def run(group_name: str):
    def noise(sigma: float):
        return Gaussian(0, sigma)

    gaussians = {}
    gaussians["Euclidean"] = np.array([noise(1), noise(50), noise(50)])
    gaussians["Similarity"] = np.array([noise(1), noise(50), noise(50), noise(0.3)])
    gaussians["Affine"] = np.array(
        [noise(1), noise(50), noise(50), noise(0.1), noise(0.15), noise(0.15)]
    )
    gaussians["Homography"] = np.array(
        [
            noise(1),
            noise(50),
            noise(50),
            noise(0.1),
            noise(0.1),
            noise(0.1),
            noise(0.0001),
            noise(0.0001),
        ]
    )

    group = globals()[group_name]

    image_path = os.path.join("assets", f"{group_name.lower()}.png")
    image = cv2.imread(image_path)
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_CUBIC)

    h, w = image.shape[:2]
    H = np.identity(3)
    running = True
    while running:
        H_next = LieGroup.random(group, gaussians.get(group_name))

        for t in np.linspace(0, 1, num=60, endpoint=True):
            H_increment = LieGroup.interpolate(H, H_next, t)
            image_warped = cv2.warpPerspective(
                image,
                center_homography(H_increment, w / 2, h / 2),
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            cv2.imshow(f"Lie Group: {group_name}", image_warped)

            # Close the program by pressing 'q' on the keyboard
            if cv2.waitKey(1) == ord("q"):
                running = False
                break

        H = H_next


if __name__ == "__main__":
    run()
