import click
import cv2
import numpy as np
import os

from lie_group import *
from matrix import center_homography


@click.command()
@click.option("--user-image", help="Optional override image for visualization")
def run(user_image: str = None):
    group_names = ["Euclidean", "Similarity", "Affine", "Homography"]
    images = {}
    if user_image is None:
        for group_name in group_names:
            image_path = os.path.join("assets", f"{group_name.lower()}.png")
            image = cv2.imread(image_path)
            images[group_name] = image
    else:
        user_image = cv2.imread(user_image)

    window_name = f"Lie Group"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    ui_window_name = "User Interface"
    cv2.namedWindow(ui_window_name, cv2.WINDOW_AUTOSIZE)

    v = np.zeros((8, 1))
    max_trackbar_value = 10000
    default_trackbar_value = max_trackbar_value // 2

    generator_scales = [3e-3, 1, 1, 1e-3, 1e-3, 1e-3, 3e-6, 3e-6]

    def update_image():
        group_idx = cv2.getTrackbarPos("group", ui_window_name)
        for i in range(8):
            x = cv2.getTrackbarPos(f"x{i}", ui_window_name) - default_trackbar_value
            v[i] = generator_scales[i] * x

        group_name = group_names[group_idx]
        cv2.setWindowTitle(window_name, group_name)
        image = user_image if user_image is not None else images.get(group_name)
        h, w = image.shape[:2]
        group = globals()[group_name]
        N = len(group.generators())
        H = LieGroup.exp(group, v[:N])
        image_warped = cv2.warpPerspective(
            image,
            center_homography(H, w / 2, h / 2),
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # Show the image warped by the homography
        cv2.imshow(window_name, image_warped)

    def reset_trackbars():
        for i in range(8):
            cv2.setTrackbarPos(f"x{i}", ui_window_name, default_trackbar_value)

    def go_to_next_group():
        reset_trackbars()
        update_image()

    # Create a trackbar to swap between Lie groups
    cv2.createTrackbar("group", ui_window_name, 3, 3, lambda x: go_to_next_group())

    # Create trackbars for each parameter
    for i in range(8):
        cv2.createTrackbar(
            f"x{i}",
            ui_window_name,
            default_trackbar_value,
            max_trackbar_value,
            lambda x: update_image(),
        )

    # Initialize the window
    update_image()

    while True:
        # Close the program by pressing 'q' on the keyboard
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    run()
