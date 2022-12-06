import click
import cv2
import numpy as np
import os

from lie_group import *


def on_mouse_event(event, x, y, flags, params):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    w, h, points, Hs, inverse = params
    if len(points) >= 4:
        points.clear()
    points.append(np.array((x, y)))

    if len(points) == 4:
        image_bounds = np.array([(0, 0), (w, 0), (w, h), (0, h)])
        H_new, _ = cv2.findHomography(image_bounds, np.array(points))
        if inverse:
            H_new = np.linalg.inv(H_new)
        H_last = Hs[-1] if Hs else np.identity(3)
        for t in np.linspace(0, 1, num=60, endpoint=True):
            Hs.append(LieGroup.interpolate(H_last, H_new, t))

        if inverse:
            points.clear()


@click.command()
@click.option("--image", help="Image to warp")
@click.option("--background", help="Optional background image")
@click.option("--inverse", is_flag=True, default=False)
def run(image: str = None, background: str = None, inverse: bool = False):
    window_name = f"Smooth Homography"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    image = (
        cv2.imread(image)
        if image
        else cv2.imread(os.path.join("assets", "checkerboard.jpg"))
    )
    h, w = image.shape[:2]

    has_background = background is not None
    background = (
        cv2.imread(background) if background else np.zeros(image.shape).astype(np.uint8)
    )
    bh, bw = background.shape[:2]

    Hs = [] if has_background else [np.identity(3)]
    points = []
    cv2.setMouseCallback(window_name, on_mouse_event, (w, h, points, Hs, inverse))

    while True:
        render_target = background.copy()

        if Hs:
            H = Hs[0]
            warped_mask = cv2.warpPerspective(np.ones(image.shape), H, (bw, bh))
            warped_image = cv2.warpPerspective(image, H, (bw, bh))
            np.putmask(render_target, warped_mask, warped_image)
            if len(Hs) > 1:
                Hs.pop(0)

        colors = [(35, 59, 194), (39, 154, 243), (60, 192, 3), (190, 154, 87)]

        for i, p in enumerate(points):
            cv2.drawMarker(
                render_target, p, colors[i], cv2.MARKER_CROSS, 15, 1, cv2.LINE_AA
            )

        cv2.imshow(window_name, render_target)

        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    run()
