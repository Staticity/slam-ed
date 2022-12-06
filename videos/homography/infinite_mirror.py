import cv2
import cv2.aruco as aruco
import numpy as np


def make_square_image(image):
    """
    Takes a rectangular image and forces the image to be a square
    by cropping out the largest centered-square of the image
    """
    h, w = image.shape[0:2]
    if h != w:
        s = min(h, w)
        dh = abs((h - s) // 2)
        dw = abs((w - s) // 2)
        image = image[dh:h-dh, dw: w-dw, :]
    return image


if __name__ == "__main__":
    # Load the first available camera as a video stream
    cap = cv2.VideoCapture(0)

    # How many recursive images to visualize
    recursive_depth = 3

    # Initialize some variables we need to detect 5x5 ArUco markers
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    aruco_detection_params = cv2.aruco.DetectorParameters_create()

    # Image processing loop
    while True:
        # Load the image from our video camera
        ret, image = cap.read()
        if not ret:
            break

        # Force the image to be square for prettier visualizations
        image = make_square_image(image)
        h, w = image.shape[:2]

        # Convert the image to gray and find the markers
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        quads, ids, rejected_img_points = aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_detection_params)

        # Did we find any markers?
        if quads:
            # Take the first marker and solve for a Homography to warp
            # the original image into the marker's space
            quad = quads[0]
            image_bounds = np.array([(0, 0), (w, 0), (w, h), (0, h)])
            H_dst_src, inliers = cv2.findHomography(image_bounds, quad[0])

        
            # Pre-compute a mask which defines which pixels we need to warp
            mask = cv2.warpPerspective(
                np.ones(image.shape), H_dst_src, (w, h))

            # Warp the image into the marker's bounds multiple times to create
            # an infinite mirror.
            for i in range(recursive_depth):
                warped = cv2.warpPerspective(
                    image, H_dst_src, (w, h)).astype(np.uint8)
                np.putmask(image, mask, warped)

        # Show the image
        cv2.imshow('image', image)

        # Close the program by pressing 'q' on the keyboard
        if cv2.waitKey(1) == ord('q'):
            break
