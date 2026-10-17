import cv2
import numpy as np


def default_src_pts(img_h, img_w):
    """ Trapezoid aligned with the lane lines of a straight road, for the default camera position """
    p1_top_left = (int(img_w * 0.45), int(img_h * 0.63)) 
    p2_top_right = (int(img_w * 0.52), int(img_h * 0.63)) 
    p3_bot_right = (int(img_w * 0.84), int(img_h * 0.99)) 
    p4_bot_left  = (int(img_w * 0.28), int(img_h * 0.99)) 

    return np.float32([p1_top_left, p2_top_right, p3_bot_right, p4_bot_left])


def default_dst_pts(img_h, img_w):
    """ Rectangle where the src_pts trapezoid is mapped in the "bird eye" view """
    return np.float32([(int(img_w*0.4), 0), 
                       (int(img_w*0.6), 0), 
                       (int(img_w*0.6), img_h), 
                       (int(img_w*0.4), img_h)
                      ])


def get_homography(img_h, img_w, src_pts=None, dst_pts=None):
    """
        Calculate the homography that removes the road perspective and its inverse.
        If src_pts or dst_pts are not given, the default ones are used.

    Returns:
        M: The homography matrix (camera -> bird eye)
        Minv: The inverse homography matrix (bird eye -> camera)
    """
    if src_pts is None:
        src_pts = default_src_pts(img_h, img_w)
    if dst_pts is None:
        dst_pts = default_dst_pts(img_h, img_w)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = np.linalg.inv(M)
    return M, Minv


def warp(image, M, img_h, img_w):
    """ Apply the homography M to an image """
    return cv2.warpPerspective(image, M, (img_w, img_h), flags=cv2.INTER_LINEAR)


def remove_road_perspective(image, 
                            img_h, img_w,
                            src_pts, dst_pts):
    """
        This method calculate the homography of the road, given the points of a rectangle aling with straight road lines
    Args:
        img: The road image from the camera perspective
        src_pts: The points of a rectangle align with the road lines in the image from the perspective of the car.
        dst_pts: The points of the src_pts rectanggle, but from a "bird eye" perspective. 

    Returns:
        no_perspective_image: The image with removed perspective, like seen from above
        M: The homography matrix calculated for removing perspective
    """
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    no_perspective_image = cv2.warpPerspective(image, M, (img_w, img_h), flags=cv2.INTER_LINEAR)

    return no_perspective_image, M
