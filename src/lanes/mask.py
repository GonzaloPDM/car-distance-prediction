import cv2
import numpy as np


def get_road_mask(no_perspective_image, left_fit, right_fit, Minv, h_img, img_w):
    """
        Create the mask of the detected road area
    """
    
    # Calculate left and right lanes
    ploty = np.linspace(0, h_img-1, h_img)
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)
    
    # Create empty RGB image to draw the mask
    warp_zero = np.zeros_like(no_perspective_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Set format to points for cv2.fillpoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the mask on the empty image, this will be the mask
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0)) # Verde

    # Apply inverse homography to the road mask
    road_mask = cv2.warpPerspective(color_warp, Minv, (img_w, h_img)) 

    return road_mask
