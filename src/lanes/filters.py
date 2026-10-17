import cv2
import numpy as np


def sobel_filter(image, saturation_thresh=(170, 255), sobelx_thresh=(20, 100), debug_image=False):
    """ Filter the road lines in the image using color and gradient thresholds.
    
        Args:
            image: Input image to be filtered
            saturation_thresh: Saturation threshold
            sobelx_thresh: Sobel x threshold
        Returns: 
            combined_binary: Binary image after applying the filters
    """
    # Transform to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(float)
    h_channel = hls[:,:,0] # Hue channel
    l_channel = hls[:,:,1] # Lightness channel
    s_channel = hls[:,:,2] # Saturation channel

    # Filter white pixels (Hihg luminosity)
    white_mask = np.zeros_like(l_channel)
    white_mask[(l_channel > 220) & (l_channel <= 255)] = 1

    # Filter yellow pixels (High saturation + Hue: 15 - 35)
    yellow_mask = np.zeros_like(s_channel)
    yellow_mask[((h_channel >= 15) & (h_channel <= 35)) & 
                ((s_channel > saturation_thresh[0]) & (s_channel <= saturation_thresh[1]))] = 1
    
    # Sobel X: vertical edges
    l_blur = cv2.GaussianBlur(l_channel, (5,5), 1)
    sobel_x = cv2.Sobel(l_blur, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
    
    sobelx_mask = np.zeros_like(scaled_sobel)
    sobelx_mask[(scaled_sobel >= sobelx_thresh[0]) & (scaled_sobel <= sobelx_thresh[1])] = 1
    
    # Create final mask
    color_binary = None
    if debug_image:
        color_binary = np.dstack((np.zeros_like(sobelx_mask), sobelx_mask, (yellow_mask==1) | (white_mask==1))) * 255
    
    combined_binary = np.zeros_like(sobelx_mask)
    combined_binary[(yellow_mask == 1) | (white_mask == 1) | (sobelx_mask == 1)] = 1
    
    return combined_binary, color_binary


def canny_filter(image, lower_threshold=5, upper_threshold=250):
    """ Apply Canny algorithm to an image.   
    
        Args:
            image: Input image to be binarized
            lower_threshold: bottom value for hysteresis
            upper_threshold: top value for hysteresis
        Returns: 
            canny: Canny output image
    """  
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray, (7,7), 1)
    canny = cv2.Canny(blurred_img,lower_threshold,upper_threshold)
    
    return canny


LANE_FILTER_MODES = ('sobel', 'canny')


def filter_lane_pixels(image, mode='sobel'):
    """ Return the binary image with the potential lane pixels, using the chosen filter.

        Args:
            image: Input image
            mode: 'sobel' (color + gradient thresholds) or 'canny' (edges)
        Returns:
            binary: Binary image where the non zero pixels are lane candidates
    """
    if mode == 'sobel':
        binary, _ = sobel_filter(image)
        return binary
    if mode == 'canny':
        return canny_filter(image)
    raise ValueError(f"Unknown lane filter mode: {mode}. Use one of {LANE_FILTER_MODES}")
