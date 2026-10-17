import cv2
import numpy as np


def calculate_road_lines(binary_warped, debug_image=False):
    """ Calculate the road lines in a edges image with non perspective transformation applied, using sliding windows and polynomial fitting. 
        Then, draw the windows and the lines in an output image.
    
        Args:
            binary_warped: The binary image with the road lines in white
            debug_image: If True, then the debug image is processed and return as out_img
        Returns:
            left_fit: The polynomial coefficients for the left road line
            right_fit: The polynomial coefficients for the right road line
            out_img: The output image with the detected lines and windows drawn
    """  
    img_h, img_w = binary_warped.shape[:2]

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(img_h*0.8):int(img_h*0.98),int(img_w*0.35):int(img_w*0.65)], axis=0)

    # The peak of the left and right halves of the histogram will be the starting point for the left and right lines
    mid_point = histogram.shape[0]//2
    left_peak = np.argmax(histogram[:mid_point]) + int(img_w*0.35)
    right_peak = np.argmax(histogram[mid_point:]) + mid_point + int(img_w*0.35)

    if left_peak == 0 or right_peak == 0:
        return None, None, None

    # Set parameters for sliding windows
    nwindows = 9
    window_height = int(img_h // nwindows)
    margin = img_w // 20
    minpix = 50
    left_current = left_peak 
    right_current = right_peak
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    out_img = None
    if debug_image:
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Slide windows from bottom to top, finding line pixels
    for window in range(nwindows):
        # Define new window
        win_y_low = img_h - window * window_height
        win_y_high = img_h - (window + 1) * window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (binary_warped[win_y_high:win_y_low, win_xleft_low:win_xleft_high]).nonzero()
        good_right_inds = (binary_warped[win_y_high:win_y_low, win_xright_low:win_xright_high]).nonzero()

        # Adjust pixels to image coordinates
        good_left_inds = (good_left_inds[0] + win_y_high, good_left_inds[1] + win_xleft_low)
        good_right_inds = (good_right_inds[0] + win_y_high, good_right_inds[1] + win_xright_low)

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        new_left_current = left_current
        new_right_current = right_current

        # If found > minpix pixels, recenter next window on their mean position (ponderated to center of the image)
        if len(good_left_inds[0]) > minpix:
            new_left_current = int(np.mean(good_left_inds[1]))
        if len(good_right_inds[0]) > minpix:
            new_right_current = int(np.mean(good_right_inds[1]))

        # Check wether boxes colides
        left_box_right_edge = new_left_current + margin
        right_box_left_edge = new_right_current - margin
        buffer = 10 
        
        # If no colision, we let the means, if not, do not renew them
        if right_box_left_edge > (left_box_right_edge + buffer):
            
            left_current = new_left_current
            right_current = new_right_current

        if debug_image:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,
                          (left_current - margin, win_y_high), 
                          (left_current + margin, win_y_low),
                          (0,255,0), 2)
            cv2.rectangle(out_img,
                          (right_current - margin, win_y_high), 
                          (right_current + margin, win_y_low),
                          (0,255,0), 2)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds, axis=1)
    right_lane_inds = np.concatenate(right_lane_inds, axis=1)

    if (len(left_lane_inds[0]) == 0) | (len(right_lane_inds[0]) == 0):
        return None, None, None

    if debug_image:
        # Draw lane pixels
        out_img[left_lane_inds[0], left_lane_inds[1]] = [255, 0, 0]
        out_img[right_lane_inds[0], right_lane_inds[1]] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_lane_inds[0], left_lane_inds[1], 1)
    right_fit = np.polyfit(right_lane_inds[0], right_lane_inds[1], 1)

    return left_fit, right_fit, out_img


def search_around_poly(binary_warped, left_fit, right_fit, margin=50, debug_image=False):
    """
    Busca líneas de carril basándose en los polinomios del frame anterior (ROI Tracking).
    Esto evita usar histogramas y ventanas deslizantes nuevamente.
    """
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Define the search area around the previous polynomials
    left_center_x = np.polyval(left_fit, nonzeroy)
    right_center_x = np.polyval(right_fit, nonzeroy)

    # Define the search area around the previous polynomials
    left_lane_inds = ((nonzerox > (left_center_x - margin)) & 
                      (nonzerox < (left_center_x + margin)))
    
    right_lane_inds = ((nonzerox > (right_center_x - margin)) & 
                       (nonzerox < (right_center_x + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If we don't find enough pixels, we assume detection failed
    min_inds = 50
    if len(leftx) < min_inds or len(rightx) < min_inds:
        return None, None, None

    # Fit new polynomials
    new_left_fit = np.polyfit(lefty, leftx, 1)
    new_right_fit = np.polyfit(righty, rightx, 1)
    
    # Generate x and y values for plotting
    out_img = None
    if debug_image:
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        out_img = out_img.astype(np.uint8)

        # Draw green area where research have been done for each line
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        # Calculate the trajectory of the PREVIOUS polynomial (where we are searching)
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)

        # Create an empty image to draw the transparency
        window_img = np.zeros_like(out_img)

        # Generate polygon points for the left margin
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        
        # Generate polygon points for the right margin
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the tunnels in green on the empty image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        
        # Blend with the original image (Transparency)
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        out_img[lefty, leftx] = [255, 0, 0]   # Red (or Blue in BGR)
        out_img[righty, rightx] = [0, 0, 255] # Blue (or Red in BGR)

    return new_left_fit, new_right_fit, out_img


class LaneTracker:
    """
        Keeps the lane lines found in the previous frame, so that the next search is done around them
        (search_around_poly) instead of a full sliding windows search (calculate_road_lines).
    """
    def __init__(self, margin=50, reset_every=25):
        """
        Args:
            margin: Width of the search area around the previous lines
            reset_every: Every this amount of frames the history is dropped, forcing a full search.
                         None means never reset.
        """
        self.margin = margin
        self.reset_every = reset_every
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.frame_count = 0

    def update(self, binary_warped, debug_image=False):
        """ Find the lane lines of a new bird eye binary frame.

            Returns:
                left_fit, right_fit: The polynomial coefficients (None if not found)
                out_img: The debug image (None if debug_image is False)
        """
        left_fit, right_fit = None, None
        out_img = None

        # A) Try Optimized Search (if we have history)
        if self.prev_left_fit is not None and self.prev_right_fit is not None:
            left_fit, right_fit, out_img = search_around_poly(binary_warped, 
                                                              self.prev_left_fit, 
                                                              self.prev_right_fit, 
                                                              margin=self.margin,
                                                              debug_image=debug_image
                                                              )

        # B) Fallback: Full Sliding Windows Search (if optimization failed)
        if left_fit is None and right_fit is None:
            left_fit, right_fit, out_img = calculate_road_lines(binary_warped, debug_image=debug_image)
            if left_fit is None and self.prev_left_fit is not None:
                left_fit, right_fit = self.prev_left_fit, self.prev_right_fit

        if left_fit is not None and right_fit is not None:
            # Save for next frame
            if self.reset_every and self.frame_count % self.reset_every == 0:
                self.prev_left_fit = None
                self.prev_right_fit = None
            else:
                self.prev_left_fit = left_fit
                self.prev_right_fit = right_fit

        self.frame_count += 1
        return left_fit, right_fit, out_img
